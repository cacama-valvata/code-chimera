"""Utilities related to configuration and handling of framework logging.
"""
import os
import sys
import argparse
import datetime
import io
import subprocess
import signal
import traceback

from . import exceptions

import logging
import logging.config
import logging.handlers
_log = logging.getLogger(__name__)

class MultiFlushMemoryHandler(logging.handlers.MemoryHandler):
    """Subclass :py:class:`logging.handlers.MemoryHandler` to enable flushing
    the contents of its log buffer to multiple targets. We do this to solve the 
    chicken-and-egg problem of logging any events that happen before the log 
    outputs are configured: those events are captured by an instance of this 
    handler and then transfer()'ed to other handlers once they're set up.
    See `<https://stackoverflow.com/a/12896092>`__.
    """

    def transfer(self, target_handler):
        """Transfer contents of buffer to target_handler.
        
        Args:
            target_handler (:py:class:`logging.Handler`): log handler to transfer
                contents of buffer to.
        """
        self.acquire()
        try:
            self.setTarget(target_handler)
            if self.target:
                for record in self.buffer:
                    if self.target.level <= record.levelno:
                        self.target.handle(record)
                # self.buffer = [] # don't clear buffer!
        finally:
            self.release()

    def transfer_to_non_streams(self, logger):
        """Transfer contents of buffer to all non-console-based handlers attached 
        to logger (handlers that aren't :py:class:`~logging.StreamHandler`.)

        If no handlers are attached to the logger, a warning is printed and the
        buffer is transferred to the :py:data:`logging.lastResort` handler, i.e.
        printed to stderr.
        
        Args:
            logger (:py:class:`logging.Logger`): logger to transfer
                contents of buffer to.
        """
        no_transfer_flag = True
        for h in logger.handlers:
            if not isinstance(h, MultiFlushMemoryHandler) \
                and not (isinstance(h, logging.StreamHandler) \
                    and not isinstance(h, logging.FileHandler)):
                self.transfer(h)
                no_transfer_flag = False
        if no_transfer_flag:
            logger.warning("No non-console-based loggers configured.")
            self.transfer(logging.lastResort)

class HeaderFileHandler(logging.FileHandler):
    """Subclass :py:class:`logging.FileHandler` to print system information to
    start of file without writing it to other loggers.
    """
    def _log_header(self):
        return ""

    def _open(self):
        """Write header information right after we open the log file, then
        proceed normally.
        """
        fp = super(HeaderFileHandler, self)._open()
        fp.write(self._log_header())
        return fp

class MDTFHeaderFileHandler(HeaderFileHandler):
    def _log_header(self):
        """Returns string of system debug information to use as log file header.
        Calls :func:`git_info`.
        """
        try:
            git_branch, git_hash, _ = git_info()
            str_ = (
                "MDTF PACKAGE LOG\n\n"
                f"Started logging at {datetime.datetime.now()}\n"
                f"git hash/branch: {git_hash} (on {git_branch})\n"
                # f"uncommitted files: {git_dirty}\n"
                f"sys.platform: '{sys.platform}'\n"
                # f"sys.executable: '{sys.executable}'\n"
                f"sys.version: '{sys.version}'\n"
                # f"sys.path: {sys.path}\nsys.argv: {sys.argv}\n"
            ) 
            return str_ + (80 * '-') + '\n\n'
        except Exception:
            err_str = "Couldn't gather log file header information."
            _log.exception(err_str)
            return "ERROR: " + err_str + "\n"

def _hanging_indent(str_, initial_indent, subsequent_indent):
    """Poor man's indenter. Easier than using textwrap for this case.

    Args:
        str_ (str): String to be indented.
        initial_indent (str): string to insert as the indent for the first 
            line.
        subsequent_indent (str): string to insert as the indent for all 
            subsequent lines.

    Returns:
        Indented string.
    """
    lines_ = str_.splitlines()
    lines_out = []
    if len(lines_) > 0:
        lines_out = lines_out + [initial_indent + lines_[0]]
    if len(lines_) > 1:
        lines_out = lines_out + [(subsequent_indent+l) for l in lines_[1:]]
    return '\n'.join(lines_out)

class HangingIndentFormatter(logging.Formatter):
    """:py:class:`logging.Formatter` that applies a hanging indent, making it 
    easier to tell where one entry stops and the next starts.
    """
    # https://blog.belgoat.com/python-textwrap-wrap-your-text-to-terminal-size/
    def __init__(self, fmt=None, datefmt=None, style='%', tabsize=0, header="", footer=""):
        """Initialize formatter with extra arguments.

        Args:
            fmt (str): format string, as in :py:class:`logging.Formatter`.
            datefmt (str): date format string, as in :py:class:`logging.Formatter`
                or `strftime <https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior>`__.
            style (str): string templating style, as in :py:class:`logging.Formatter`.
            tabsize (int): Number of spaces to use for hanging indent.
            header (str): Optional constant string to prepend to each log entry.
            footer (str): Optional constant string to append to each log entry.
        """
        super(HangingIndentFormatter, self).__init__(fmt=fmt, datefmt=datefmt, style=style)
        self.indent = (tabsize * ' ')
        self.stack_indent = self.indent
        self.header = str(header)
        self.footer = str(footer)

    def format(self, record):
        """Format the specified :py:class:`logging.LogRecord` as text, adding
        indentation and header/footer.

        Args:
            record (:py:class:`logging.LogRecord`): logging record to be formatted.

        Returns:
            String representation of the log entry.

        This essentially repeats the method's `implementation 
        <https://github.com/python/cpython/blob/4e02981de0952f54bf87967f8e10d169d6946b40/Lib/logging/__init__.py#L595-L625>`__
        in the python standard library. See comments there and the logging module
        `documentation <https://docs.python.org/3.7/library/logging.html>`__.
        """
        record.message = record.getMessage()
        if self.usesTime():
            record.asctime = self.formatTime(record, self.datefmt)
        s = self.formatMessage(record)
        # indent the text of the log message itself
        s = _hanging_indent(s, '', self.indent)
        if self.header:
            s = self.header + s

        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            # text from formatting the exception. NOTE that this includes the 
            # stack trace without populating stack_info or calling formatStack.
            if not s.endswith('\n'):
                s = s + '\n'
            s = s + _hanging_indent(
                record.exc_text, 
                self.indent, self.stack_indent
            )
        if record.stack_info:
            # stack_info apparently only used if our format string (ie, 'fmt' in
            # init) requests a stack trace?
            if not s.endswith('\n'):
                s = s + '\n'
            s = s + _hanging_indent(
                self.formatStack(record.stack_info),
                self.stack_indent, self.stack_indent
            )
        if self.footer:
            s = s + self.footer
        return s


class GeqLevelFilter(logging.Filter):
    """:py:class:`logging.Filter` to include only log messages with a severity of
    level or worse. This is normally done by setting the level attribute on a
    :py:class:`logging.Handler`, but we need to add a filter when transferring
    records from another logger, as shown in 
    `<https://stackoverflow.com/a/24324246>`__."""
    def __init__(self, name="", level=None):
        super(GeqLevelFilter, self).__init__(name=name)
        if level is None:
            level = logging.NOTSET
        if not isinstance(level, int):
            if hasattr(logging, str(level)):
                level = getattr(logging, str(level))
            else:
                level = int(level)
        self.levelno = level

    def filter(self, record):
        return record.levelno >= self.levelno

class LtLevelFilter(logging.Filter):
    """:py:class:`logging.Filter` to include only log messages with a severity 
    less than level.
    """
    def __init__(self, name="", level=None):
        super(LtLevelFilter, self).__init__(name=name)
        if level is None:
            level = logging.NOTSET
        if not isinstance(level, int):
            if hasattr(logging, str(level)):
                level = getattr(logging, str(level))
            else:
                level = int(level)
        self.levelno = level

    def filter(self, record):
        return record.levelno < self.levelno

class EqLevelFilter(logging.Filter):
    """:py:class:`logging.Filter` to include only log messages with a severity 
    equal to level.
    """
    def __init__(self, name="", level=None):
        super(EqLevelFilter, self).__init__(name=name)
        if level is None:
            level = logging.NOTSET
        if not isinstance(level, int):
            if hasattr(logging, str(level)):
                level = getattr(logging, str(level))
            else:
                level = int(level)
        self.levelno = level

    def filter(self, record):
        return record.levelno == self.levelno

class NameMatchFilter(logging.Filter):
    """:py:class:`logging.Filter` that rejects all log events coming from child
    loggers.

    Intended to be attached to a handler -- the effect of attaching this to a 
    logger is the same as setting *propagate* = False on it.
    """
    def __init__(self, name=""):
        super(NameMatchFilter, self).__init__()
        self._name = name
    
    def filter(self, record):
        return (record.name == self._name)

# ------------------------------------------------------------------------------

# standardize 
OBJ_LOG_ATTR_NAME = 'log'
OBJ_LOG_ROOT = 'MDTF' # "root logger" of the object logger hierarchy

class MDTFObjectLoggerWrapper(logging.Logger):
    """This class wraps functionality for use by :class:`MDTFObjectLoggerMixin`:

    - A :py:class:`~logging.Logger` to record events affecting the parent object
        only. This logger does not propagate events up the log hierarchy: the 
        module-level logger should be passed if that functionality is desired. 

    - A list for holding :py:class:`Exception` objects received by the parent 
        object.
    """
    def __init__(self, name):
        super(MDTFObjectLoggerWrapper, self).__init__(name)
        self._module_log = None
        # init exception recording
        self._exceptions = []

    def log(self, level, msg, *args, **kwargs):
        super(MDTFObjectLoggerWrapper, self).log(level, msg, *args, **kwargs)
        if self._module_log is not None:
            self._module_log.log(level, msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        super(MDTFObjectLoggerWrapper, self).debug(msg, *args, **kwargs)
        if self._module_log is not None:
            self._module_log.debug(msg, *args, **kwargs)
        
    def info(self, msg, *args, **kwargs):
        super(MDTFObjectLoggerWrapper, self).info(msg, *args, **kwargs)
        if self._module_log is not None:
            self._module_log.info(msg, *args, **kwargs)
        
    def warning(self, msg, *args, **kwargs):
        super(MDTFObjectLoggerWrapper, self).warning(msg, *args, **kwargs)
        if self._module_log is not None:
            self._module_log.warning(msg, *args, **kwargs)
        
    def error(self, msg, *args, **kwargs):
        super(MDTFObjectLoggerWrapper, self).error(msg, *args, **kwargs)
        if self._module_log is not None:
            self._module_log.error(msg, *args, **kwargs)
    
    def critical(self, msg, *args, **kwargs):
        super(MDTFObjectLoggerWrapper, self).critical(msg, *args, **kwargs)
        if self._module_log is not None:
            self._module_log.critical(msg, *args, **kwargs)

    @property
    def has_exceptions(self):
        return (len(self._exceptions) == 0)

    def exception(self, msg, *args, exc=None, **kwargs):
        # NB not just wrapping the exception() method here.
        if not msg and exc is not None:
            msg = f"Caught {repr(exc)}."
        if msg and exc is None:
            exc = exceptions.MDTFBaseException(msg)
        
        super(MDTFObjectLoggerWrapper, self).exception(msg, *args, **kwargs)
        wrapped_exc = traceback.TracebackException.from_exception(exc)
        self._exceptions.append(wrapped_exc)

class MDTFObjectLoggerMixin():
    """Class to implement per-object logging via inheritance. Used for 
    :class:`~diagnostic.VarlistEntry` and :class:`~diagnostic.Diagnostic`. 
    Based on `<https://stackoverflow.com/q/57813096>`__.

    This wraps related functionalities:
    
    - A :py:class:`~logging.Logger` to record events affecting the object
        only. Log messages are cached in a :py:class:`~io.StringIO` buffer.

    - A method :meth:`~MDTFObjectLogger.format` for formatting the contents of
        the above into a string, along with log messages of any child objects.
        This is intended for preparing log files (at the per-POD or per-case 
        level); logging intended for the console should use the module loggers.
    """
    def init_log(self, log_format=None, module_logger=None):
        # Mixin class, so no __init__ for simplicity
        if log_format is None:
            log_format = '%(asctime)s in %(funcName)s: %(message)s'
        self._log_buffer = io.StringIO()

        self.log.propagate = True # propagate up to case
        self.log._module_log = module_logger
        self.log.setLevel(logging.DEBUG)
        if self.log.hasHandlers():
            for handler in self.log.handlers:
                self.log.removeHandler(handler)
        handler = logging.StreamHandler(self._log_buffer)
        handler.addFilter(NameMatchFilter(self._log_name)) # don't record events from children
        formatter = logging.Formatter(fmt=log_format, datefmt='%H:%M:%S')
        handler.setFormatter(formatter)
        self.log.addHandler(handler)

    def format_log(self, child_objs=None):
        """Return contents of log buffer, as well as that of any child objects
        in *child_objs*, as a formatted string.
        """
        if child_objs is None:
            child_objs = []
            
        str_ = self.name + ':\n'
        # list exceptions before anything else:
        if self.log.has_exceptions:
            strs_ = [''.join(exc.format()) for exc in self.log._exceptions]
            strs_ = [f"*** {self.name} caught exception #{i+1}:\n{exc}\n" \
                for i, exc in enumerate(strs_)]
            str_ += "".join(strs_) + '\n'
        # then log contents:
        str_ += self._log_buffer.getvalue().rstrip()
        # then contents of children:
        for child in child_objs:
            str_ += '\n'
            if hasattr(child, 'format_log'):
                str_ += child.format_log()
            else:
                str_ += f"<{child} log placeholder>\n"
        return _hanging_indent(str_, '', 4 * ' ')

class MDTFCaseLoggerMixin():
    """Wrapper to handle configuring logging from a file and transferring 
    the temporary log cache to the newly-configured loggers.

    Args:
        config_mgr ( :class:`~core.ConfigManager` ): Passed as an argument 
            here in order to avoid complicating the import structure of the
            'util' subpackage.
        new_paths (dict): Dict of new log file names to assign. Keys are the 
            names of :py:class:`logging.Handler` handlers in the config file, 
            and values are the new paths.
    """
    def init_log(self, config_mgr, module_logger=None, **new_paths):
        # Mixin class, so no __init__ for simplicity
        def _set_log_file_paths(d, new_paths):
            """Assign paths to log files. Paths are assumed to be well-formed 
            and in writeable locations.
            """
            if not new_paths:
                self.log.error("Log file paths not set.")
                return
            handlers = d.setdefault('handlers', dict())
            for h in handlers:
                if h in new_paths:
                    d['handlers'][h]["filename"] = new_paths[h]
                    del new_paths[h]
            if new_paths:
                self.log.warning(("Couldn't find handlers for the following log "
                    "files: %s"), new_paths)

        self.log.propagate = False
        self.log._module_log = module_logger
    
        if config_mgr.log_config is None:
            return

        root_logger = logging.getLogger()
        cache_idx = [i for i,handler in enumerate(root_logger.handlers) \
            if isinstance(handler, MultiFlushMemoryHandler)]
        first_call = len(cache_idx) > 0
        if first_call:
            temp_log_cache = root_logger.handlers[cache_idx[0]]

        # log uncaught exceptions
        _set_excepthook(root_logger)
        # configure loggers from the specification we loaded
        try:
            # set console verbosity level
            log_d = config_mgr.log_config.copy()
            log_d = _configure_logging_dict(log_d, config_mgr)
            _set_log_file_paths(log_d, new_paths)
            logging.config.dictConfig(log_d)
        except Exception:
            self.log.exception("Logging config failed.")

        if first_call:
            # transfer cache contents to newly-configured loggers and delete it
            temp_log_cache.transfer_to_non_streams(root_logger)
            temp_log_cache.close()
            root_logger.removeHandler(temp_log_cache)
            self.log.debug('Logger successfully configured.')

# ------------------------------------------------------------------------------

def git_info():
    """Get the current git branch, hash, and list of uncommitted files, if 
    available.

    Called by :meth:`DebugHeaderFileHandler._debug_header`. Based on NumPy's 
    implementation: `<https://stackoverflow.com/a/40170206>`__.
    """
    def _minimal_ext_cmd(cmd):
        # construct minimal environment
        env = os.environ.copy()
        env.update({'LANGUAGE':'C', 'LANG':'C', 'LC_ALL':'C'})
        try:
            out = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=env
            ).communicate()[0]
        except subprocess.CalledProcessError:
            out = ''
        return out.strip().decode('utf-8')

    git_branch = ""
    git_hash = ""
    git_dirty = ""
    try:
        git_branch = _minimal_ext_cmd(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
        git_hash = _minimal_ext_cmd(['git', 'rev-parse', 'HEAD'])
        git_dirty = _minimal_ext_cmd(['git', 'diff', '--no-ext-diff', '--name-only'])
    except OSError:
        pass
        
    if git_dirty:
        git_dirty = git_dirty.splitlines()
    elif git_hash:
        git_dirty = "<none>"
    else:
        git_dirty = "<couldn't get uncomitted files>"
    if not git_branch:
        git_branch = "<no branch>"
    if not git_hash:
        git_hash = "<couldn't get git hash>"
    return (git_branch, git_hash, git_dirty)

def signal_logger(caller_name, signum=None, frame=None, log=_log):
    """Lookup signal name from number and write to log.
    
    Taken from `<https://stackoverflow.com/a/2549950>`__.

    Args:
        caller_name (str): Calling function name, only used in log message.
        signum, frame: parameters of the signal we recieved.
    """
    if signum:
        sig_lookup = {
            k:v for v, k in reversed(sorted(list(signal.__dict__.items()))) \
                if v.startswith('SIG') and not v.startswith('SIG_')
        }
        log.info(
            "%s caught signal %s (%s)",
            caller_name, sig_lookup.get(signum, 'UNKNOWN'), signum
        )
    else:
        log.info("%s caught unknown signal.", caller_name)

def _set_excepthook(root_logger):
    """Ensure all uncaught exceptions, other than user KeyboardInterrupt, are 
    logged to the root logger.

    See `<https://docs.python.org/3/library/sys.html#sys.excepthook>`__.
    """
    def uncaught_exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Skip logging for user interrupt
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        root_logger.critical(
            (70*'*') + "\nUncaught exception:\n", # banner so it stands out 
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = uncaught_exception_handler

def _configure_logging_dict(log_d, log_args):
    """Convert CLI flags (``--verbose``/``--quiet``) into log levels. Configure 
    log level and filters on console handlers in a logging config dictionary.
    """
    # smaller number = more verbose  
    level = getattr(log_args, 'quiet', 0) - getattr(log_args, 'verbose', 0)
    if level <= 1:
        stderr_level = logging.WARNING
    elif level == 2:
        stderr_level = logging.ERROR
    else:
        stderr_level = logging.CRITICAL
    log_d['filters'] = {
        "stderr_filter": {"()": GeqLevelFilter, "level": stderr_level},
        "stdout_filter": {"()": EqLevelFilter, "level": logging.INFO},
        "debug_filter": {"()": LtLevelFilter, "level": logging.INFO}
    }
    for h in ('stderr', 'stdout', 'debug'):
        if h in log_d['handlers']:
            log_d['handlers'][h]['filters'] = [h+'_filter']
    if 'stderr' in log_d['handlers']:
        log_d['handlers']['stderr']['level'] = stderr_level

    if level <= -2:
        for d in log_d['handlers'].values():
            d['formatter'] = 'debug'
    if level == 0:
        del log_d['handlers']['debug']
        log_d['root']['handlers'] = ["stdout", "stderr"]
    elif level == 1:
        del log_d['handlers']['debug']
        del log_d['handlers']['stdout']
        log_d['root']['handlers'] = ["stderr"]
    return log_d

def configure_console_loggers():
    """Configure console loggers for top-level script. This is redundant with
    what's in logging.jsonc, but for debugging purposes we want to get console 
    output set up before we've parsed input paths, read config files, etc.
    """
    logging.captureWarnings(True)
    log_d = {
        "version": 1,
        "disable_existing_loggers":  False,
        "root": {"level": "NOTSET", "handlers": ["debug", "stdout", "stderr"]},
        "handlers": {
            "debug": {
                "class": "logging.StreamHandler",
                "formatter": "level",
                "level" : logging.DEBUG,
                "stream" : "ext://sys.stdout"
            },
            "stdout": {
                "class": "logging.StreamHandler",
                "formatter": "normal",
                "level" : logging.INFO,
                "stream" : "ext://sys.stdout"
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "formatter": "level",
                "level" : logging.WARNING,
                "stream" : "ext://sys.stderr"
            }
        },
        "formatters": {
            "normal": {"format": "%(message)s"},
            "level": {"format": "%(levelname)s: %(message)s"},
            "debug": {
                "()": HangingIndentFormatter,
                "format": ("%(levelname)s: %(funcName)s (%(filename)s line "
                    "%(lineno)d):\n%(message)s"),
                "tabsize": 4,
                "footer": "\n"
            }
        }
    }
    log_parser = argparse.ArgumentParser(add_help=False)
    log_parser.add_argument('--verbose', '-v', default=0, action="count")
    log_parser.add_argument('--quiet', '-q', default=0, action="count")
    log_args, _ = log_parser.parse_known_args()

    log_d = _configure_logging_dict(log_d, log_args)
    logging.config.dictConfig(log_d)
    _log.debug('Console loggers configured.')
