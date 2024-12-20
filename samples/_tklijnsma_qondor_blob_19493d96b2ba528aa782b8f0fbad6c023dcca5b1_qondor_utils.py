#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, shutil, logging, subprocess, glob, pprint, time, datetime
import os.path as osp
import qondor
logger = logging.getLogger('qondor')
subprocess_logger = logging.getLogger('subprocess')


def _create_directory_no_checks(dirname, dry=False):
    """
    Creates a directory without doing any further checks.

    :param dirname: Name of the directory to be created
    :type dirname: str
    :param dry: Don't actually create the directory, only log
    :type dry: bool, optional
    """
    logger.warning('Creating directory {0}'.format(dirname))
    if not dry: os.makedirs(dirname)

def create_directory(dirname, force=False, must_not_exist=False, dry=False):
    """
    Creates a directory if certain conditions are met.

    :param dirname: Name of the directory to be created
    :type dirname: str
    :param force: Removes the directory `dirname` if it already exists
    :type force: bool, optional
    :param must_not_exist: Throw an OSError if the directory already exists
    :type must_not_exist: bool, optional
    :param dry: Don't actually create the directory, only log
    :type dry: bool, optional
    """
    if osp.isfile(dirname):
        raise OSError('{0} is a file'.format(dirname))
    isdir = osp.isdir(dirname)

    if isdir:
        if must_not_exist:
            raise OSError('{0} must not exist but exists'.format(dirname))
        elif force:
            logger.warning('Deleting directory {0}'.format(dirname))
            if not dry: shutil.rmtree(dirname)
        else:
            logger.warning('{0} already exists, not recreating')
            return
    _create_directory_no_checks(dirname, dry=dry)

def copy_file(src, dst, dry=False):
    logger.info('Copying %s --> %s', src, dst)
    if not dry: shutil.copy(src, dst)

class switchdir(object):
    """
    Context manager to temporarily change the working directory.

    :param newdir: Directory to change into
    :type newdir: str
    :param dry: Don't actually change directory if set to True
    :type dry: bool, optional
    """
    def __init__(self, newdir, dry=False):
        super(switchdir, self).__init__()
        self.newdir = newdir
        self._backdir = os.getcwd()
        self._no_need_to_change = (self.newdir == self._backdir)
        self.dry = dry

    def __enter__(self):
        if self._no_need_to_change:
            logger.info('Already in right directory, no need to change')
            return
        logger.info('chdir to {0}'.format(self.newdir))
        if not self.dry: os.chdir(self.newdir)

    def __exit__(self, type, value, traceback):
        if self._no_need_to_change:
            return
        logger.info('chdir back to {0}'.format(self._backdir))
        if not self.dry: os.chdir(self._backdir)

def run_command(cmd, env=None, dry=False, shell=False):
    logger.warning('Issuing command: {0}'.format(' '.join(cmd) if not is_string(cmd) else cmd))
    if dry: return

    if shell and not is_string(cmd):
        cmd = ' '.join(cmd)

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        universal_newlines=True,
        shell=shell
        )

    output = []
    for stdout_line in iter(process.stdout.readline, ""):
        subprocess_logger.info(stdout_line.rstrip('\n'))
        output.append(stdout_line)
    process.stdout.close()
    process.wait()
    returncode = process.returncode

    if returncode == 0:
        logger.info('Command exited with status 0 - all good')
    else:
        logger.error('Exit status {0} for command: {1}'.format(returncode, cmd))
        raise subprocess.CalledProcessError(cmd, returncode)
    return output


def run_multiple_commands(cmds, env=None, dry=False):
    logger.info('Sending cmds:\n{0}'.format(pprint.pformat(cmds)))
    if dry:
        logger.info('Dry mode - not running command')
        return

    process = subprocess.Popen(
        'bash',
        stdin = subprocess.PIPE,
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
        env = env,
        bufsize = 1,
        close_fds = True
        )

    # Break on first error (stdin will still be written but execution will be stopped)
    process.stdin.write('set -e\n')
    process.stdin.flush()

    for cmd in cmds:
        if not(type(cmd) is str):
            cmd = ' '.join(cmd)
        if not(cmd.endswith('\n')):
            cmd += '\n'
        process.stdin.write(cmd)
        process.stdin.flush()
    process.stdin.close()

    process.stdout.flush()
    for line in iter(process.stdout.readline, ""):
        if len(line) == 0: break
        subprocess_logger.info(line.rstrip('\n'))

    process.stdout.close()
    process.wait()
    returncode = process.returncode

    if (returncode == 0):
        logger.info('Command exited with status 0 - all good')
    else:
        raise subprocess.CalledProcessError(cmd, returncode)


def is_string(string):
    """
    Checks strictly whether `string` is a string
    Python 2 / 3 compatibility (https://stackoverflow.com/a/22679982/9209944)
    """
    try:
        basestring
    except NameError:
        basestring = str
    return isinstance(string, basestring)


def tarball_python_module(module, outdir=None, allow_uncommitted=True, dry=False):
    """
    Takes a python module or a path to a file of said module, goes to the associated
    top-level git directory, and creates a tarball.
    Will throw subprocess.CalledProcessError if there are uncommitted changes.
    """
    # Input variable may be a path
    if is_string(module):
        # Treat the input variable as a path
        path = module
    else:
        # path = module.__file__
        path = osp.abspath(module.__path__[0])
        if not osp.exists(path):
            logger.warning(
                'Path %s for module %s does not exist; reloading and trying again',
                path, module
                )
            reload(module)
            path = osp.abspath(module.__path__[0])
            if osp.exists(path):
                logger.warning(
                    'Path %s for module %s exists. '
                    'Did you chdir before calling this function? '
                    'module.__path__ is a relative path set at import time.',
                    path, module
                    )

    # Make sure path exists and is a directory
    if not osp.exists(path):
        logger.error('Path %s does not seem to exist; cwd: %s', path, os.getcwd())
        raise OSError('{0} is not a valid path'.format(path))
    elif osp.isfile(path):
        path = osp.dirname(path)

    # Get the top-level git dir
    with switchdir(path):
        toplevel_git_dir = run_command(['git', 'rev-parse', '--show-toplevel'])[0].strip()

    # Fix the output name of the tarball
    outdir = os.getcwd() if outdir is None else outdir
    outfile = osp.join(osp.abspath(outdir), osp.basename(toplevel_git_dir) + '.tar')

    with switchdir(toplevel_git_dir):
        if allow_uncommitted:
            logger.info('Creating tarball for %s including uncommitted changes', toplevel_git_dir)
             # Create the tarball with uncommitted changes in it
            # if not dry: run_multiple_commands([
            #     'stashName=`git stash create` ',
            #     'git archive $stashName -o {0}'.format(outfile)
            #     ])
            if not dry:
                run_command(
                    'git ls-files -z | xargs -0 tar -cvf {0}'
                    .format(outfile),
                    shell=True
                    )
        else:
            # Check if there are uncommitted changes
            try:
                run_command(['git', 'diff-index', '--quiet', 'HEAD', '--'])
            except subprocess.CalledProcessError:
                logger.error(
                    'Uncommitted changes detected; it is unlikely you want a tarball '
                    'with some changes not committed.'
                    )
                raise
            # Create the actual tarball of the latest commit
            if not dry: run_command(['git', 'archive', '-o', outfile, 'HEAD'])

        logger.info('Created tarball {0}'.format(outfile))
        return outfile


def extract_tarball(tarball, outdir='.', dry=False):
    """
    Extracts a tarball to outdir
    """
    tarball = osp.abspath(tarball)
    outdir = osp.abspath(outdir)
    logger.warning(
        'Extracting {0} ==> {1}'
        .format(tarball, outdir)
        )
    cmd = [
        'tar', '-xvf', tarball,
        '-C', outdir
        ]
    run_command(cmd, dry=dry)


def extract_tarball_cmssw(tarball, outdir='.', dry=False):
    """
    Extracts a tarball to outdir, and returns the extracted CMSSW dir
    """
    extract_tarball(tarball, outdir, dry)
    # return the CMSSW directory
    if dry: return 'CMSSW_dry'
    return [ d for d in glob.glob(osp.join(outdir, 'CMSSW*')) if not d.endswith('.gz')][0]


def check_is_cmssw_path(path):
    """
    Checks whether the passed path contains a CMSSW distribution.
    """
    abs_path = osp.abspath(path)
    if not osp.basename(path).startswith('CMSSW'):
        raise ValueError(
            'Expected {0} to start with "CMSSW" (path: {1})'
            .format(osp.basename(path), abs_path)
            )
    if not osp.isdir(path):
        raise OSError(
            '{0} is not a directory (path: {1})'
            .format(path, abs_path)
            )
    if not osp.isdir(osp.join(path, 'src')):
        raise OSError(
            '{0} is not a directory (path: {1})'
            .format(osp.join(path, 'src'), abs_path)
            )

def get_clean_env():
    env = os.environ.copy()
    for var in [
        'ROOTSYS',
        'PATH',
        'LD_LIBRARY_PATH',
        'DYLD_LIBRARY_PATH',
        'SHLIB_PATH',
        'LIBPATH',
        'PYTHONPATH',
        'MANPATH',
        'CMAKE_PREFIX_PATH',
        'JUPYTER_PATH',
        # Added due to ROOT-env.sh
        'CPLUS_INCLUDE_PATH', 'CXX',
        'ZLIB_HOME',          'CURL_HOME',
        'DAVIX_HOME',         'GSL_HOME',
        'SETUPTOOLS_HOME',    'FONTCONFIG_HOME',
        'CAIRO_HOME',         'SQLITE_HOME',
        'PIXMAN_HOME',        'FREETYPE_HOME',
        'TBB_HOME',           'FC',
        'PKG_CONFIG_HOME',    'VC_HOME',
        'PNG_HOME',           'FFTW_HOME',
        'BOOST_HOME',         'VDT_HOME',
        'ROOT_HOME',          'ZEROMQ_HOME',
        'LIBXML2_HOME',       'PKG_CONFIG_PATH',
        'EXPAT_HOME',         'COMPILER_PATH',
        'BLAS_HOME',          'R_HOME',
        'XROOTD_HOME',        'MYSQL_HOME',
        'GFAL_HOME',          'CC',
        'C_INCLUDE_PATH',     'PYTHON_HOME',
        'PYTHONHOME',         'ORACLE_HOME',
        'GPERF_HOME',         'SRM_IFCE_HOME',
        'NUMPY_HOME',         'DCAP_HOME',
        ]:
        if var in env: del env[var]
    return env


def convert_to_utc(local_time):
    """
    Converts a local time to UTC.
    Implemented for only a few basic timezones.
    Needs to be extended with pytz if this function
    get seriously used.
    """
    # See: https://stackoverflow.com/a/10854983/9209944
    delta = datetime.timedelta(
        seconds = time.timezone if (time.localtime().tm_isdst == 0) else time.altzone
        )
    new_time = local_time + delta
    logger.debug('Offsetting %s by %s hours to get to UTC %s', local_time, delta, new_time)
    return new_time

def get_now_utc():
    return convert_to_utc(datetime.datetime.now())


def sleep_until(runtime_utc, allowed_lateness=300, is_not_utc=False):
    if is_not_utc:
        runtime_utc = convert_to_utc(runtime_utc)
    now_utc = get_now_utc()
    logger.info('Current time (UTC):       %s', now_utc)
    logger.info('Scheduled run time (UTC): %s', runtime_utc)

    delta = runtime_utc - now_utc
    delta_seconds = abs(delta.total_seconds())

    if delta < datetime.timedelta(seconds=0):
        # The job is too late; runtime_utc has already passed
        if delta_seconds < allowed_lateness:
            logger.info(
                'Job is late by %s seconds, which is within the allowed window'
                ' of max %s seconds late',
                delta_seconds, allowed_lateness
                )
            return 0
        else:
            logger.error(
                'Job is late by %s seconds, which is OUTSIDE the allowed window'
                ' of max %s seconds late - throwing exception',
                delta_seconds, allowed_lateness
                )
            raise RuntimeError
    else:
        logger.info(
            'Job is early by %s seconds, sleeping', delta_seconds
            )
        time.sleep(delta_seconds)
        return 0


def check_proxy():
    """
    Asserts that the user has a grid proxy that is valid for at least 168 more hours (1 week)
    """
    # cmd = 'voms-proxy-info -exists -valid 168:00' # Check if there is an existing proxy for a full week
    try:
        proxy_valid = subprocess.check_output(['grid-proxy-info', '-exists', '-valid', '168:00']) == 0
        logger.info('Found a valid proxy')
    except subprocess.CalledProcessError:
        logger.error(
            'Grid proxy is not valid for at least 1 week. Renew it using:\n'
            'voms-proxy-init -voms cms -valid 192:00'
            )
        raise
