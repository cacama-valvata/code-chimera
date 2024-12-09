'''
version_manager.py

Copyright 2011 Andres Riancho

This file is part of w3af, http://w3af.org/ .

w3af is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 2 of the License.

w3af is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with w3af; if not, write to the Free Software
Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
'''
from datetime import date

import core.controllers.output_manager as om

from core.controllers.misc.homeDir import W3AF_LOCAL_PATH
from core.controllers.auto_update.git_client import GitClient, GitClientError
from core.data.db.startup_cfg import StartUpConfig


class VersionMgr(object):
    '''
    Perform SVN w3af code update and commit. When an instance is created loads
    data from a .conf file that will be used when actions are executed.
    Also provides some callbacks as well as events to register to.

    Callbacks on:
        UPDATE:
            * callback_onupdate_confirm(msg)
                Return True/False

            * callback_onupdate_show_log(msg, log_func)
                Displays 'msg' to the user and depending on user's answer
                call 'log_func()' which returns a string with the summary of
                the commit logs from the from local revision to repo's.

            * callback_onupdate_error
                If an SVNError occurs this callback is called in order to the
                client class handles the error. Probably notify the user.
        COMMIT:
            {implementation pending}
    Events:
        ON_UPDATE
        ON_UPDATE_ADDED_DEP
        ON_UPDATE_CHECK
        ON_ACTION_ERROR
    '''

    # Events constants
    ON_UPDATE = 1
    ON_UPDATE_ADDED_DEP = 2
    ON_UPDATE_CHECK = 3
    ON_ACTION_ERROR = 4
    ON_COMMIT = 6

    # Callbacks
    callback_onupdate_confirm = None
    callback_onupdate_show_log = None
    callback_onupdate_error = None

    # Revision constants
    HEAD = 'HEAD'
    PREVIOUS = 'BACK'

    def __init__(self, localpath=W3AF_LOCAL_PATH, log=None):
        '''
        w3af version manager class. Handles the logic concerning the
        automatic update/commit process of the code.

        @param localpath: Working directory
        @param log: Default output function
        '''
        self._localpath = localpath
        self._client = GitClient(localpath)
        
        log = log if log is not None else om.out.console
        self._log = log
        
        # Set default events
        self.register_default_events(log)
        # Startup configuration
        self._start_cfg = StartUpConfig()
    
    def register_default_events(self, log):
        '''
        Default events registration
        
        @param log: Log function to call for events
        @return: None, all saved in self._reg_funcs
        '''
        # Registered functions
        self._reg_funcs = {}
        
        msg = ('Checking if a new version is available in our SVN repository.'
               ' Please wait...')
        self.register(VersionMgr.ON_UPDATE_CHECK, log, msg)
        
        msg = 'w3af is updating from the official SVN server...'
        self.register(VersionMgr.ON_UPDATE, log, msg)
        
        msg = ('The third-party dependencies for w3af have changed, please'
               ' exit the framework and run it again to load all changes'
               ' and install any missing modules.')
        self.register(VersionMgr.ON_UPDATE_ADDED_DEP, log, msg)

    def update(self, force=False, commit=HEAD, print_result=False):
        '''
        Perform code update if necessary. Return three elems tuple with the
        ChangeLog of the changed files, the local and the repo's revision.

        @param force: Force update ignoring the startup config.
        @param rev: Revision number. If != 'HEAD' then update will be forced.
                    Also, if commit equals 'BACK' assume revision number is
                    the last that worked.
        @param print_result: If True print the result files using instance's
                             log function.
        '''
        client = self._client
        
        rev = int(rev)
        localrev = client.get_revision(local=True)
        files = SVNFilesList(rev=localrev)
        # If revision is not HEAD then force = True
        if rev != VersionMgr.HEAD:
            if rev == -1:  # Use previous working revision
                rev = self._start_cfg.last_commit_id
            remrev = rev
        else:
            remrev = None

        if force or self._has_to_update():
            self._notify(VersionMgr.ON_UPDATE_CHECK)
            remrev = (Revision(remrev, None) if remrev
                      else client.get_revision(local=False))

            # If local and repo's rev are the same => Nothing to do.
            if localrev != remrev:
                proceed_upd = True
                callback = self.callback_onupdate_confirm
                # Call callback function
                if callback is not None:
                    # pylint: disable=E1102
                    # pylint: disable=E1103
                    proceed_upd = callback(
                        'Your current w3af installation is r%s. Do you want '
                        'to update to r%s?' % (localrev.number, remrev.number))

                if proceed_upd:
                    self._notify(VersionMgr.ON_UPDATE)
                    
                    # Finally do the code update!
                    try:
                        changelog = client.pull(rev=remrev)
                    except GitClientError, err:
                        msg = 'An error occurred while updating:\n%s' % str(err.args)
                        self._notify(VersionMgr.ON_ACTION_ERROR, msg)
                        return (None, None, None)
                    
                    # Update last-rev.
                    self._start_cfg.last_commit_id = min(localrev, remrev)
                    # Reload all modules to make sure we have all the latest
                    # versions of py files in memory.
                    self.reload_all_modules()

                    newdeps = self._added_new_dependencies(changelog)
                    if newdeps:
                        self._notify(VersionMgr.ON_UPDATE_ADDED_DEP)

            # Save today as last-update date and persist it.
            self._start_cfg.last_upd = date.today()
            self._start_cfg.save()

            # Before returning perform some interaction with the user if
            # requested.
            if print_result:
                self._log(str(changelog))

            
            callback = self.callback_onupdate_show_log
            
            # Skip downgrades
            if remrev > localrev and callback:
                changelog_str = lambda: str(changelog)
                callback('Do you want to see a summary of the new code '
                         'commits log messages?', changelog_str)
                
        return (changelog, localrev, remrev)

    def reload_all_modules(self):
        '''
        After an update, which changes .py files, it is a good idea
        to reload all modules (and get those changes from the py files into
        memory) before continuing.

        @return: None.

        TODO: This still needs to be implemented, I tried some ideas from:
        http://stackoverflow.com/questions/437589/how-do-i-unload-reload-a-python-module
        http://code.activestate.com/recipes/81731-reloading-all-modules/

        But both failed. What I want to avoid are bugs like the ones related to
        the "complex type needs to implement..." DiskList.
        '''
        pass

    def register(self, event, func, msg):
        '''
        Register the caller to `event` so when it takes place call its `func`
        with `msg` as param.
        '''
        self._reg_funcs[event] = (func, msg)

    def _notify(self, event, msg=''):
        '''
        Call registered function for event. If `msg` is not empty use it.
        '''
        f, _msg = self._reg_funcs.get(event)
        f(msg or _msg)

    def _added_new_dependencies(self, changelog):
        '''
        @return: True if the changelog shows any modifications to the
                 dependency_check.py files.
        '''
        for commit in changelog.get_changes():
            for action, filename in commit.changes:
                if filename.endswith('dependency_check.py') and action == 'M':
                    return True
        return False

    def _has_to_update(self):
        '''
        Helper method that figures out if an update should be performed
        according to the startup cfg file.
        Some rules:
            1) IF auto_upd is False THEN return False
            2) IF last_upd == 'yesterday' and freq == 'D' THEN return True
            3) IF last_upd == 'two_days_ago' and freq == 'W' THEN return False.

        @return: Boolean value.
        '''
        startcfg = self._start_cfg
        # That's it!
        if not startcfg.auto_upd:
            return False
        else:
            freq = startcfg.freq
            diff_days = max((date.today() - startcfg.last_upd).days, 0)

            if ((freq == StartUpConfig.FREQ_DAILY and diff_days > 0) or
                (freq == StartUpConfig.FREQ_WEEKLY and diff_days > 6) or
                (freq == StartUpConfig.FREQ_MONTHLY and diff_days > 29)):
                return True
            return False




