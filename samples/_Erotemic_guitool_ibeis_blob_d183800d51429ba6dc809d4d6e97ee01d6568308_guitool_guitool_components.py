# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import six
from six.moves import map, range
from guitool.__PYQT__ import QtCore, QtGui
from guitool.__PYQT__.QtGui import QSizePolicy
from guitool.__PYQT__.QtCore import Qt
import functools
import utool
import utool as ut  # NOQA
from guitool import guitool_dialogs
import weakref
(print, print_, printDBG, rrr, profile) = utool.inject(__name__,
                                                       '[guitool_components]')


ALIGN_DICT = {
    'center': Qt.AlignCenter,
    'right': Qt.AlignRight | Qt.AlignVCenter,
    'left': Qt.AlignLeft | Qt.AlignVCenter,
    'justify': Qt.AlignJustify,
}


def newSizePolicy(widget,
                  verticalSizePolicy=QSizePolicy.Expanding,
                  horizontalSizePolicy=QSizePolicy.Expanding,
                  horizontalStretch=0,
                  verticalStretch=0):
    """
    input: widget - the central widget
    """
    sizePolicy = QSizePolicy(horizontalSizePolicy, verticalSizePolicy)
    sizePolicy.setHorizontalStretch(horizontalStretch)
    sizePolicy.setVerticalStretch(verticalStretch)
    #sizePolicy.setHeightForWidth(widget.sizePolicy().hasHeightForWidth())
    return sizePolicy


def newSplitter(widget, orientation=Qt.Horizontal, verticalStretch=1):
    """
    input: widget - the central widget
    """
    hsplitter = QtGui.QSplitter(orientation, widget)
    # This line makes the hsplitter resize with the widget
    sizePolicy = newSizePolicy(hsplitter, verticalStretch=verticalStretch)
    hsplitter.setSizePolicy(sizePolicy)
    setattr(hsplitter, '_guitool_sizepolicy', sizePolicy)
    return hsplitter


def newTabWidget(parent, horizontalStretch=1):
    tabwgt = QtGui.QTabWidget(parent)
    sizePolicy = newSizePolicy(tabwgt, horizontalStretch=horizontalStretch)
    tabwgt.setSizePolicy(sizePolicy)
    setattr(tabwgt, '_guitool_sizepolicy', sizePolicy)
    return tabwgt


def newMenubar(widget):
    """ Defines the menubar on top of the main widget """
    menubar = QtGui.QMenuBar(widget)
    menubar.setGeometry(QtCore.QRect(0, 0, 1013, 23))
    menubar.setContextMenuPolicy(Qt.DefaultContextMenu)
    menubar.setDefaultUp(False)
    menubar.setNativeMenuBar(False)
    widget.setMenuBar(menubar)
    return menubar


def newQPoint(x, y):
    return QtCore.QPoint(int(round(x)), int(round(y)))


def newMenu(widget, menubar, name, text):
    """ Defines each menu category in the menubar """
    menu = QtGui.QMenu(menubar)
    menu.setObjectName(name)
    menu.setTitle(text)
    # Define a custom newAction function for the menu
    # The QT function is called addAction
    newAction = functools.partial(newMenuAction, widget, name)
    setattr(menu, 'newAction', newAction)
    # Add the menu to the menubar
    menubar.addAction(menu.menuAction())
    return menu


def newMenuAction(front, menu_name, name=None, text=None, shortcut=None,
                  tooltip=None, slot_fn=None, enabled=True):
    assert name is not None, 'menuAction name cannot be None'
    # Dynamically add new menu actions programatically
    action_name = name
    action_text = text
    action_shortcut = shortcut
    action_tooltip  = tooltip
    if hasattr(front, action_name):
        raise Exception('menu action already defined')
    # Create new action
    action = QtGui.QAction(front)
    setattr(front, action_name, action)
    action.setEnabled(enabled)
    action.setShortcutContext(QtCore.Qt.ApplicationShortcut)
    menu = getattr(front, menu_name)
    menu.addAction(action)
    if action_text is None:
        action_text = action_name
    if action_text is not None:
        action.setText(action_text)
    if action_tooltip is not None:
        action.setToolTip(action_tooltip)
    if action_shortcut is not None:
        action.setShortcut(action_shortcut)
    if slot_fn is not None:
        action.triggered.connect(slot_fn)
    return action


SHOW_TEXT = ut.get_argflag('--progtext')


class ProgressHooks(QtCore.QObject):
    """
    hooks into utool.ProgressIterator

    TODO:
        use signals and slots to connect to the progress bar
        still doesn't work correctly even with signals and slots, probably
          need to do task function in another thread

    References:
        http://stackoverflow.com/questions/19442443/busy-indication-with-pyqt-progress-bar
    """
    set_progress_signal = QtCore.pyqtSignal(int, int)
    show_indefinite_progress_signal = QtCore.pyqtSignal()

    def __init__(proghook, progressBar, substep_min=0, substep_size=1, level=0):
        super(ProgressHooks, proghook).__init__()
        proghook.progressBarRef = weakref.ref(progressBar)
        proghook.substep_min = substep_min
        proghook.substep_size = substep_size
        proghook.count = 0
        proghook.nTotal = None
        proghook.progiter = None
        proghook.lbl = ''
        proghook.level = level
        proghook.child_hook_gen = None
        proghook.set_progress_signal.connect(proghook.set_progress_slot)
        proghook.show_indefinite_progress_signal.connect(proghook.show_indefinite_progress_slot)

    def initialize_subhooks(proghook, num_child_subhooks):
        proghook.child_hook_gen = iter(proghook.make_substep_hooks(num_child_subhooks))

    def next_subhook(proghook):
        return six.next(proghook.child_hook_gen)

    def register_progiter(proghook, progiter):
        proghook.progiter = weakref.ref(progiter)
        proghook.nTotal = proghook.progiter().nTotal
        proghook.lbl = proghook.progiter().lbl

    def make_substep_hooks(proghook, num_substeps):
        """ make hooks that take up a fraction of this hooks step size.
            substep sizes are all fractional
        """
        step_min = ((proghook.progiter().count - 1) / proghook.nTotal) * proghook.substep_size  + proghook.substep_min
        step_size = (1.0 / proghook.nTotal) * proghook.substep_size

        substep_size = step_size / num_substeps
        substep_min_list = [(step * substep_size) + step_min for step in range(num_substeps)]

        DEBUG = False
        if DEBUG:
            with ut.Indenter(' ' * 4 * proghook.level):
                print('\n')
                print('+____<NEW SUBSTEPS>____')
                print('Making %d substeps for proghook.lbl = %s' % (num_substeps, proghook.lbl,))
                print(' * step_min         = %.2f' % (step_min,))
                print(' * step_size        = %.2f' % (step_size,))
                print(' * substep_size     = %.2f' % (substep_size,))
                print(' * substep_min_list = %r' % (substep_min_list,))
                print('L____<\NEW SUBSTEPS>____')
                print('\n')

        subhook_list = [ProgressHooks(proghook.progressBarRef(), substep_min, substep_size, proghook.level + 1)
                        for substep_min in substep_min_list]
        return subhook_list

    @QtCore.pyqtSlot()
    def show_indefinite_progress_slot(proghook):
        progbar = proghook.progressBarRef()
        progbar.reset()
        progbar.setMaximum(0)
        progbar.setProperty('value', 0)
        proghook.force_event_update()

    def show_indefinite_progress(proghook):
        proghook.show_indefinite_progress_signal.emit()

    def force_event_update(proghook):
        # major hack
        import guitool
        qtapp = guitool.get_qtapp()
        qtapp.processEvents()

    def set_progress(proghook, count, nTotal=None):
        if nTotal is None:
            nTotal = proghook.nTotal
        else:
            proghook.nTotal = nTotal
        if nTotal is None:
            nTotal = 100
        proghook.set_progress_signal.emit(count, nTotal)

    @QtCore.pyqtSlot(int, int)
    def set_progress_slot(proghook, count, nTotal=None):
        if nTotal is None:
            nTotal = proghook.nTotal
        else:
            proghook.nTotal = nTotal
        proghook.count = count
        local_fraction = (count) / nTotal
        global_fraction = (local_fraction * proghook.substep_size) + proghook.substep_min
        DEBUG = False

        if DEBUG:
            with ut.Indenter(' ' * 4 * proghook.level):
                print('\n')
                print('+-----------')
                print('proghook.substep_min = %.3f' % (proghook.substep_min,))
                print('proghook.lbl = %r' % (proghook.lbl,))
                print('proghook.substep_size = %.3f' % (proghook.substep_size,))
                print('global_fraction = %.3f' % (global_fraction,))
                print('local_fraction = %.3f' % (local_fraction,))
                print('L___________')
        if SHOW_TEXT:
            resolution = 75
            num_full = int(round(global_fraction * resolution))
            num_empty = resolution - num_full
            print('\n')
            print('[' + '#' * num_full + '.' * num_empty + '] %7.3f%%' % (global_fraction * 100))
            print('\n')
        #assert local_fraction <= 1.0
        #assert global_fraction <= 1.0
        progbar = proghook.progressBarRef()
        progbar.setRange(0, 10000)
        progbar.setMinimum(0)
        progbar.setMaximum(10000)
        value = int(round(progbar.maximum() * global_fraction))
        progbar.setFormat(proghook.lbl + ' %p%')
        progbar.setValue(value)
        #progbar.setProperty('value', value)
        # major hack
        proghook.force_event_update()
        #import guitool
        #qtapp = guitool.get_qtapp()
        #qtapp.processEvents()

    def __call__(proghook, count, nTotal=None):
        proghook.set_progress(count, nTotal)
        #proghook.set_progress_slot(count, nTotal)


def newProgressBar(parent, visible=True, verticalStretch=1):
    r"""
    Args:
        parent (?):
        visible (bool):
        verticalStretch (int):

    Returns:
        QProgressBar: progressBar

    CommandLine:
        python -m guitool.guitool_components --test-newProgressBar:0
        python -m guitool.guitool_components --test-newProgressBar:0 --show
        python -m guitool.guitool_components --test-newProgressBar:1 --progtext

    Example:
        >>> # GUI_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> import guitool
        >>> guitool.ensure_qtapp()
        >>> parent = None
        >>> visible = True
        >>> verticalStretch = 1
        >>> # hook into utool progress iter
        >>> progressBar = newProgressBar(parent, visible, verticalStretch)
        >>> progressBar.show()
        >>> progressBar.utool_prog_hook.show_indefinite_progress()
        >>> #progressBar.utool_prog_hook.set_progress(0)
        >>> #import time
        >>> qtapp = guitool.get_qtapp()
        >>> [(qtapp.processEvents(), ut.get_nth_prime_bruteforce(300)) for x in range(100)]
        >>> #time.sleep(5)
        >>> progiter = ut.ProgressIter(range(100), freq=1, autoadjust=False, prog_hook=progressBar.utool_prog_hook)
        >>> results1 = [ut.get_nth_prime_bruteforce(300) for x in progiter]
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> guitool.qtapp_loop(freq=10)

    Example:
        >>> # GUI_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> import guitool
        >>> guitool.ensure_qtapp()
        >>> parent = None
        >>> visible = True
        >>> verticalStretch = 1
        >>> def complex_tasks(proghook):
        ...     progkw = dict(freq=1, backspace=False, autoadjust=False)
        ...     num = 800
        ...     for x in ut.ProgressIter(range(4), lbl='TASK', prog_hook=proghook, **progkw):
        ...         ut.get_nth_prime_bruteforce(num)
        ...         subhooks = proghook.make_substep_hooks(2)
        ...         for task1 in ut.ProgressIter(range(2), lbl='task1.1', prog_hook=subhooks[0], **progkw):
        ...             ut.get_nth_prime_bruteforce(num)
        ...             subsubhooks = subhooks[0].make_substep_hooks(3)
        ...             for task1 in ut.ProgressIter(range(7), lbl='task1.1.1', prog_hook=subsubhooks[0], **progkw):
        ...                 ut.get_nth_prime_bruteforce(num)
        ...             for task1 in ut.ProgressIter(range(11), lbl='task1.1.2', prog_hook=subsubhooks[1], **progkw):
        ...                 ut.get_nth_prime_bruteforce(num)
        ...             for task1 in ut.ProgressIter(range(3), lbl='task1.1.3', prog_hook=subsubhooks[2], **progkw):
        ...                 ut.get_nth_prime_bruteforce(num)
        ...         for task2 in ut.ProgressIter(range(10), lbl='task1.2', prog_hook=subhooks[1], **progkw):
        ...             ut.get_nth_prime_bruteforce(num)
        >>> # hook into utool progress iter
        >>> progressBar = newProgressBar(parent, visible, verticalStretch)
        >>> complex_tasks(progressBar.utool_prog_hook)
        >>> # verify results
        >>> ut.quit_if_noshow()
        >>> guitool.qtapp_loop(freq=10)


    Ignore:
        from guitool.guitool_components import *  # NOQA
        # build test data
        import guitool
        guitool.ensure_qtapp()

    """
    progressBar = QtGui.QProgressBar(parent)
    sizePolicy = newSizePolicy(progressBar,
                               verticalSizePolicy=QSizePolicy.Maximum,
                               verticalStretch=verticalStretch)
    progressBar.setSizePolicy(sizePolicy)
    progressBar.setMaximum(10000)
    progressBar.setProperty('value', 0)
    #def utool_prog_hook(count, nTotal):
    #    progressBar.setProperty('value', int(100 * count / nTotal))
    #    # major hack
    #    import guitool
    #    qtapp = guitool.get_qtapp()
    #    qtapp.processEvents()
    #    pass
    progressBar.utool_prog_hook = ProgressHooks(progressBar)
    #progressBar.setTextVisible(False)
    progressBar.setTextVisible(True)
    progressBar.setFormat('%p%')
    progressBar.setVisible(visible)
    progressBar.setMinimumWidth(600)
    setattr(progressBar, '_guitool_sizepolicy', sizePolicy)
    if visible:
        # hack to make progres bar show up immediately
        import guitool
        progressBar.show()
        qtapp = guitool.get_qtapp()
        qtapp.processEvents()
    return progressBar


def newOutputLog(parent, pointSize=6, visible=True, verticalStretch=1):
    from guitool.guitool_misc import QLoggedOutput
    outputLog = QLoggedOutput(parent)
    sizePolicy = newSizePolicy(outputLog,
                               #verticalSizePolicy=QSizePolicy.Preferred,
                               verticalStretch=verticalStretch)
    outputLog.setSizePolicy(sizePolicy)
    outputLog.setAcceptRichText(False)
    outputLog.setVisible(visible)
    #outputLog.setFontPointSize(8)
    outputLog.setFont(newFont('Courier New', pointSize))
    setattr(outputLog, '_guitool_sizepolicy', sizePolicy)
    return outputLog


def newTextEdit(parent, visible=True):
    """ This is a text area """
    outputEdit = QtGui.QTextEdit(parent)
    sizePolicy = newSizePolicy(outputEdit, verticalStretch=1)
    outputEdit.setSizePolicy(sizePolicy)
    outputEdit.setAcceptRichText(False)
    outputEdit.setVisible(visible)
    setattr(outputEdit, '_guitool_sizepolicy', sizePolicy)
    return outputEdit


def newLineEdit(parent, text=None, enabled=True, align='center',
                textChangedSlot=None, textEditedSlot=None,
                editingFinishedSlot=None, visible=True, readOnly=False,
                fontkw={}):
    """ This is a text line

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> parent = None
        >>> text = None
        >>> visible = True
        >>> # execute function
        >>> widget = newLineEdit(parent, text, visible)
        >>> # verify results
        >>> result = str(widget)
        >>> print(result)
    """
    widget = QtGui.QLineEdit(parent)
    sizePolicy = newSizePolicy(widget, verticalStretch=1)
    widget.setSizePolicy(sizePolicy)
    if text is not None:
        widget.setText(text)
    widget.setEnabled(enabled)
    widget.setAlignment(ALIGN_DICT[align])
    widget.setReadOnly(readOnly)
    if textChangedSlot is not None:
        widget.textChanged.connect(textChangedSlot)
    if editingFinishedSlot is not None:
        widget.editingFinished.connect(editingFinishedSlot)
    if textEditedSlot is not None:
        widget.textEdited.connect(textEditedSlot)

    #outputEdit.setAcceptRichText(False)
    #outputEdit.setVisible(visible)
    adjust_font(widget, **fontkw)
    setattr(widget, '_guitool_sizepolicy', sizePolicy)
    return widget


def newWidget(parent, orientation=Qt.Vertical,
              verticalSizePolicy=QSizePolicy.Expanding,
              horizontalSizePolicy=QSizePolicy.Expanding,
              verticalStretch=1):
    widget = QtGui.QWidget(parent)

    sizePolicy = newSizePolicy(widget,
                               horizontalSizePolicy=horizontalSizePolicy,
                               verticalSizePolicy=verticalSizePolicy,
                               verticalStretch=1)
    widget.setSizePolicy(sizePolicy)
    if orientation == Qt.Vertical:
        layout = QtGui.QVBoxLayout(widget)
    elif orientation == Qt.Horizontal:
        layout = QtGui.QHBoxLayout(widget)
    else:
        raise NotImplementedError('orientation')
    # Black magic
    widget._guitool_layout = layout
    widget.addWidget = widget._guitool_layout.addWidget
    widget.addLayout = widget._guitool_layout.addLayout
    setattr(widget, '_guitool_sizepolicy', sizePolicy)
    return widget


def newFont(fontname='Courier New', pointSize=-1, weight=-1, italic=False):
    """ wrapper around QtGui.QFont """
    #fontname = 'Courier New'
    #pointSize = 8
    #weight = -1
    #italic = False
    font = QtGui.QFont(fontname, pointSize=pointSize, weight=weight, italic=italic)
    return font


def adjust_font(widget, bold=False, pointSize=None, italic=False):
    if bold or pointSize is not None:
        font = widget.font()
        font.setBold(bold)
        font.setItalic(italic)
        if pointSize is not None:
            font.setPointSize(pointSize)
        widget.setFont(font)


def newButton(parent=None, text='', clicked=None, qicon=None, visible=True,
              enabled=True, bgcolor=None, fgcolor=None, fontkw={}):
    """ wrapper around QtGui.QPushButton
    connectable signals:
        void clicked(bool checked=false)
        void pressed()
        void released()
        void toggled(bool checked)

    Args:
        parent (None):
        text (str):
        clicked (None):
        qicon (None):
        visible (bool):
        enabled (bool):
        bgcolor (None):
        fgcolor (None):
        bold (bool):

    Returns:
        ?: button

    CommandLine:
        python -m guitool.guitool_components --test-newButton

    Example:
        >>> # ENABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> parent = None
        >>> text = ''
        >>> clicked = None
        >>> qicon = None
        >>> visible = True
        >>> enabled = True
        >>> bgcolor = None
        >>> fgcolor = None
        >>> bold = False
        >>> # execute function
        >>> button = newButton(parent, text, clicked, qicon, visible, enabled, bgcolor, fgcolor, bold)
        >>> # verify results
        >>> result = str(button)
        >>> print(result)
    """
    but_args = [text]
    but_kwargs = {
        'parent': parent
    }
    if clicked is not None:
        but_kwargs['clicked'] = clicked
    else:
        enabled = False
    if qicon is not None:
        but_args = [qicon] + but_args
    button = QtGui.QPushButton(*but_args, **but_kwargs)
    style_sheet_str = make_style_sheet(bgcolor=bgcolor, fgcolor=fgcolor)
    if style_sheet_str is not None:
        button.setStyleSheet(style_sheet_str)
    button.setVisible(visible)
    button.setEnabled(enabled)
    adjust_font(button, **fontkw)
    return button


def newComboBox(parent=None, options=None, changed=None, default=None, visible=True,
                enabled=True, bgcolor=None, fgcolor=None, fontkw={}):
    """ wrapper around QtGui.QComboBox

    Args:
        parent (None):
        options (list): a list of tuples, which are a of the following form:
            [
                (visible text 1, backend value 1),
                (visible text 2, backend value 2),
                (visible text 3, backend value 3),
            ]
        changed (None):
        default (str): backend value of default item
        visible (bool):
        enabled (bool):
        bgcolor (None):
        fgcolor (None):
        bold (bool):

    Returns:
        ?: combo

    CommandLine:
        python -m guitool.guitool_components --test-newComboBox

    Example:
        >>> # DISABLE_DOCTEST
        >>> from guitool.guitool_components import *  # NOQA
        >>> # build test data
        >>> parent = None
        >>> options = None
        >>> changed = None
        >>> default = None
        >>> visible = True
        >>> enabled = True
        >>> bgcolor = None
        >>> fgcolor = None
        >>> bold = False
        >>> # execute function
        >>> combo = newComboBox(parent, options, changed, default, visible, enabled, bgcolor, fgcolor, bold)
        >>> # verify results
        >>> result = str(combo)
        >>> print(result)
    """
    class CustomComboBox(QtGui.QComboBox):
        def __init__(combo, parent=None, default=None, options=None, changed=None):
            QtGui.QComboBox.__init__(combo, parent)
            combo.ibswgt = parent
            combo.options = options
            combo.changed = changed
            combo.setEditable(True)
            combo.addItems( [ option[0] for option in combo.options ] )
            combo.currentIndexChanged['int'].connect(combo.currentIndexChangedCustom)
            combo.setDefault(default)

        def setOptionText(combo, option_text_list):
            for index, text in enumerate(option_text_list):
                combo.setItemText(index, text)
            #combo.removeItem()

        def currentIndexChangedCustom(combo, index):
            combo.changed(index, combo.options[index][1])

        def setDefault(combo, default=None):
            """ finds index of backend value and sets the current index """
            if default is not None:
                for index, (text, value) in enumerate(options):
                    if value == default:
                        combo.setCurrentIndex(index)
                        break
            else:
                combo.setCurrentIndex(0)

    combo_kwargs = {
        'parent' : parent,
        'options': options,
        'default': default,
        'changed': changed,
    }
    combo = CustomComboBox(**combo_kwargs)
    if changed is None:
        enabled = False
    combo.setVisible(visible)
    combo.setEnabled(enabled)
    adjust_font(combo, **fontkw)
    return combo


def newCheckBox(parent=None, text='', changed=None, checked=False, visible=True,
                enabled=True, bgcolor=None, fgcolor=None):
    """ wrapper around QtGui.QCheckBox
    """
    class CustomCheckBox(QtGui.QCheckBox):
        def __init__(check, text='', parent=None, checked=False, changed=None):
            QtGui.QComboBox.__init__(check, text, parent=parent)
            check.ibswgt = parent
            check.changed = changed
            if checked:
                check.setCheckState(2)  # 2 is equivelant to checked, 1 to partial, 0 to not checked
            check.stateChanged.connect(check.stateChangedCustom)

        def stateChangedCustom(check, state):
            check.changed(state == 2)

    check_kwargs = {
        'text'   : text,
        'checked': checked,
        'parent' : parent,
        'changed': changed,
    }
    check = CustomCheckBox(**check_kwargs)
    if changed is None:
        enabled = False
    check.setVisible(visible)
    check.setEnabled(enabled)
    return check


def make_style_sheet(bgcolor=None, fgcolor=None):
    style_list = []
    fmtdict = {}
    if bgcolor is not None:
        style_list.append('background-color: rgb({bgcolor})')
        fmtdict['bgcolor'] = ','.join(map(str, bgcolor))
    if fgcolor is not None:
        style_list.append('color: rgb({fgcolor})')
        fmtdict['fgcolor'] = ','.join(map(str, fgcolor))
    if len(style_list) > 0:
        style_sheet_fmt = ';'.join(style_list)
        style_sheet_str = style_sheet_fmt.format(**fmtdict)
        return style_sheet_str
    else:
        return None

#def make_qstyle():
#    style_factory = QtGui.QStyleFactory()
#    style = style_factory.create('cleanlooks')
#    #app_style = QtGui.QApplication.style()


def newLabel(parent=None, text='', align='center', fontkw={}):
    label = QtGui.QLabel(text, parent=parent)
    label.setAlignment(ALIGN_DICT[align])
    adjust_font(label, **fontkw)
    return label


def getAvailableFonts():
    fontdb = QtGui.QFontDatabase()
    available_fonts = list(map(str, list(fontdb.families())))
    return available_fonts


def layoutSplitter(splitter):
    old_sizes = splitter.sizes()
    print(old_sizes)
    phi = utool.get_phi()
    total = sum(old_sizes)
    ratio = 1 / phi
    sizes = []
    for count, size in enumerate(old_sizes[:-1]):
        new_size = int(round(total * ratio))
        total -= new_size
        sizes.append(new_size)
    sizes.append(total)
    splitter.setSizes(sizes)
    print(sizes)
    print('===')


def msg_event(title, msg):
    """ Returns a message event slot """
    return lambda: guitool_dialogs.msgbox(msg=msg, title=title)


if __name__ == '__main__':
    """
    CommandLine:
        python -m guitool.guitool_components
        python -m guitool.guitool_components --allexamples
        python -m guitool.guitool_components --allexamples --noface --nosrc
    """
    import multiprocessing
    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA
    ut.doctest_funcs()
