"""Timer with tray icon implemented in PyQT"""
from PyQt5.QtWidgets import QMainWindow, qApp, QApplication, QDesktopWidget, QWidget, QVBoxLayout,\
    QPushButton, QLabel, QHBoxLayout, QSystemTrayIcon, QAction, QMenu, QInputDialog, QMessageBox
from PyQt5.QtGui import QFont, QIcon
from PyQt5.QtCore import Qt, QTimer
from munch import munchify, Munch
import sys
import os

# Import modules from './timer/' dir
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/timer')
from timerEndedDialog import TimeEndedDialog
import parseString

# Import modules from './include/' dir
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/include')
import helpers
from single_instance_windows import SingleInstanceWindows
from single_instance_unix import SingleInstanceUnix


class Timer(QMainWindow, SingleInstanceUnix, SingleInstanceWindows):
    """
    Main window of Timer
    """

    SETTINGS_FILE = "timer.json"

    # Default settings to be written in json file (if file is absent)
    DEFAULT_SETTINGS = {
        "counted": 0,
        "chosenInterval": None
    }

    # Color of text in tray icon
    COLOR1 = "#fff"

    # Background of tray icon, border and font color of label in the window
    COLOR2 = "#000"

    state = Munch(
        chosenInterval=0,
        counted=0,
        currentCentiseconds=0,
        isRunning=False,
        isPaused=False,
    )

    # Grouping all widgets into a single object/namespace
    widgets = Munch()

    def __init__(self):
        super().__init__()

        if self.isAlreadyRunningUnix() or self.isAlreadyRunningWindows():
            print('Another instance is already running. Exiting')
            QMessageBox.about(
                self, "Error", 'Another instance is already running. Exiting')
            sys.exit()

        self.settings = munchify(helpers.readOrWriteSettings(
            self.SETTINGS_FILE, self.DEFAULT_SETTINGS))
        self.state.counted = self.settings.counted
        self.state.chosenInterval = self.settings.chosenInterval
        if self.state.counted > 0 and self.state.chosenInterval > 0:
            self.state.isRunning = True
            self.state.isPaused = True

        # self.setWindowFlags(QtCore.Qt.WindowStaysOnTopHint)

        self.setWindowIcon(QIcon(
            helpers.getCurrentDirectory() + "/timer/" + 'icon.png'))

        self.setWindowTitle("PythonTimer")

        self.setGeometry(100, 100, 400, 520)

        self.moveWindowToCenter()

        self.addUiComponents()

        self.addTrayIcon()

        self.addTimer()

        self.show()

    def setTrayText(self, text="--"):
        """Render text in tray icon"""
        self.widgets.tray.setIcon(helpers.drawIcon(text, self.COLOR1, self.COLOR2))

    def addTrayIcon(self):
        """Adds tray icon with menu"""
        self.widgets.tray = QSystemTrayIcon()

        actionShow = QAction("Show", self)
        actionQuit = QAction("Exit", self)
        actionHide = QAction("Hide", self)
        actionShow.triggered.connect(self.show)
        actionHide.triggered.connect(self.hide)
        # actionQuit.triggered.connect(qApp.quit)
        actionQuit.triggered.connect(self.areYouSureAndClose)

        menu = QMenu()
        menu.addAction(actionShow)
        menu.addAction(actionHide)
        menu.addAction(actionQuit)
        self.widgets.tray.setContextMenu(menu)

        self.setTrayText("--")

        self.widgets.tray.activated.connect(self.onTrayIconActivated)

        self.widgets.tray.setVisible(True)

    def onTrayIconActivated(self, reason):
        """
        When tray icon is clicked:
        * Show/Hide window if left mouse button is clicked
        * Start/pause if middle mouse button is clicked
        """
        if reason == QSystemTrayIcon.Trigger:
            if self.isHidden():
                self.show()
            else:
                self.hide()
        elif reason == QSystemTrayIcon.MiddleClick:
            self.onClickStartPause()

    def moveWindowToCenter(self):
        """
        Center PyQt window
        https://pythonprogramminglanguage.com/pyqt5-center-window/
        """
        qtRectangle = self.frameGeometry()
        centerPoint = QDesktopWidget().availableGeometry().center()
        qtRectangle.moveCenter(centerPoint)
        self.move(qtRectangle.topLeft())

    def addTimer(self):
        """Add timer, connect it to handler function, start it"""
        timer = QTimer(self)
        timer.timeout.connect(self.onTimer)
        timer.start(10)
        self.onTimer()

    def addUiComponents(self):  # pylint: disable=too-many-statements
        """Add UI components and connect them to handler functions"""
        layout = QVBoxLayout()

        self.widgets.buttonSet = QPushButton("Set time", self)
        self.widgets.buttonSet.clicked.connect(self.onClickSet)

        layout.addWidget(self.widgets.buttonSet)

        self.widgets.buttonSetStart = QPushButton("Set time and start", self)
        self.widgets.buttonSetStart.clicked.connect(self.onClickSetStart)

        layout.addWidget(self.widgets.buttonSetStart)

        self.widgets.labelTimeSet = QLabel("time set: --", self)
        self.widgets.labelTimeSet.setStyleSheet(
            "border : 4px solid " + self.COLOR2 + "; color: " + self.COLOR2 + ";")
        self.widgets.labelTimeSet.setFont(QFont('Times', 15))
        self.widgets.labelTimeSet.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.widgets.labelTimeSet)

        self.widgets.labelCountdown = QLabel("countdown: --", self)
        self.widgets.labelCountdown.setStyleSheet(
            "border : 4px solid " + self.COLOR2 + "; color: " + self.COLOR2 + ";")
        self.widgets.labelCountdown.setFont(QFont('Times', 15))
        self.widgets.labelCountdown.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.widgets.labelCountdown)

        self.widgets.buttonStartPause = QPushButton("Start", self)
        self.widgets.buttonStartPause.clicked.connect(self.onClickStartPause)

        layout.addWidget(self.widgets.buttonStartPause)

        hlayout = QHBoxLayout()

        self.widgets.buttonMinus1h = QPushButton("-1h", self)
        self.widgets.buttonMinus1h.pressed.connect(self.onClickMinus1h)
        hlayout.addWidget(self.widgets.buttonMinus1h)

        self.widgets.buttonMinus10m = QPushButton("-10m", self)
        self.widgets.buttonMinus10m.pressed.connect(self.onClickMinus10m)
        hlayout.addWidget(self.widgets.buttonMinus10m)

        self.widgets.buttonMinus1m = QPushButton("-1m", self)
        self.widgets.buttonMinus1m.pressed.connect(self.onClickMinus1m)
        hlayout.addWidget(self.widgets.buttonMinus1m)

        self.widgets.buttonPlus1m = QPushButton("+1m", self)
        self.widgets.buttonPlus1m.pressed.connect(self.onClickPlus1m)
        hlayout.addWidget(self.widgets.buttonPlus1m)

        self.widgets.buttonPlus10m = QPushButton("+10m", self)
        self.widgets.buttonPlus10m.pressed.connect(self.onClickPlus10m)
        hlayout.addWidget(self.widgets.buttonPlus10m)

        self.widgets.buttonPlus1h = QPushButton("+1h", self)
        self.widgets.buttonPlus1h.pressed.connect(self.onClickPlus1h)
        hlayout.addWidget(self.widgets.buttonPlus1h)

        layout.addLayout(hlayout)

        self.widgets.buttonReset = QPushButton("Reset", self)
        self.widgets.buttonReset.clicked.connect(self.onClickReset)

        layout.addWidget(self.widgets.buttonReset)

        self.widgets.buttonMinimize = QPushButton("Hide to tray", self)
        self.widgets.buttonMinimize.pressed.connect(self.hide)

        layout.addWidget(self.widgets.buttonMinimize)

        self.widgets.buttonExit = QPushButton("Exit (not to tray)", self)
        self.widgets.buttonExit.clicked.connect(self.areYouSureAndClose)
        layout.addWidget(self.widgets.buttonExit)

        if not self.state.isRunning:
            self.widgets.buttonStartPause.setDisabled(True)
            self.widgets.buttonReset.setDisabled(True)
            self.minusPlusButtonSetDisabled(True)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def areYouSureAndClose(self):
        """
        Ask user's confirmation and exit
        """
        quitMsg = "Are you sure you want to exit the program?"
        reply = QMessageBox.question(self, 'Message',
                                     quitMsg, QMessageBox.Yes, QMessageBox.No)

        if reply == QMessageBox.Yes:
            qApp.quit()

    def onTimer(self):
        """
        When timer is triggered:
        * Update time counter
        * Write time counter into settings JSON file
        * Update texts in UI
        * When counter becomes zero, show TimeEndedDialog and disable buttons
        """
        previousCentiseconds = self.state.currentCentiseconds or helpers.getCentiseconds()
        self.state.currentCentiseconds = helpers.getCentiseconds()

        if self.state.isRunning and not self.state.isPaused:
            # Timer can be not accurate
            # https://stackoverflow.com/questions/58699630/accurate-timer-with-pyqt
            # self.settings.counted = self.state.counted = self.state.counted - 1
            delta = self.state.currentCentiseconds - previousCentiseconds
            self.settings.counted = self.state.counted = self.state.counted - delta

            helpers.writeSettingsFile(self.SETTINGS_FILE, self.settings)

            if self.state.counted == 0:
                self.state.isRunning = False
                self.updateTexts(True)
                TimeEndedDialog.run()
                self.updateTexts()
                self.widgets.buttonStartPause.setText('Start')
                self.widgets.buttonReset.setDisabled(True)
                self.minusPlusButtonSetDisabled(True)

        if self.state.isRunning:
            self.updateTexts()

    def updateTexts(self, completed=False):
        """Update texts in window and tray icon according to self.state"""
        if completed:
            self.widgets.labelCountdown.setText("Completed !!!! ")
            self.setTrayText("!!!")
        elif self.state.isRunning:
            text = helpers.genTextFull(self.state.counted)
            if self.state.isPaused:
                text += " p"
            self.widgets.labelCountdown.setText("countdown: " + text)
            if not self.state.isPaused:
                self.setTrayText(helpers.genTextShort(self.state.counted))
            else:
                self.setTrayText("p")
        else:
            self.setTrayText("--")
            self.widgets.labelCountdown.setText("countdown: --")

        if self.state.chosenInterval > 0:
            self.widgets.labelTimeSet.setText(
                "time set: " + helpers.genTextFull(self.state.chosenInterval))

    def parseInputtedValue(self, value):
        """
        Parse time interval inputted by user
        """
        if not parseString.isStringValid(value):
            return 0

        w = parseString.withoutQualifier(value)
        m = parseString.getQualifierMult(value)
        return w * m

    def askTime(self):
        """
        Show dialog asking user for time interval
        """
        second, done = QInputDialog.getText(self, 'Seconds', 'Enter interval\nExamples: "60" or "60s" or "1m", "3600" or "60m" or "1h", "12h" or "1d"')
        v = self.parseInputtedValue(second)
        return v, done

    def onClickSet(self):
        """When "Set time" button is pressed"""
        v, done = self.askTime()

        if done and v > 0:
            self.settings.chosenInterval = self.state.chosenInterval = v * 10
            helpers.writeSettingsFile(self.SETTINGS_FILE, self.settings)

            self.state.isRunning = False
            self.state.isPaused = False

            self.widgets.buttonStartPause.setText("Start")
            self.widgets.buttonStartPause.setDisabled(False)
            self.widgets.buttonReset.setDisabled(True)
            self.minusPlusButtonSetDisabled(True)

            self.updateTexts()

    def onClickSetStart(self):
        """When "Set time and start" button is pressed"""
        v, done = self.askTime()

        if done and v > 0:
            self.settings.chosenInterval = self.state.chosenInterval = self.state.counted = v * 10
            helpers.writeSettingsFile(self.SETTINGS_FILE, self.settings)

            self.state.isRunning = True
            self.state.isPaused = False

            self.widgets.buttonStartPause.setText("Pause")
            self.widgets.buttonStartPause.setDisabled(False)
            self.widgets.buttonReset.setDisabled(False)
            self.minusPlusButtonSetDisabled(False)

            self.updateTexts()

    def onClickStartPause(self):
        """When Start/Pause button is pressed"""
        if self.state.counted == 0 and not self.state.isPaused and self.state.chosenInterval != 0:
            self.state.isRunning = True
            self.state.counted = self.state.chosenInterval
            self.widgets.buttonStartPause.setText("Pause")
            self.widgets.buttonReset.setDisabled(False)
            self.minusPlusButtonSetDisabled(False)
        else:
            if not self.state.isPaused:
                self.state.isPaused = True
                self.widgets.buttonStartPause.setText("Start")
            elif self.state.isPaused:
                self.state.isPaused = False
                self.widgets.buttonStartPause.setText("Pause")

            self.updateTexts()

    def onClickReset(self):
        """When Reset button is pressed"""
        self.state.isRunning = False
        self.state.isPaused = False

        self.state.counted = 0

        self.updateTexts()

        self.widgets.buttonStartPause.setText('Start')
        self.widgets.buttonReset.setDisabled(True)
        self.minusPlusButtonSetDisabled(True)

    def minusPlusButtonSetDisabled(self, trueOrFalse):
        """Disable/enable all the buttons that add or substract time"""
        buttons = [
            self.widgets.buttonMinus1h, self.widgets.buttonMinus10m, self.widgets.buttonMinus1m,
            self.widgets.buttonPlus1m, self.widgets.buttonPlus10m, self.widgets.buttonPlus1h
        ]
        for button in buttons:
            button.setDisabled(trueOrFalse)

    def onClickMinus1h(self):
        """When -1h button is pressed"""
        self.changeTimeByDeltaAndUpdateUI(-60 * 60 * 100)

    def onClickMinus10m(self):
        """When -10m button is pressed"""
        self.changeTimeByDeltaAndUpdateUI(-60 * 10 * 100)

    def onClickMinus1m(self):
        """When -1m button is pressed"""
        self.changeTimeByDeltaAndUpdateUI(-60 * 100)

    def onClickPlus1m(self):
        """When +1m button is pressed"""
        self.changeTimeByDeltaAndUpdateUI(60 * 100)

    def onClickPlus10m(self):
        """When +10m button is pressed"""
        self.changeTimeByDeltaAndUpdateUI(60 * 10 * 100)

    def onClickPlus1h(self):
        """When +1h button is pressed"""
        self.changeTimeByDeltaAndUpdateUI(60 * 60 * 100)

    def changeTimeByDeltaAndUpdateUI(self, delta):
        """
        * Add (integer) delta to time counter.
        * Write new counter into JSON settings file.
        * Update UI (texts in window and tray icon)
        """
        newVal = self.state.counted + delta

        self.changeTimeToValAndUpdateUI(newVal)

    def changeTimeToValAndUpdateUI(self, newVal):
        """
        * Update time counter with new (integer) value.
        * Write new counter into JSON settings file.
        * Update UI (texts in window and tray icon)
        """

        if newVal < 0:
            return

        self.settings.counted = self.state.counted = newVal

        helpers.writeSettingsFile(self.SETTINGS_FILE, self.settings)

        self.updateTexts()

    def closeEvent(self, event):
        """
        Overriding PyQt close event.
        Ask user's confirmation before exiting
        https://learndataanalysis.org/example-of-how-to-use-the-qwidget-close-event-pyqt5-tutorial/
        """
        event.ignore()
        self.areYouSureAndClose()


App = QApplication(sys.argv)
App.setQuitOnLastWindowClosed(False)
window = Timer()
sys.exit(App.exec())
