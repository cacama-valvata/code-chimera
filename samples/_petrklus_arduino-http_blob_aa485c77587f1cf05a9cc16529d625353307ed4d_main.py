# coding=utf-8

import serial
import json
import random
import time
import datetime
import threading, Queue
import logging
import struct
import collections

from bottle import route, run, template
import time


"""
logging.basicConfig(filename=__file__.replace('.py','.log'),level=logging.DEBUG,format='%(asctime)s [%(name)s.%(funcName)s] %(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', filemode='a')
"""

# from http://stackoverflow.com/questions/18752980/reading-serial-data-from-arduino-with-python
class IRSerialCommunicator(threading.Thread):
    def __init__(self, dataQ, errQ, port, baudrate=115200):
        self.logger = logging.getLogger('IRSerialCommunicator')
        self.logger.debug('initializing')
        threading.Thread.__init__(self)
        self.ser = serial.Serial(port, baudrate)
        self.ser.timeout = 1
        #self.ser.flushInput()
        self.readCount = 0
        self.sleepDurSec = 5
        self.waitMaxSec = self.sleepDurSec * self.ser.baudrate / 10
        self.dataQ = dataQ
        self.errQ = errQ
        self.keepAlive = True
        self.stoprequest = threading.Event()
        self.setDaemon(True)
        self.dat = None
        self.inputStarted = False
        self.ver = 0.1

    def run(self):
        self.logger.debug('Serial reader running')
        dataIn = False
        while not self.stoprequest.isSet():
          if not self.isOpen():
            self.connectForStream()

          while self.keepAlive:
            dat = self.ser.readline()
            # some data validation goes here before adding to Queue...
            if len(dat) > 2:
                self.dataQ.put([time.time(), dat])
            if not self.inputStarted:
                self.logger.debug('reading')
            self.inputStarted = True
          self.dat.close()
          self.close()
          self.join_fin()

    def join_fin(self):
        self.logger.debug('stopping')
        self.stoprequest.set()

    def isOpen(self):
        self.logger.debug('Open? ' + str(self.ser.isOpen()))
        return self.ser.isOpen()

    def open(self):
          self.ser.open()

    def stopDataAquisition(self):
        self.logger.debug('Setting keepAlive to False')
        self.keepAlive = False

    def close(self):
        self.logger.debug('closing')
        self.stopDataAquisition()
        self.ser.close()

    def write(self, msg):
        self.ser.write(msg)

    def readline(self):
        return self.ser.readline()

    def connectForStream(self, debug=True):
        '''Attempt to connect to the serial port and fail after waitMaxSec seconds'''
        self.logger.debug('connecting')
        if not self.isOpen():
          self.logger.debug('not open, trying to open')
          try:
            self.open()
          except serial.serialutil.SerialException:
            self.logger.debug('Unable to use port ' + str(self.ser.port) + ', please verify and try again')
            return
        while self.readline() == '' and self.readCount < self.waitMaxSec and self.keepAlive:
            self.logger.debug('reading initial')
            self.readCount += self.sleepDurSec
            if not self.readCount % (self.ser.baudrate / 100):
              self.logger.debug("Verifying MaxSonar data..")
              #//some sanity check

        if self.readCount >= self.waitMaxSec:
            self.logger.debug('Unable to read from MaxSonar...')
            self.close()
            return False
        else:
          self.logger.debug('MaxSonar data is streaming...')

        return True


##### command sending dispatcher
command_q = Queue.Queue(maxsize=2)
def command_sender():
    while True:
        item = command_q.get() #blocking call        
        # TODO try twice/check confirmation?
        try:
            print item.run_action()     
            print "Command sent"       
        except Exception as e:
            msg = "Error send: {}".format(e)
            print msg
        # wait between issuing commands
        time.sleep(5)



# defaults for readings/state
current_state = {}
# prevent concurrent modification of the dictionary
state_lock = threading.Lock()
def set_state(temp, mode, fan_speed, power_toggle):
    with state_lock:
        current_state["temp"] = temp
        current_state["mode"] = mode
        current_state["fan_speed"] = fan_speed
        current_state["power_toggle"] = power_toggle
        current_state["timestamp"] = time.time()
    
LIGHT_DIFF = 100 #TODO move to config
def determine_state():
    ref     = int(current_state["L0"])
    on_off  = int(current_state["L1"])
    standby = int(current_state["L2"])
    
    # is the A/C running?
    if on_off - ref > LIGHT_DIFF:
        current_state["state_onoff"] = True
    else:
        current_state["state_onoff"] = False
        
    if standby - ref > LIGHT_DIFF:
        current_state["state_standby"] = True
    else:
        current_state["state_standby"] = False


##### command receiving processing
lines = collections.deque(maxlen=50)

SMOOTH_SIZE = 5
def store_read(key, val):
    with state_lock:
        if key in current_state.keys():
            items = current_state[key]
        else:
            items = collections.deque(maxlen=SMOOTH_SIZE)
            current_state[key] = items    
        items.append(val)


def generate_output():
    with state_lock:
        output = dict()
        for key, val in current_state.iteritems():
            output[key] = sum(val)/float(SMOOTH_SIZE)

    return output


def command_reader():
    while True:
        try:
            a = dataQ.get()
            if len(a) > 1:
                d_time = datetime.datetime.fromtimestamp(a[0])
                time_formatted = d_time.strftime('%H:%M:%S')
                lines.append("{}: {}".format(time_formatted, str(a[1]).strip()))                
                # now parse the command:                
                text_contents = a[1].strip()
                if text_contents[:2] != "[[" or text_contents[-2:] != "]]":
                    raise Exception(
                        "Invalid packet start {}".format(text_contents))
                    
                readings = text_contents[2:-2].split(";")
                for i, reading in enumerate(readings):
                    try:                        
                        key = "A{}".format(i)
                        store_read(key, float(reading))                      
                    except ValueError, e:
                        print "Error: {}".format(e)
                        # invalid combination, ignore..
                        # TODO paste error message..
                        continue                
                        
        except Exception as e:
            msg = "Error receive: {}".format(e)
            print msg


##### web routing
@route('/hello/<name>')
def index(name):
    return template('<b>Hello {{name}}</b>!', name=name)

import json
@route('/json_info')
def json_out():    
    return json.dumps(generate_output())


output_template = """<html>
    <head>
        <meta name="author" content="Petr">
        <meta charset="UTF-8">
        <meta http-equiv="refresh" content="2">
    </head>
    <body>
        <pre>{}</pre>
    </body>
</html>
"""
@route('/read')
def read_out():
    # trigger_read()
    return output_template.format("\n".join(list(lines)[::-1]))

"""
The THERM200 is a soil temperature probe, which has a temperature span from -40°C to 85°C.  It outputs a voltage linearly proportional to the temperature, so no complex equations are required, to calculate the temperature from voltage.  It is highly accurate with 0.125°C of resolution.
The sensor has a simple 3 wire interface: ground, power, and output,  and  is powered from 3.3V to 20VDC, and outputs a voltage 0 to 3V. Where 0 represents -40°C and 3V represents 85°.
"""
get_voltage = lambda x: x * (5.0 / 1023.0)
get_temp = lambda x: get_voltage(x) * 125/3.0 -40

if __name__=="__main__":    
    
    # TODO make this configurable
    port = "/dev/tty.usbmodem1411"
    dataQ = Queue.Queue(maxsize=100)
    errQ = Queue.Queue(maxsize=100)

    mock_serial = False
    if mock_serial:
        import os, pty, serial
        master, slave = pty.openpty()
        s_name = os.ttyname(slave)
        ser = IRSerialCommunicator(dataQ, errQ, port=s_name, baudrate=9600)
    else:
        ser = IRSerialCommunicator(dataQ, errQ, port=port, baudrate=19200)
    ser.daemon = True
    ser.start()
    
    
    # start command dispatcher    
    num_worker_threads = 1
    for i in range(num_worker_threads):
         # t = threading.Thread(target=command_sender)
         # t.daemon = True
         # t.start()
         # star
         t = threading.Thread(target=command_reader)
         t.daemon = True
         t.start()
    
    # run webserver
    # run(server='cherrypy', host='0.0.0.0', port=8080)
    run(host='0.0.0.0', port=8080)
