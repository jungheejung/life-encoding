#!/usr/bin/env python

import sys
import serial
import subprocess as sp

debug = True

part = sys.argv[1]
#cmd = 'mplayer -fs -ss 00:00:08 new_part{0}.mp4'.format(part)
if part == 3:
    cmd = 'cvlc -f --start-time 8.6 --no-video-title --hotkeys-mousewheel-mode 2 new_part{0}.mp4'.format(part)
else:
    cmd = 'cvlc -f --start-time 8.0 --no-video-title --hotkeys-mousewheel-mode 2 new_part{0}.mp4'.format(part)	
#cmd = cmd.split()

ser = serial.Serial('/dev/serial/by-id/usb-Keyspan__a_division_of_InnoSys_Inc._Keyspan_USA-19H-if00-port0', timeout=0)
# requires ASCII 9600
#ser = serial.Serial('/dev/tty.KeySerial1', 115200, timeout=0)

trigger = None

import time
from datetime import datetime

print "Waiting for scanner..."

while trigger != '5':
   trigger = ser.read()

print "Got trigger: ", datetime.now()
sp.call(cmd, shell=True)
