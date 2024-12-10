#!/usr/bin/env python
#https://stackoverflow.com/questions/46265835/how-to-debug-a-python-module-from-the-command-line

from __future__ import print_function
import pdb
import runpy
import sys
import traceback

if len(sys.argv) == 0:
    print("Usage: mdb.py module_name [args ...]")
    exit(1)

modulename = sys.argv[1]
del sys.argv[0]

try:
    runpy.run_module(modulename, run_name='__main__')
except:
    traceback.print_exception(*sys.exc_info())
    print("")
    print("-" * 40)
    print("mdb: An exception occurred while executing module ", modulename)
    print("mdb: See the traceback above.")
    print("mdb: Entering post-mortem debugging.")
    print("-" * 40)
    pdb.post_mortem(sys.exc_info()[2])
