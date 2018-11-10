#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Provides functions for printing debug information to the terminal output.

Functions:
    dprint(...):
        'Debug' print a single line to the terminal with optional ANSI color
        codes. Particularly well suited for multithreaded PyQt programs where
        multiple threads are printing information to the same terminal.

    print_fancy_traceback(...):
        Prints the exception to the terminal, using ANSI color codes that mimic
        the IPython command shell.
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_debug_functions"
__date__        = "23-08-2018"
__version__     = "1.0.0"

import os
import sys

try:
    from PyQt5 import QtCore
except:
    PYQT5_IS_PRESENT = False
else:
    PYQT5_IS_PRESENT = True
    dprint_mutex = QtCore.QMutex()

class ANSI:
    NONE   = ''
    RED    = '\033[1;31m'
    GREEN  = '\033[1;32m'
    YELLOW = '\033[1;33m'
    BLUE   = '\033[1;34m'
    PURPLE = '\033[1;35m'
    CYAN   = '\033[1;36m'
    WHITE  = '\033[1;37m'

def dprint(str_msg, ANSI_color=None):
    """'Debug' print a single line to the terminal with optional ANSI color
    codes. The line will be terminated with a newline character and the
    terminal output buffer is forced to flush before and after every print.
    In addition, if PyQt5 is present in the Python environment, then a mutex
    lock will be obtained and released again for each dprint execution.

    There is a lot of overhead using this print statement, but it is
    particularly well suited for multithreaded PyQt programs where multiple
    threads are printing information to the same terminal. On the contrary, a
    regular print statement will likely result in mixed up text output.
    """
    # Explicitly ending the string with a newline '\n' character, instead
    # of letting the print statement end it for you (end='\n'), fixes the
    # problem of single lines getting printed to the terminal with
    # intermittently delayed newlines when coming from different threads.
    # I.e. it prevents:
    # >: Output line of thread 1Output line of thread 2   (\n)
    # >:                                                  (\n)
    # and makes sure we get:
    # >: Output line of thread 1                          (\n)
    # >: Output line of thread 2                          (\n)

    if PYQT5_IS_PRESENT: locker = QtCore.QMutexLocker(dprint_mutex)
    
    sys.stdout.flush()
    if ANSI_color is None:
        print("%s\n" % str_msg, end='')
    else:
        print("%s%s%s\n" % (ANSI_color, str_msg, ANSI.WHITE), end='')
    sys.stdout.flush()
    
    if PYQT5_IS_PRESENT: locker.unlock()

def print_fancy_traceback(err, back=3):
    """Print the exception `err` to the terminal with a traceback that is
    `back` deep, using ANSI color codes that mimic the IPython command shell.
    """
    print(ANSI.WHITE + '\nFancy traceback ' +
          ANSI.CYAN + '(most recent call last)' +
          ANSI.WHITE + ':')

    while back >= 1:
        try:
            err_file = os.path.basename(sys._getframe(back).f_code.co_filename)
        except ValueError:
            # Call stack is not deep enough. Proceed to next in line.
            back -= 1
        else:
            err_fun = sys._getframe(back).f_code.co_name
            err_line = sys._getframe(back).f_lineno
            print((ANSI.CYAN + 'File '   + ANSI.GREEN + '"%s"' +
                   ANSI.CYAN + ', line ' + ANSI.GREEN + '%s' +
                   ANSI.CYAN + ', in '   + ANSI.PURPLE + '%s' +
                   ANSI.WHITE) % (err_file, err_line, err_fun))
            back -= 1

    if isinstance(err, Exception):
        if not hasattr(err, 'abbreviation'): err.abbreviation = ''
        if not hasattr(err, 'description'): err.description = ''

        if (err.abbreviation == '' and err.description == ''):
            print((ANSI.RED + '%s: ' + ANSI.WHITE) % sys.exc_info()[0].__name__,
                  end='')
            print(err)
        else:
            print((ANSI.RED + '%s: ' + ANSI.WHITE + '%s: %s') %
                  (sys.exc_info()[0].__name__,
                   err.abbreviation,
                   err.description))

    elif isinstance(err, str):
        print((ANSI.RED + 'Error: ' + ANSI.WHITE + '%s') % err)
