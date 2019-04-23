# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import timeit
import os
import numpy.distutils.system_info as sysinfo

setup = '''
import numpy as np

a = range(40001)

def np_float(a):
    b = np.array(a, dtype=np.float)
    for i in range(1000):
        b += (b/(i+1))
        
def np_float64(a):
    b = np.array(a, dtype=np.float64)
    for i in range(1000):
        b += (b/(i+1))
        
def np_float32(a):
    b = np.array(a, dtype=np.float32)
    for i in range(1000):
        b += (b/(i+1))
'''

try: f = open("timeit_np_types.log", "w")
except: f = None

def report(str_txt):
    if not(f == None):
        f.write("%s\n" % str_txt)
    print(str_txt)

N = 100
report("Test on computer: %s" % os.environ['COMPUTERNAME'])
report("timeit N = %i" % N)
report("")
report("numpy.distutils.system_info.get_info('mkl'):")
report(sysinfo.get_info('mkl'))
report("")
report("np_float  : %.2f ms" %
       (timeit.timeit('np_float(a)', setup=setup, number=N)/N*1000))
report("np_float64: %.2f ms" %
       (timeit.timeit('np_float64(a)', setup=setup, number=N)/N*1000))
report("np_float32: %.2f ms" %
       (timeit.timeit('np_float32(a)', setup=setup, number=N)/N*1000))
       
try: f.close()
except: pass