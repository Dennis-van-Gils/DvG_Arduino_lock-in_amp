# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import timeit
import os

setup = '''
import numpy as np
import cupy
import sigpy as sp
from scipy.signal import firwin, fftconvolve
from collections import deque

buffer_size = 1000
N_buffers_in_deque = 81

N_taps  = buffer_size * (N_buffers_in_deque - 1) + 1
N_deque = buffer_size * N_buffers_in_deque

np.random.seed(0)

a_np = np.random.randn(N_deque)
a_dq = deque(a_np, maxlen=N_deque)

b_np = firwin(N_taps,
              [0.0001, 0.015, 0.025, .5-1e-4],
              window=("chebwin", 50),
              fs=1,
              pass_zero=False)
b_cp = cupy.array(b_np)

def cuda_fftconv():
    a_dq.extend(np.random.randn(buffer_size))
    a_cp = cupy.array(list(a_dq))
    sp.convolve(a_cp, b_cp, mode='valid')
    
def fftconv_dq_np__dq_to_list():
    a_dq.extend(np.random.randn(buffer_size))
    a_ls = list(a_dq)
    fftconvolve(a_ls, b_np, mode='valid')
'''

try: f = open("timeit_cuda_fftconvolve.log", "w")
except: f = None

def report(str_txt):
    if not(f == None):
        f.write("%s\n" % str_txt)
    print(str_txt)
    
N = 1000
p = {'setup': setup, 'number': N}
report("cuda_fftconv")
report("len(a) = 81000, len(b) = 80001")
report("Test on computer: %s" % os.environ['COMPUTERNAME'])
report("timeit N = %i" % N)
report("")
report("cuda_fftconv             : %.1f ms" %
       (timeit.timeit('cuda_fftconv()', **p)/N*1000))
report("fftconv_dq_np__dq_to_list: %.1f ms" %
       (timeit.timeit('fftconv_dq_np__dq_to_list()', **p)/N*1000))

try: f.close()
except: pass