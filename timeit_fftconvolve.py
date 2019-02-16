# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import timeit
import os

setup = '''
import numpy as np
import scipy as sp
from scipy.signal import firwin, fftconvolve
from collections import deque

buffer_size = 500
N_buffers_in_deque = 41

N_taps  = buffer_size * (N_buffers_in_deque - 1) + 1
N_deque = buffer_size * N_buffers_in_deque

np.random.seed(0)

b_np = firwin(N_taps,
              [0.0001, 0.015, 0.025, .5-1e-4],
              window=("chebwin", 50),
              fs=1,
              pass_zero=False)
b_ls = b_np.tolist()

def init_a():
    a_np = np.random.randn(N_deque)
    a_ls = a_np.tolist()
    a_dq = deque(a_ls, maxlen=N_deque)
    
    return (a_np, a_ls, a_dq)
  
def fftconv_ls_ls():
    (a_np, a_ls, a_dq) = init_a()
    fftconvolve(a_ls, b_ls, mode='valid')

def fftconv_dq_np():
    (a_np, a_ls, a_dq) = init_a()
    fftconvolve(a_dq, b_np, mode='valid')
    
def fftconv_dq_np__dq_to_list():
    (a_np, a_ls, a_dq) = init_a()
    fftconvolve(list(a_dq), b_np, mode='valid')

def fftconv_dq_np__dq_to_nparr():
    (a_np, a_ls, a_dq) = init_a()
    fftconvolve(np.array(a_dq), b_np, mode='valid')
    
def fftconv_dq_np__all_to_list():
    (a_np, a_ls, a_dq) = init_a()
    fftconvolve(list(a_dq), b_np.tolist(), mode='valid')

'''

try: f = open("timeit_fftconvolve.log", "w")
except: f = None

def report(str_txt):
    if not(f == None):
        f.write("%s\n" % str_txt)
    print(str_txt)
    
N = 10000
p = {'setup': setup, 'number': N}
report("fftconvolve(a, b)")
report("where a and b can each be cast into other types")
report("len(a) = 20500, len(b) = 20001")
report("Test on computer: %s" % os.environ['COMPUTERNAME'])
report("timeit N = %i" % N)
report("")
report("fftconv_ls_ls             : %.1f ms" %
       (timeit.timeit('fftconv_ls_ls()', **p)/N*1000))
report("fftconv_dq_np             : %.1f ms" %
       (timeit.timeit('fftconv_dq_np()', **p)/N*1000))
report("fftconv_dq_np__dq_to_list : %.1f ms" %
       (timeit.timeit('fftconv_dq_np__dq_to_list()', **p)/N*1000))
report("fftconv_dq_np__dq_to_nparr: %.1f ms" %
       (timeit.timeit('fftconv_dq_np__dq_to_nparr()', **p)/N*1000))
report("fftconv_dq_np__all_to_list: %.1f ms" %
       (timeit.timeit('fftconv_dq_np__all_to_list()', **p)/N*1000))

try: f.close()
except: pass