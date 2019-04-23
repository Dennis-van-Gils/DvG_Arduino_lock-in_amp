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

# NumPy arrays
#np.random.seed(0) # Seed is not set for real test
a_np = np.random.randn(N_deque)
b_np = firwin(N_taps,
              [0.0001, 0.015, 0.025, .5-1e-4],
              window=("chebwin", 50),
              fs=1,
              pass_zero=False)

# Regular lists
a_ls = a_np.tolist()
b_ls = b_np.tolist()

# Deque
a_dq = deque(maxlen=N_deque)
a_dq.extend(a_np)


def wrap(answer):
    #print(np.sum(answer), end=' ')
    pass
    

def conv_sp_basic(a, b):
    wrap(sp.convolve(a, b, mode='valid'))
    
def conv_sp_a_list(a, b):
    wrap(sp.convolve(list(a), b, mode='valid'))

def conv_sp_a_nparr(a, b):
    wrap(sp.convolve(np.array(a), b, mode='valid'))

def conv_sp_all_list(a, b):
    wrap(sp.convolve(list(a), b.tolist(), mode='valid'))


def conv_np_basic(a, b):
    wrap(np.convolve(a, b, mode='valid'))
        
def conv_np_a_list(a, b):
    wrap(np.convolve(list(a), b, mode='valid'))
    
def conv_np_a_nparr(a, b):
    wrap(np.convolve(np.array(a), b, mode='valid'))
    
def conv_np_all_list(a, b):
    wrap(np.convolve(list(a), b.tolist(), mode='valid'))


def fftconv_basic(a, b):
    wrap(fftconvolve(a, b, mode='valid'))
    
def fftconv_a_list(a, b):
    wrap(fftconvolve(list(a), b, mode='valid'))

def fftconv_a_nparr(a, b):
    wrap(fftconvolve(np.array(a), b, mode='valid'))
    
def fftconv_all_list(a, b):
    wrap(fftconvolve(list(a), b.tolist(), mode='valid'))

'''

try: f = open("timeit_convolve.log", "w")
except: f = None

def report(str_txt):
    if not(f == None):
        f.write("%s\n" % str_txt)
    print(str_txt)
    
N = 10000
p = {'setup': setup, 'number': N}
report("convolve(a, b)")
report("where a and b can each be cast into other types")
report("len(a) = 20500, len(b) = 20001")
report("Test on computer: %s" % os.environ['COMPUTERNAME'])
report("timeit N = %i" % N)

# NOTE: I took out the scipy.signal.convolve tests because their
# resuls match with numpy.convolve
# NOTE 2: I took out regular convolve tests because 
# scipy.signal.fftconvolve is superior

report("\n------ list, list\n")

#report("conv_sp_basic   (ls, ls): %.1f ms" %
#       (timeit.timeit('conv_sp_basic(a_ls, b_ls)', **p)/N*1000))
#report("conv_np_basic   (ls, ls): %.1f ms" %
#       (timeit.timeit('conv_np_basic(a_ls, b_ls)', **p)/N*1000))
report("fftconv_basic   (ls, ls): %.1f ms" %
       (timeit.timeit('fftconv_basic(a_ls, b_ls)', **p)/N*1000))

report("\n------ deque, numpy array\n")

#report("conv_sp_basic   (dq, np): %.1f ms" %
#       (timeit.timeit('conv_sp_basic(a_dq, b_np)', **p)/N*1000))
#report("conv_np_basic   (dq, np): %.1f ms" %
#       (timeit.timeit('conv_np_basic(a_dq, b_np)', **p)/N*1000))
report("fftconv_basic   (dq, np): %.1f ms" %
       (timeit.timeit('fftconv_basic(a_dq, b_np)', **p)/N*1000))
report("")
#report("conv_sp_a_list  (dq, np): %.1f ms" %
#       (timeit.timeit('conv_sp_a_list(a_dq, b_np)', **p)/N*1000))
#report("conv_np_a_list  (dq, np): %.1f ms" %
#       (timeit.timeit('conv_np_a_list(a_dq, b_np)', **p)/N*1000))
report("fftconv_a_list  (dq, np): %.1f ms" %
       (timeit.timeit('fftconv_a_list(a_dq, b_np)', **p)/N*1000))
report("")
#report("conv_sp_a_nparr (dq, np): %.1f ms" %
#       (timeit.timeit('conv_sp_a_nparr(a_dq, b_np)', **p)/N*1000))
#report("conv_np_a_nparr (dq, np): %.1f ms" %
#       (timeit.timeit('conv_np_a_nparr(a_dq, b_np)', **p)/N*1000))
report("fftconv_a_nparr (dq, np): %.1f ms" %
       (timeit.timeit('fftconv_a_nparr(a_dq, b_np)', **p)/N*1000))
report("")
#report("conv_sp_all_list(dq, np): %.1f ms" %
#       (timeit.timeit('conv_sp_all_list(a_dq, b_np)', **p)/N*1000))
#report("conv_np_all_list(dq, np): %.1f ms" %
#       (timeit.timeit('conv_np_all_list(a_dq, b_np)', **p)/N*1000))
report("fftconv_all_list(dq, np): %.1f ms" %
       (timeit.timeit('fftconv_all_list(a_dq, b_np)', **p)/N*1000))

report("\n------ deque, list\n")

#report("conv_sp_basic   (dq, ls): %.1f ms" %
#       (timeit.timeit('conv_sp_basic(a_dq, b_ls)', **p)/N*1000))
#report("conv_np_basic   (dq, ls): %.1f ms" %
#       (timeit.timeit('conv_np_basic(a_dq, b_ls)', **p)/N*1000))
report("fftconv_basic   (dq, ls): %.1f ms" %
       (timeit.timeit('fftconv_basic(a_dq, b_ls)', **p)/N*1000))
report("")
#report("conv_sp_a_list  (dq, ls): %.1f ms" %
#       (timeit.timeit('conv_sp_a_list(a_dq, b_ls)', **p)/N*1000))
#report("conv_np_a_list  (dq, ls): %.1f ms" %
#       (timeit.timeit('conv_np_a_list(a_dq, b_ls)', **p)/N*1000))
report("fftconv_a_list  (dq, ls): %.1f ms" %
       (timeit.timeit('fftconv_a_list(a_dq, b_ls)', **p)/N*1000))

try: f.close()
except: pass
