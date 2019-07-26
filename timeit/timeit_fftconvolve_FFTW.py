# -*- coding: utf-8 -*-
"""
Dennis van Gils
26-07-2019

Under test: fast Fourier transforms

valid_out = scipy.signal.fftconvolve(deque_sig_in, self.b, mode='valid')

"""

import timeit
import os

setup = '''
import numpy as np
from collections import deque
import scipy
from scipy import signal

buffer_size = 500
N_buffers_in_deque = 41

N_taps  = buffer_size * (N_buffers_in_deque - 1) + 1
N_deque = buffer_size * N_buffers_in_deque

np.random.seed(0)

b_np = signal.firwin(N_taps,
                     [0.0001, 0.015, 0.025, .5-1e-4],
                     window=("chebwin", 50),
                     fs=1,
                     pass_zero=False)
 
def init_a():
    a_np = np.random.randn(N_deque)
    #a_dq = deque(a_np, maxlen=N_deque)
    a_dq = None
    
    return (a_np, a_dq)
 
def fftconv_scipy():
    (a_np, a_dq) = init_a()
    signal.fftconvolve(a_np, b_np, mode='valid')
'''

setup_fftw = '''
import numpy as np
from collections import deque
import scipy
from scipy import signal
import pyfftw
import multiprocessing

buffer_size = 500
N_buffers_in_deque = 41

N_taps  = buffer_size * (N_buffers_in_deque - 1) + 1
N_deque = buffer_size * N_buffers_in_deque

np.random.seed(0)

b_np = signal.firwin(N_taps,
                     [0.0001, 0.015, 0.025, .5-1e-4],
                     window=("chebwin", 50),
                     fs=1,
                     pass_zero=False)

a_fftw = pyfftw.empty_aligned(N_deque, dtype='float64')
b_fftw = pyfftw.empty_aligned(N_taps, dtype='float64')
b_fftw[:] = b_np

# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

# Monkey patch fftpack with pyfftw.interfaces.scipy_fftpack
scipy.fftpack = pyfftw.interfaces.scipy_fftpack

# We cheat a bit by doing the planning first
a_np = np.random.randn(N_deque)
a_fftw[:] = a_np
signal.fftconvolve(a_fftw, b_fftw)

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

def init_a():
    a_np = np.random.randn(N_deque)
    #a_dq = deque(a_np, maxlen=N_deque)
    a_dq = None
    
    return (a_np, a_dq)

def fftconv_fftw():
    (a_np, a_dq) = init_a()
    a_fftw[:] = a_np
    #a_fftw[:] = a_dq
    #a_fftw[:] = np.asarray(a_dq, dtype=np.float64)
    #a_fftw[:] = list(a_dq)
    signal.fftconvolve(a_fftw, b_fftw, mode='valid')

'''

try: f = open("timeit_fftconvolve_FFTW.log", "w")
except: f = None

def report(str_txt):
    if not(f == None):
        f.write("%s\n" % str_txt)
    print(str_txt)
    
N = 10000
p = {'setup': setup, 'number': N}
p_fftw = {'setup': setup_fftw, 'number': N}
report("fftconvolve(a, b)")
report("len(a) = 20500, len(b) = 20001")
report("Test on computer: %s" % os.environ['COMPUTERNAME'])
report("timeit N = %i" % N)
report("")
report("fftconv_scipy             : %.1f ms" %
       (timeit.timeit('fftconv_scipy()', **p)/N*1000))
report("fftconv_fftw              : %.1f ms" %
       (timeit.timeit('fftconv_fftw()', **p_fftw)/N*1000))

try: f.close()
except: pass
