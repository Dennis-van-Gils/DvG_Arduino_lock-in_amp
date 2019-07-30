# -*- coding: utf-8 -*-
"""
Dennis van Gils
26-07-2019

Under test: fast Fourier transforms

valid_out = scipy.signal.fftconvolve(deque_sig_in, self.b, mode='valid')

"""

import timeit
import os

setup_scipy = '''
import numpy as np
from collections import deque
import scipy
from scipy import signal

np.random.seed(0)
buffer_size = 500
N_buffers_in_deque = 41
N_taps  = buffer_size * (N_buffers_in_deque - 1) + 1
N_deque = buffer_size * N_buffers_in_deque

b_np = signal.firwin(N_taps,
                     [0.0001, 0.015, 0.025, .5-1e-4],
                     window=("chebwin", 50),
                     fs=1,
                     pass_zero=False)



def init_a():
    return np.random.randn(N_deque)
    
    

def fftconv_scipy():
    a_np = init_a()
    signal.fftconvolve(a_np, b_np, mode='valid')
'''

setup_fftw = '''
import numpy as np
from collections import deque
import scipy
from scipy import signal
import pyfftw
import multiprocessing

np.random.seed(0)
buffer_size = 500
N_buffers_in_deque = 41
N_taps  = buffer_size * (N_buffers_in_deque - 1) + 1
N_deque = buffer_size * N_buffers_in_deque

# Configure PyFFTW to use all cores (the default is single-threaded)
pyfftw.config.NUM_THREADS = multiprocessing.cpu_count()

# Monkey patch fftpack with pyfftw.interfaces.scipy_fftpack
scipy.fftpack = pyfftw.interfaces.scipy_fftpack

# Turn on the cache for optimum performance
pyfftw.interfaces.cache.enable()

b_np = signal.firwin(N_taps,
                     [0.0001, 0.015, 0.025, .5-1e-4],
                     window=("chebwin", 50),
                     fs=1,
                     pass_zero=False)

a_fftw = pyfftw.empty_aligned(N_deque, dtype='float64')
b_fftw = pyfftw.empty_aligned(N_taps, dtype='float64')
b_fftw[:] = b_np



def init_a():
    return np.ascontiguousarray(np.random.randn(N_deque))    
    #return np.random.randn(N_deque)



def fftconv_fftw():
    a_np = init_a()
    #print(a_np.__array_interface__['data'][0])
    a_fftw[:] = a_np
    signal.fftconvolve(a_fftw, b_fftw, mode='valid')
'''

try: f = open("timeit_fftconvolve_FFTW.log", "w")
except: f = None

def report(str_txt):
    if not(f == None):
        f.write("%s\n" % str_txt)
    print(str_txt)
    
N = 2000
p_scipy = {'setup': setup_scipy, 'number': N, 'repeat': 3}
p_fftw  = {'setup': setup_fftw , 'number': N, 'repeat': 3}
report("fftconvolve(a, b)")
report("len(a) = 20500, len(b) = 20001")
report("Test on computer: %s" % os.environ['COMPUTERNAME'])
report("timeit N = %i" % N)
report("")
report("fftconv_scipy: %.1f ms" %
       (min(timeit.repeat('fftconv_scipy()', **p_scipy))/N*1000))
report("fftconv_fftw : %.1f ms" %
       (min(timeit.repeat('fftconv_fftw()', **p_fftw))/N*1000))

try: f.close()
except: pass
