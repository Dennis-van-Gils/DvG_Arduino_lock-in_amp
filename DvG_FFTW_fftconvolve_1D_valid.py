# -*- coding: utf-8 -*-
"""
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "01-08-2019"
__version__     = "1.0.0"

import numpy as np
from scipy import fftpack

def fftconvolve(in1, in2, mode=None):
    # Defaults:
    # in1.ndim = 1, not imaginary
    # in2.ndim = 1, not imaginary
    # mode = "valid"
    # axes = None
    # shape = None
    # complex_result = False
    
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    # Check that input sizes are compatible with 'valid' mode
    if in2.size > in1.size:
        in1, in2 = in2, in1
    
    # Speed up FFT by padding to optimal size for FFTPACK
    shape  = in1.size + in2.size - 1
    fshape = [fftpack.helper.next_fast_len(shape)]
    
    # Fourier transformations
    # Might apply zero-padding to speed up calculation
    sp1 = np.fft.rfftn(in1, fshape)
    sp2 = np.fft.rfftn(in2, fshape)
    ret = np.fft.irfftn(sp1 * sp2, fshape)[slice(shape)].copy()
    #ret = ret.real

    # Return valid convolve results
    newshape = in1.size - in2.size + 1
    startind = (ret.size - newshape) // 2
    myslice  = slice(startind, startind + newshape)

    return ret[myslice]

if __name__ == "__main__":    
    """Demo when run from main
    """
    import matplotlib.pyplot as plt
    from scipy.signal import firwin
    
    buffer_size = 500
    N_buffers_in_deque = 41
    
    N_taps  = buffer_size * (N_buffers_in_deque - 1) + 1
    N_deque = buffer_size * N_buffers_in_deque
    
    # Generate signal
    Fs = 5000                       # [Hz]
    t = np.arange(N_deque) * 1/Fs   # [s]
    f1 = 100                        # [Hz]
    f2 = 200                        # [Hz]
    f_fluct = 1 + np.sin(2*np.pi*1*t)/600
    wave  = np.sin(2*np.pi*(f1 * f_fluct)*t)
    wave += np.sin(2*np.pi*f2*t)
    
    b_np = firwin(N_taps,
                  cutoff=[190, 210],
                  window=("chebwin", 50),
                  pass_zero=True,
                  fs=Fs)
    
    result = fftconvolve(wave, b_np)
    
    plt.plot(result, '.-')
    plt.show
