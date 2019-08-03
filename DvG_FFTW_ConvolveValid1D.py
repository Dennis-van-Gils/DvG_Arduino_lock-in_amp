# -*- coding: utf-8 -*-
"""
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "02-08-2019"
__version__     = "1.0.0"

import numpy as np
import pyfftw

class FFTW_ConvolveValid1D:
    def __init__(self, len_in1, len_in2):
        
        # Check that input sizes are compatible with 'valid' mode
        self.switch_inputs = (len_in2 > len_in1)
        if self.switch_inputs:
            len_in1, len_in2 = len_in2, len_in1
            
        self.len_in1 = len_in1
        self.len_in2 = len_in2

        # Speed up FFT by zero-padding to optimal size for FFTW
        self.shape  = len_in1 + len_in2 - 1
        self.fshape = pyfftw.next_fast_len(self.shape)
        
        self.zero_pad_in1 = np.zeros(self.fshape - self.len_in1)
        self.zero_pad_in2 = np.zeros(self.fshape - self.len_in2)
        
        # Valid convolve results
        self.newshape = len_in1 - len_in2 + 1
        idx_start = (2 * len_in2 - 2) // 2
        self.valid_slice = slice(idx_start, idx_start + self.newshape)
        
        # Prepare the FFTW plans
        fshape_2 = self.fshape // 2 + 1
        self._rfft_in1  = pyfftw.empty_aligned(self.fshape, dtype='float64')
        self._rfft_out1 = pyfftw.empty_aligned(fshape_2   , dtype='complex128')

        self._rfft_in2  = pyfftw.empty_aligned(self.fshape, dtype='float64')
        self._rfft_out2 = pyfftw.empty_aligned(fshape_2   , dtype='complex128')

        self._irfft_in  = pyfftw.empty_aligned(fshape_2   , dtype='complex128')
        self._irfft_out = pyfftw.empty_aligned(self.fshape, dtype='float64')
        
        print("Creating FFTW plans for convolution...", end="")
        flags = ('FFTW_MEASURE', 'FFTW_DESTROY_INPUT')
        self._fftw_rfft1 = pyfftw.FFTW(self._rfft_in1, self._rfft_out1,
                                       flags=flags)
        self._fftw_rfft2 = pyfftw.FFTW(self._rfft_in2, self._rfft_out2,
                                       flags=flags)
        self._fftw_irfft = pyfftw.FFTW(self._irfft_in, self._irfft_out,
                                       direction='FFTW_BACKWARD',
                                       flags=flags)
        print(" done.")

    def process(self, in1, in2):
        in1 = np.asarray(in1)
        in2 = np.asarray(in2)
        
        if self.len_in1 != len(in1) or self.len_in2 != len(in2):
            return np.full(self.newshape, np.nan)
        
        # Check that input sizes are compatible with 'valid' mode
        if self.switch_inputs:
            in1, in2 = in2, in1
            
        # FFT convolve
        self._rfft_in1[:] = np.concatenate((in1, self.zero_pad_in1))
        self._rfft_in2[:] = np.concatenate((in2, self.zero_pad_in2))
        self._fftw_rfft1()
        self._fftw_rfft2()
        
        self._irfft_in[:] = self._rfft_out1 * self._rfft_out2 
        ret = self._fftw_irfft()
        
        return ret[self.valid_slice]

"""
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
    
    # Speed up FFT by padding with 0's to optimal size for FFTPACK
    shape  = in1.size + in2.size - 1
    fshape = fftpack.helper.next_fast_len(shape)
    
    # Fourier transformations
    # Might apply zero-padding to speed up calculation
    sp1 = np.fft.rfftn(in1, [fshape])
    sp2 = np.fft.rfftn(in2, [fshape])
    ret = np.fft.irfftn(sp1 * sp2, [fshape])[:shape].copy()
    #ret = ret.real

    # Return valid convolve results
    newshape = in1.size - in2.size + 1
    startind = (ret.size - newshape) // 2
    #startind = (2 * in2.size - 2) // 2

    return ret[startind:startind + newshape]

if __name__ == "__main__":    
    # Demo when run from main
    import matplotlib.pyplot as plt
    from scipy.signal import firwin
    from scipy import fftpack
    
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
"""