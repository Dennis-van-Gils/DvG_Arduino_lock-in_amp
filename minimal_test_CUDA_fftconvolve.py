# -*- coding: utf-8 -*-
"""
Minimal test case for CUDA hardware support on NVidia GPU's using numba.

Dennis van Gils
07-04-2019
"""

from __future__ import division
from numba import cuda
import numpy as np
import cupy
import sigpy

"""FFTconvolve example C++
cufftHandle _planKernel                             // you fft handle
cufftPlan1d(&_planKernel, _fftLen, CUFFT_C2C, 1);   // create 1D fft handle
cufftComplex* VECTOR1, *VECTOR2, *PRODUCT;
MakeVector1Complex<<<blockSize, GridSize>>>()       // simply set real part of the VECTOR1 = vector1, and set the imaginary part VECTOR to 0 
MakeVector2Complex<<<blockSize, GridSize>>>()       // simply set real part of the VECTOR2 = vector2, and set the imaginary part VECTOR to 0
cufftExecC2C(planKernel, VECTOR1, VECTOR1, CUFFT_FORWARD);  // apply fft to VECTOR1
cufftExecC2C(planKernel, VECTOR2, VECTOR2, CUFFT_FORWARD);  // apply fft to VECTOR2
ComplexMutiplication<<<blockSize, GridSize>>>(VECTOR1, VECTOR2) // complex multiplication of VECTOR1 and VECTOR2

cufftExecC2C(planKernel, PRODUCT, PRODUCT, CUFFT_INVERSE); // inverse fft on the product of VECTOR1 AND VECTOR2

MakeProductReal<<<blockSize, GridSize>>>(PRODUCT)   // extract the real part of PRODUCT
"""

if __name__ == "__main__":
    print(cuda.gpus) # Reads '<Managed Device 0>' if a CUDA-enabled GPU is found
    
    a = np.identity(int(1e4), dtype=np.float32)
    b = np.identity(int(1e4), dtype=np.float32)
    c = np.empty((int(1e4), int(1e4)), dtype=np.float32)
    
    x = cupy.array([1, 2, 3, 4, 5])
    y = cupy.array([1, 1, 1])
    z = sigpy.convolve(a, b)
    