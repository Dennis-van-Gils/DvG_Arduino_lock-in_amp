# -*- coding: utf-8 -*-
"""
Minimal test case for CUDA hardware support on NVidia GPU's using numba.

Dennis van Gils
23-04-2019
"""

import matplotlib.pyplot as plt
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
    
    a_np = np.identity(1000)
    b_np = np.ones((100, 100))
    
    # Transfer to GPU memory
    a_cp = cupy.array(a_np)
    b_cp = cupy.array(b_np)
    
    # Perform fft convolution on the GPU
    z_cp = sigpy.convolve(a_cp, b_cp, mode='valid')
    
    # Transfer result back to CPU memory
    z_np = cupy.asnumpy(z_cp)
    z_np = z_np/np.max(z_np)
    
    plt.imshow(z_np)
    plt.gray()
    plt.show()
