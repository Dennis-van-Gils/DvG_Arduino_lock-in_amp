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


if __name__ == "__main__":
    print(cuda.gpus) # Reads '<Managed Device 0>' if a CUDA-enabled GPU is found
    
    a_np = np.zeros(40500)
    a_np[20250 - 2000 : 20250 + 2000] = 1.0    
    b_np = np.sin(2*np.pi*np.arange(4001)/1000)
    
    # Transfer to GPU memory
    # We also transform the 1-D array to a column vector, preferred by CUDA
    a_cp = cupy.array(a_np[:, None])
    b_cp = cupy.array(b_np[:, None])
    
    # Perform fft convolution on the GPU
    z_cp = sigpy.convolve(a_cp, b_cp, mode='valid')
    
    # Transfer result back to CPU memory
    z_np = cupy.asnumpy(z_cp)
    z_np = z_np[:, 0]         # Reduce dimension again
    
    plt.plot(np.arange(-len(z_np)/2, len(z_np)/2), z_np, '.-')
    plt.xlim([-5000, 5000])
    plt.show