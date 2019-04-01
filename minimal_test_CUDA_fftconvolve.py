# -*- coding: utf-8 -*-
"""
Minimal test case for CUDA hardware support on NVidia GPU's using numba.

Dennis van Gils
31-03-2019
"""

from __future__ import division
from numba import cuda
import numpy
import math

# CUDA kernel
@cuda.jit
def my_kernel(io_array):
    pos = cuda.grid(1)
    if pos < io_array.size:
        io_array[pos] *= 2 # do the computation

# Host code   
data = numpy.ones(256)
threadsperblock = 256
blockspergrid = math.ceil(data.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](data)
print(data)

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

"""
import numpy as np
from numba import cuda, float32

@cuda.jit
def matmul(A, B, C):
    # Perform square matrix multiplication of C = A * B
    i, j = cuda.grid(2)
    if i < C.shape[0] and j < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[i, k] * B[k, j]
        C[i, j] = tmp

# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16

@cuda.jit
def fast_matmul(A, B, C):
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(bpg):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp
    
if __name__ == "__main__":
    print(cuda.gpus) # Reads '<Managed Device 0>' if a CUDA-enabled GPU is found
    
    a = np.identity(int(1e4), dtype=np.float32)
    b = np.identity(int(1e4), dtype=np.float32)
    c = np.empty((int(1e4), int(1e4)), dtype=np.float32)
    
    matmul(a, b, c)
"""