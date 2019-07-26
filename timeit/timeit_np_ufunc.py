# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

from timeit import timeit

setup = '''
import numpy as np

np.random.seed(0)

N_array = 200000
a_np = np.random.randn(N_array)
b_np = np.random.randn(N_array)
c_np = np.empty(N_array)

def no_ufunc(a_np, b_np):
    c_np = a_np * b_np
    #print(c_np[0])
    
def ufunc_buffered(a_np, b_np):
    c_np = np.multiply(a_np, b_np)
    #print(c_np[0])
    
def ufunc_unbuffered(a_np, b_np):
    np.multiply(a_np, b_np, out=c_np)
    #print(c_np[0])
'''
    
N = 1000
print("Numpy multiply strategies")
print("no ufunc    : %.3f ms" %
      (timeit('no_ufunc(a_np, b_np)', setup=setup, number=N)/N*1000))

print("ufunc buf   : %.3f ms" %
      (timeit('ufunc_buffered(a_np, b_np)', setup=setup, number=N)/N*1000))

print("ufunc no buf: %.3f ms" %
      (timeit('ufunc_unbuffered(a_np, b_np)', setup=setup, number=N)/N*1000))