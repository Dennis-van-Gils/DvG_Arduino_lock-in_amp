# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

from timeit import timeit

setup = '''
import numpy as np

np.random.seed(0)

N_array = 20000
a_np = np.random.randn(N_array)
b_np = np.random.randn(N_array)
c_np = np.empty(N_array, dtype=np.float64)

def ufunc_0(a_np, b_np, c_np):
    a_np = np.random.randn(N_array)
    b_np = np.random.randn(N_array)
    
    c_np = np.sqrt(a_np**2 + b_np**2)
    
def ufunc_1(a_np, b_np, c_np):
    a_np = np.random.randn(N_array)
    b_np = np.random.randn(N_array)

    np.sqrt(a_np**2 + b_np**2, out=c_np)
    
def ufunc_2(a_np, b_np, c_np):
    a_np = np.random.randn(N_array)
    b_np = np.random.randn(N_array)

    #np.power(a_np, 2, out=a_np)
    #np.power(b_np, 2, out=b_np)
    a_np **= 2
    b_np **= 2
    
    np.add(a_np, b_np, out=c_np)
    np.sqrt(c_np, out=c_np)
    
'''
    
N = 1000
print("Numpy multiply strategies")
print("ufunc_0: %.3f ms" %
      (timeit('ufunc_0(a_np, b_np, c_np)', setup=setup, number=N)/N*1000))

print("ufunc_1: %.3f ms" %
      (timeit('ufunc_1(a_np, b_np, c_np)', setup=setup, number=N)/N*1000))
      
print("ufunc_2: %.3f ms" %
      (timeit('ufunc_2(a_np, b_np, c_np)', setup=setup, number=N)/N*1000))