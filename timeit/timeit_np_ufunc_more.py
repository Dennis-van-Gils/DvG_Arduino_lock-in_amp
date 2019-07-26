# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

from timeit import timeit

setup = '''
import numpy as np
import scipy as sc

np.random.seed(0)

N_array = 200000
D = np.double(2.0)
a_np = np.random.randn(N_array)
b_np = np.random.randn(N_array)
c_np = np.empty(N_array, dtype=np.float64)

class State():
    def __init__(self):
        self.N_array = 200000
        self.D = np.double(2.0)
        self.a_np = np.random.randn(self.N_array)
        self.b_np = np.random.randn(self.N_array)
        self.c_np = np.empty(self.N_array, dtype=np.float64)
        
state = State()

def no_ufunc(a_np, b_np, c_np, D):
    c_np = (a_np - D) * b_np
    
def ufunc_1(a_np, b_np, c_np, D):
    c_np = np.multiply((a_np - D), b_np)
    
def ufunc_2(a_np, b_np, c_np, D):
    np.multiply((a_np - D), b_np, out=c_np)
    
def ufunc_3(a_np, b_np, c_np, D):
    np.subtract(a_np, D, out=a_np)
    np.multiply(a_np, b_np, out=c_np)
    
def ufunc_4(state):
    np.subtract(state.a_np, state.D, out=state.a_np)
    np.multiply(state.a_np, state.b_np, out=state.c_np)
    
def ufunc_5(state):
    sc.subtract(state.a_np, state.D, out=state.a_np)
    sc.multiply(state.a_np, state.b_np, out=state.c_np)
    
def ufunc_6(state):
    state.a_np -= state.D
    np.multiply(state.a_np, state.b_np, out=state.c_np)
'''
    
N = 1000
print("Numpy multiply strategies")
print("no ufunc: %.3f ms" %
      (timeit('no_ufunc(a_np, b_np, c_np, D)', setup=setup, number=N)/N*1000))

print("ufunc_1 : %.3f ms" %
      (timeit('ufunc_1(a_np, b_np, c_np, D)', setup=setup, number=N)/N*1000))

print("ufunc_2 : %.3f ms" %
      (timeit('ufunc_2(a_np, b_np, c_np, D)', setup=setup, number=N)/N*1000))
      
print("ufunc_3 : %.3f ms" %
      (timeit('ufunc_3(a_np, b_np, c_np, D)', setup=setup, number=N)/N*1000))
      
print("ufunc_4 : %.3f ms" %
      (timeit('ufunc_4(state)', setup=setup, number=N)/N*1000))
      
print("ufunc_5 : %.3f ms" %
      (timeit('ufunc_5(state)', setup=setup, number=N)/N*1000))

print("ufunc_6 : %.3f ms" %
      (timeit('ufunc_6(state)', setup=setup, number=N)/N*1000))