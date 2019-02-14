# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 21:22:33 2018

@author: vangi
"""

import timeit

setup = '''
import numpy as np

a = range(40001)

def conversion(a):
    np.array(a)
'''
    
N = 100
print("%.2f ms" % (timeit.timeit('conversion(a)', setup=setup, number=N)/N*1000))