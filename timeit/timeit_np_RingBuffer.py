# -*- coding: utf-8 -*-
"""
30-07-2019
Dennis van Gils
"""

from timeit import timeit

setup = """
import numpy as np
from numpy_ringbuffer import RingBuffer
from collections import deque

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
from dvg_ringbuffer import RingBuffer as DvG_RingBuffer

np.random.seed(0)

N_buffers_passed = 100
buffer_size = 500
deque_size  = 20500

rb1 = RingBuffer(capacity=deque_size)
rb2 = DvG_RingBuffer(capacity=deque_size)
dq1 = deque(maxlen=deque_size)

def try1():
    for i in range(N_buffers_passed):
        rb1.extend(np.random.randn(buffer_size))

        if rb1.is_full:
            c = rb1[0:100]
            #d = np.asarray(rb2)

            #print(c.__array_interface__['data'][0])

def try2():
    for i in range(N_buffers_passed):
        rb2.extend(np.random.randn(buffer_size))

        if rb2.is_full:
            c = rb2[0:100]
            #d = np.asarray(rb2)

            #print(c.__array_interface__['data'][0])

def try3():
    for i in range(N_buffers_passed):
        dq1.extend(np.random.randn(buffer_size))

        if len(dq1) == dq1.maxlen:
            c = (np.array(dq1))[0:100]

            #print(c.__array_interface__['data'][0])

"""

N = 100
print("Numpy RingBuffer strategies")
print(
    "unwrap default    : %.3f ms"
    % (timeit("try1()", setup=setup, number=N) / N * 1000)
)

print(
    "unwrap into buffer: %.3f ms"
    % (timeit("try2()", setup=setup, number=N) / N * 1000)
)

print("Slow deque")
print(
    "deque             : %.3f ms"
    % (timeit("try3()", setup=setup, number=N) / N * 1000)
)
