#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dennis van Gils
02-12-2018
"""

import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(0, 0)
fig.show()
fig.canvas.draw()

fn = "log.txt"

with open(fn, 'r') as file :
  filedata = file.read()
filedata = filedata.replace("draw", "")
filedata = filedata.replace("samples received: 200", "")

with open(fn, 'w') as file:
  file.write(filedata)

a = np.loadtxt(fn)
time        = np.array(a[:, 0])
phase_ref_X = np.array(a[:, 1])
sig_I       = np.array(a[:, 2])

ref_X = 2 + np.cos(2*np.pi*phase_ref_X/12288)
sig_I = sig_I / (2**12 - 1)*3.3

plt.plot(time - time[0], ref_X, 'x-r')
plt.plot(time - time[0], sig_I, 'x-b')
plt.grid()
plt.xlabel("time (us)")
plt.ylabel("voltage (V)")
plt.title(fn)

while (1):
    plt.pause(0.5)