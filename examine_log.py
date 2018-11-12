# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 01:56:46 2018

@author: vangi
"""

import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
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

a = np.loadtxt("log.txt")
time  = a[:, 0]
ref_X = a[:, 1]
sig_I = a[:, 2]

plt.plot(time - time[0], ref_X/2**10*3.3, 'x-k')
plt.plot(time - time[0], sig_I/2**12*3.3, 'x-r')

while (1):
    plt.pause(0.5)