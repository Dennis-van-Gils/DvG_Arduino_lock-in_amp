#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dennis van Gils
02-12-2018
"""

import os

import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(0, 0)
fig.show()
fig.canvas.draw()

#fn = "181202_223244.txt"

root = tk.Tk()
root.withdraw()
fn = filedialog.askopenfilename(initialdir=os.getcwd())
print(fn)

with open(fn, 'r') as file :
  filedata = file.read()
filedata = filedata.replace("draw", "")
filedata = filedata.replace("samples received: 200", "")

with open(fn, 'w') as file:
  file.write(filedata)

a = np.loadtxt(fn, skiprows=1)
time  = np.array(a[:, 0])/1e3
ref_X = np.array(a[:, 1])
ref_Y = np.array(a[:, 2])
sig_I = np.array(a[:, 3])

plt.plot(time - time[0], ref_X, 'x-r')
plt.plot(time - time[0], ref_Y, 'x-y')
plt.plot(time - time[0], sig_I, 'x-b')
plt.grid()
plt.xlabel("time (ms)")
plt.ylabel("voltage (V)")
plt.title(fn)

while (1):
    plt.pause(0.5)