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

a = np.loadtxt(fn, skiprows=1)
time  = np.array(a[:, 0])
ref_X = np.array(a[:, 1])
ref_Y = np.array(a[:, 2])
sig_I = np.array(a[:, 3])
time = time - time[0]

time_diff = np.diff(time)
print("time_diff:")
print("  median = %i usec" % np.median(time_diff))
print("  mean   = %i usec" % np.mean(time_diff))
print("  min = %i usec" % np.min(time_diff))
print("  max = %i usec" % np.max(time_diff))

time_ = time[:-1]
time_gaps = time_[time_diff > 500]
time_gap_durations = time_diff[time_diff > 500]
print("number of gaps > 500 usec: %i" % len(time_gaps))
for i in range(len(time_gaps)):
    print("  gap %i @ t = %.3f msec for %.3f msec" %
          (i+1, time_gaps[i]/1e3, time_gap_durations[i]/1e3))

plt.plot(time/1e3, ref_X, 'x-r')
plt.plot(time/1e3, ref_Y, 'x-y')
plt.plot(time/1e3, sig_I, 'x-b')
plt.grid()
plt.xlabel("time (ms)")
plt.ylabel("voltage (V)")
plt.title(fn)

while (1):
    plt.pause(0.5)