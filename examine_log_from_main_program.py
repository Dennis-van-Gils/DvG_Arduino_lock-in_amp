#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dennis van Gils
26-08-2021
"""

import os
import sys

import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("dark_background")
marker = ".-"

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(0, 0)
fig.show()
fig.canvas.draw()

root = tk.Tk()
root.withdraw()
fn = filedialog.askopenfilename(
    initialdir=os.getcwd(), filetypes=[("Text files", "*.txt")]
)
if fn == "":
    sys.exit(0)
else:
    print(fn)

with open(fn, "r") as file:
    filedata = file.read()

# fmt: off
a = np.loadtxt(fn, skiprows=1)
time   = np.array(a[:, 0]) * 1e6
ref_X  = np.array(a[:, 1])
ref_Y  = np.array(a[:, 2])
sig_I  = np.array(a[:, 3])
filt_I = np.array(a[:, 4])
mix_X  = np.array(a[:, 5])
mix_Y  = np.array(a[:, 6])
X      = np.array(a[:, 7])
Y      = np.array(a[:, 8])
R      = np.array(a[:, 9])
T      = np.array(a[:, 10])
# fmt: on

# time = (time - time[0])

time_diff = np.diff(time)
print("time_diff:")
print("  median = %i usec" % np.median(time_diff))
print("  mean   = %i usec" % np.mean(time_diff))
print("  min    = %i usec" % np.min(time_diff))
print("  max    = %i usec" % np.max(time_diff))

time_ = time[:-1]
time_gaps = time_[time_diff > 500]
time_gap_durations = time_diff[time_diff > 500]
print("\nnumber of gaps > 500 usec: %i" % len(time_gaps))
for i in range(len(time_gaps)):
    print(
        "  gap %i @ t = %.3f msec for %.3f msec"
        % (i + 1, time_gaps[i] / 1e3, time_gap_durations[i] / 1e3)
    )

PEN_01 = (1, 30 / 255, 180 / 255)
PEN_02 = (1, 1, 90 / 255)
PEN_03 = (0, 1, 1)
PEN_04 = (1, 1, 1)

ax1 = plt.subplot(3, 1, 1)
plt.subplots_adjust(right=0.8)
plt.plot(time / 1e3, ref_X, marker, color=PEN_01, label="ref_X*")
plt.plot(time / 1e3, sig_I, marker, color=PEN_03, label="sig_I")
plt.grid()
plt.xlabel("time (ms)")
plt.title(fn)

ax2 = plt.subplot(3, 1, 2, sharex=ax1)
plt.plot(time / 1e3, filt_I, marker, color=PEN_04, label="filt_I")
plt.plot(time / 1e3, mix_X, marker, color=PEN_01, label="mix_X")
plt.grid()
plt.xlabel("time (ms)")

ax3 = plt.subplot(3, 1, 3, sharex=ax1)
plt.plot(time / 1e3, X, marker, color=PEN_01, label="X")
plt.grid()
plt.xlabel("time (ms)")

ax1.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
ax3.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
# fig.legend(loc=7)

fig.canvas.draw()
plt.show(block=True)
