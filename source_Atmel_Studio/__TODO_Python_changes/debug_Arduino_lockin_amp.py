#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dennis_van_Gils
29-03-2019
"""

import os
import sys
import msvcrt
import psutil
import time as Time
from pathlib2 import Path

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions

fn_log = "log.txt"
fDrawPlot = False
fVerbose = False

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    lockin = lockin_functions.Arduino_lockin_amp(baudrate=1.2e6, read_timeout=2)
    if not lockin.auto_connect(Path("port_data.txt"), "Arduino lock-in amp"):
        sys.exit(0)
    lockin.begin(ref_freq=250, ref_V_offset=1.65, ref_V_ampl=1.65)
    
    if fDrawPlot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(0, 0)
        fig.show()
        fig.canvas.draw()
    f_log = open(fn_log, 'w')

    N_SETS = 2;
    N_REPS = 2;
    
    N_deque = lockin.config.BLOCK_SIZE * N_REPS;
    deque_time  = deque(maxlen=N_deque)
    deque_ref_X = deque(maxlen=N_deque)
    deque_sig_I = deque(maxlen=N_deque)
    samples_received = np.array([], dtype=int)
    
    lockin.turn_on()
    for i_set in range(N_SETS): 
		#if i_set == 1: lockin.
        #if i_set == 2: lockin.set_ref_V_ampl(0.6
        if i_set == 1: lockin.set_ref_freq(2500)
        if i_set == 2: lockin.set_ref_V_ampl(0.6)
        
        deque_time.clear()
        deque_ref_X.clear()
        deque_sig_I.clear()
    
        tick = 0;
        buffers_received = 0;
        time_prev = 0;
        for i_rep in range(N_REPS):            
            #if msvcrt.kbhit() and msvcrt.getch().decode() == chr(27):
            #    sys.exit(0)
            
            [success, time, ref_X, tmp3, sig_I] = lockin.listen_to_lockin_amp()
    
            if success:
                buffers_received += 1
                
                if tick == 0:
                    tick = Time.perf_counter()
                    
                N_samples = len(time)
                if fVerbose: print("%3d: %d" % (i_rep, N_samples))
                
                f_log.write("samples received: %i\n" % N_samples)
                for i in range(N_samples):
                    f_log.write("%i\t%.3f\t%.3f\n" % (time[i], ref_X[i], sig_I[i]))
                
                deque_time.extend(time)
                deque_ref_X.extend(ref_X)
                deque_sig_I.extend(sig_I)
                samples_received = np.append(samples_received, N_samples)
        
        f_log.write("draw\n")
        
        if fVerbose or fDrawPlot: 
            np_time = np.array(deque_time)
            np_time = np_time - np_time[0]        
            dt = np.diff(np_time)
            
            Fs = 1/np.mean(dt)*1e6
            str_info1 = ("Fs = %.2f Hz    dt_min = %d us    dt_max = %d us" % 
                        (Fs, np.min(dt), np.max(dt)))
            str_info2 = ("N_buf = %d     %.2f buf/s" % 
                        (buffers_received,
                         buffers_received/(Time.perf_counter() - tick)))
            print(str_info1 + "    " + str_info2)
        
        if fDrawPlot:
            ax.cla()
            ax.plot(np_time/1e3, deque_ref_X, '.-k')
            ax.plot(np_time/1e3, deque_sig_I, '.-r')
            ax.set(xlabel='time [ms]', ylabel='y',
                   title=(str_info1 + '\n' + str_info2))
            ax.grid()            
            #ax.set(xlim=(90, 110))
                        
            fig.canvas.draw()
            plt.pause(0.1)
    
    # Finish test
    lockin.turn_off()
    lockin.close()
    f_log.close()
    
    print("\nSamples received per buffer: [min, max] = [%d, %d]" % 
          (np.min(samples_received), np.max(samples_received)))
    
    while (1):
        if fDrawPlot:
            plt.pause(0.5)
        else:
            break
            
        if msvcrt.kbhit(): # and msvcrt.getch().decode() == chr(27):
            break

            
            
            
            
    with open(fn_log, 'r') as file :
        filedata = file.read()
    filedata = filedata.replace("draw", "")
    filedata = filedata.replace("samples received: 2500", "")

    with open(fn_log, 'w') as file:
        file.write(filedata)

    a = np.loadtxt(fn_log)
    time  = np.array(a[:, 0])
    ref_X = np.array(a[:, 1])
    sig_I = np.array(a[:, 2])
    time = time - time[0]

    time_diff = np.diff(time)
    print("\ntime_diff:")
    print("  median = %i usec" % np.median(time_diff))
    print("  mean   = %i usec" % np.mean(time_diff))
    print("  min = %i usec" % np.min(time_diff))
    print("  max = %i usec" % np.max(time_diff))

    time_ = time[:-1]
    time_gaps = time_[time_diff > 500]
    time_gap_durations = time_diff[time_diff > 500]
    print("\nnumber of gaps > 500 usec: %i" % len(time_gaps))
    for i in range(len(time_gaps)):
        print("  gap %i @ t = %.3f msec for %.3f msec" %
              (i+1, time_gaps[i]/1e3, time_gap_durations[i]/1e3))