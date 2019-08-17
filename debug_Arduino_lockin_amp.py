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
from DvG_dev_Arduino_lockin_amp__fun_serial import Waveform

fn_log = "log.txt"
fDrawPlot = True

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    lockin = lockin_functions.Arduino_lockin_amp(baudrate=1.2e6)
    if not lockin.auto_connect(Path("port_data.txt"), "Arduino lock-in amp"):
        sys.exit(0)
    lockin.begin(ref_freq=100,
                 ref_V_offset=1.5,
                 ref_V_ampl=0.5, 
                 ref_waveform=Waveform.Cosine)
    
    if fDrawPlot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(0, 0)
        fig.show()
        fig.canvas.draw()
    f_log = open(fn_log, 'w')

    N_SETS = 3;
    N_REPS = 41;
    
    N_deque = lockin.config.BLOCK_SIZE * N_REPS;
    deque_time  = deque(maxlen=N_deque)
    deque_ref_X = deque(maxlen=N_deque)
    deque_sig_I = deque(maxlen=N_deque)
    samples_received = np.array([], dtype=int)
    
    lockin.turn_on()
    for i_set in range(N_SETS): 
        if i_set == 1: lockin.set_ref_waveform(Waveform.Triangle)
        if i_set == 2: lockin.set_ref_waveform(Waveform.Cosine)
        
        deque_time.clear()
        deque_ref_X.clear()
        deque_sig_I.clear()
    
        tick = 0;
        buffers_received = 0;
        for i_rep in range(N_REPS):            
            #if msvcrt.kbhit() and msvcrt.getch().decode() == chr(27):
            #    sys.exit(0)
            
            [success, time, ref_X, ref_Y, sig_I] = lockin.listen_to_lockin_amp()
            
            if success:
                buffers_received += 1
                
                if tick == 0:
                    tick = Time.perf_counter()
                    
                N_samples = len(time)
                print("%3d: %d" % (i_rep, N_samples))
                f_log.write("samples received: %i\n" % N_samples)
                for i in range(N_samples):
                    f_log.write("%i\t%.4f\t%.3f\n" % (time[i], ref_X[i], sig_I[i]))
                
                deque_time.extend(time)
                deque_ref_X.extend(ref_X)
                deque_sig_I.extend(sig_I)
                samples_received = np.append(samples_received, N_samples)
        
        f_log.write("draw\n")
        
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
            #lockin.turn_off()
            
            ax.cla()
            ax.plot(np_time/1e3, deque_ref_X, 'x-k')
            ax.plot(np_time/1e3, deque_sig_I, 'x-r')
            ax.set(xlabel='time (ms)', ylabel='y',
                   title=(str_info1 + '\n' + str_info2))
            ax.grid()            
            ax.set(xlim=(0, 80))
                        
            fig.canvas.draw()
            plt.pause(0.1)
            
            #lockin.turn_on()
    
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
            Time.sleep(0.1)
            
        if msvcrt.kbhit() and msvcrt.getch().decode() == chr(27):
            sys.exit(0)