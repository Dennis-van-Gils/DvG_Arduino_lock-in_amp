#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
02-12-2018
Dennis_van_Gils
"""

import os
import sys
import struct
import msvcrt
import psutil
import time as Time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import DvG_dev_Arduino_lockin_amp__fun_serial as lockin_functions

fn_log = "log.txt"
fDrawPlot = True

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)

    lockin = lockin_functions.Arduino_lockin_amp(baudrate=1.5e6)
    if not lockin.auto_connect(Path("port_data.txt"), "Arduino lock-in amp"):
        sys.exit(0)
    
    if fDrawPlot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(0, 0)
        fig.show()
        fig.canvas.draw()

    f_log = open(fn_log, 'w')
    
    lockin.begin()
    lockin.set_ref_freq(137)
    lockin.turn_on()

    samples_received = np.array([], dtype=int)
    for uber_counter in range(40): 
        #if uber_counter == 2: lockin.set_ref_freq(20)
        
        full_time  = np.array([], dtype=int)
        full_ref_X = np.array([], dtype=int)
        full_sig_I = np.array([], dtype=int)
    
        time_start = 0;
        buffers_received = 0;
        N_count = 50
        for counter in range(N_count):            
            #if msvcrt.kbhit() and msvcrt.getch().decode() == chr(27):
            #    sys.exit(0)
            
            [success, ans_bytes] = lockin.listen_to_lockin_amp()
            #print(success)
            if success:
                buffers_received += 1
                
                if time_start == 0:
                    time_start = Time.time()
                
                N_samples = int(len(ans_bytes) / struct.calcsize('LHH'))
                samples_received = np.append(samples_received, N_samples)
                
                e_byte_time  = N_samples * struct.calcsize('L');
                e_byte_ref_X = e_byte_time + N_samples * struct.calcsize('H')
                e_byte_sig_I = e_byte_ref_X + N_samples * struct.calcsize('H')
                bytes_time  = ans_bytes[0            : e_byte_time]
                bytes_ref_X = ans_bytes[e_byte_time  : e_byte_ref_X]
                bytes_sig_I = ans_bytes[e_byte_ref_X : e_byte_sig_I]
                try:
                    time  = struct.unpack('<' + 'L'*N_samples, bytes_time)
                    ref_X = struct.unpack('<' + 'H'*N_samples, bytes_ref_X)
                    sig_I = struct.unpack('<' + 'H'*N_samples, bytes_sig_I)
                except Exception as err:
                    lockin.close()
                    f_log.close()
                    raise(err)
                
                #print("%3d: %d" % (counter, N_samples))
                f_log.write("samples received: %i\n" % N_samples)
                for i in range(N_samples):
                    f_log.write("%i\t%i\t%i\n" % (time[i], ref_X[i], sig_I[i]))
                
                ref_X = np.array(ref_X)
                ref_X = np.cos(2*np.pi*ref_X/12288)
                
                full_time  = np.append(full_time , time)
                full_ref_X = np.append(full_ref_X, ref_X)
                full_sig_I = np.append(full_sig_I, sig_I)
        
        time_end = Time.time()
        f_log.write("draw\n")
        
        full_time = full_time - full_time[0]
        
        dt = np.diff(full_time)
        Fs = 1/np.mean(dt)*1e6
        str_info1 = ("Fs = %.2f Hz    dt_min = %d us    dt_max = %d us" % 
                    (Fs, np.min(dt), np.max(dt)))
        str_info2 = ("N_buf = %d     %.2f buf/s" % 
                    (buffers_received,
                     buffers_received/(time_end - time_start)))
        print(str_info1 + "    " + str_info2)
        
        if fDrawPlot:
            ax.cla()
            #ax.plot(full_time/1e3, full_ref_X/(2**10 - 1)*3.3, 'x-k')
            ax.plot(full_time/1e3, full_ref_X, 'x-k')
            ax.plot(full_time/1e3, full_sig_I/(2**12 - 1)*3.3 - 2, 'x-r')
            ax.set(xlabel='time (ms)', ylabel='y',
                   title=(str_info1 + '\n' + str_info2))
            ax.grid()            
            ax.set(xlim=(0, 80))
            
            #I = full_ref_X / (2**10)*3.3 * full_sig_I / (2**12)*3.3
            #ax.plot(full_time, I, '.-m')    
            
            fig.canvas.draw()
            plt.pause(0.1)
    
    # Turn lock-in amp off
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