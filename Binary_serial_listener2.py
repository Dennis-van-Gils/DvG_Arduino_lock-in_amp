#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11-11-2018
Dennis_van_Gils
"""

import sys
import serial
import struct
import msvcrt

import os
import psutil

import matplotlib
#matplotlib.use('GTKAgg')
#matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

import numpy as np
import time as Time
from pathlib import Path

import DvG_dev_Arduino__fun_serial as Ard_fun

SOM = bytes([0x00, 0x00, 0x00, 0x00, 0xee]) # Start of message
EOM = bytes([0x00, 0x00, 0x00, 0x00, 0xff]) # End of message
fn_log = "log.txt"
fDrawPlot = True

def delayed_query(msg_str: str, ser: serial.Serial):
    # OFF
    # ---------
    ser.write("\n".encode())
    ser.flushOutput()
   
    # OFF?
    # ---------    
    ans_bytes = ser.read_until("off\n".encode())
    #print("     off bytes %5d: %s" % (len(ans_bytes), ans_bytes))
    print("     off bytes %5d" % len(ans_bytes))
    
    # Query
    # ---------
    ser.write(msg_str.encode())
    
    # Reply
    # ---------
    ans_bytes = ser.read_until('\n'.encode())    
    #print("     id? bytes %5d: %s" % (len(ans_bytes), ans_bytes))
    print("     id? bytes %5d" % len(ans_bytes))
    ans_str = ans_bytes.decode('utf8').strip()
    
    # ON
    # ---------
    ser.write("on\n".encode())
    
    return ans_str

if __name__ == "__main__":
    p = psutil.Process(os.getpid())
    p.nice(psutil.HIGH_PRIORITY_CLASS)
    
    try: ser.close()
    except: pass
    
    """
    ard = Ard_fun.Arduino(baudrate=1.5e6)
    if ard.auto_connect(Path("port_data.txt"), "Lock-in amp: data"):
        ser = ard.ser;
    else:
        sys.exit(0)
    """

    ser = serial.Serial("COM3",
                        baudrate=1.5e6,
                        timeout=4,
                        write_timeout=4)
    #ser = serial.Serial("COM6", baudrate=1e6)
    #ser = serial.Serial("COM4")
    f_log = open(fn_log, 'w')
    
    ser.write("on\n".encode())
    
    if fDrawPlot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(0, 0)
        fig.show()
        fig.canvas.draw()

    samples_received = np.array([], dtype=int)

    uber_counter = 0
    while uber_counter < 20:
        uber_counter += 1
        full_time    = np.array([], dtype=int)
        full_ref_X = np.array([], dtype=int)
        full_sig_I  = np.array([], dtype=int)
    
        time_start = 0;
        buffers_received = 0;
        counter = 0
        N_count = 50
        while counter < N_count:
            counter += 1
            
            #if msvcrt.kbhit() and msvcrt.getch().decode() == chr(27):
            #    sys.exit(0)
            
            ans_bytes = ser.read_until(EOM)
            if (ans_bytes[:5] == SOM):
                ans_bytes = ans_bytes[5:-5] # Remove EOM & SOM
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
                    ser.close()
                    f_log.close()
                    raise(err)
                
                full_time  = np.append(full_time , time)
                full_ref_X = np.append(full_ref_X, ref_X)
                full_sig_I = np.append(full_sig_I, sig_I)
                
                #print("%3d: %d" % (counter, N_samples))
                f_log.write("samples received: %i\n" % N_samples)
                for i in range(N_samples):
                    f_log.write("%i\t%i\t%i\n" % (time[i], ref_X[i], sig_I[i]))
        
        time_end = Time.time()
        f_log.write("draw\n")
        
        full_time = full_time - full_time[0]
        
        dt = np.diff(full_time)
        #print(np.where(dt != 200))
        Fs = 1/np.mean(dt)*1e6
        str_info1 = ("Fs = %.2f Hz    dt_min = %d us    dt_max = %d us" % 
                    (Fs, np.min(dt), np.max(dt)))
        str_info2 = ("N_buf = %d     %.2f buf/s" % 
                    (buffers_received,
                     buffers_received/(time_end - time_start)))
        print(str_info1 + "    " + str_info2)
        
        if fDrawPlot:
            ax.cla()
            ax.plot(full_time/1e3, full_ref_X/(2**10)*3.3, 'x-k')
            ax.plot(full_time/1e3, full_sig_I/(2**12)*3.3, 'x-r')
            ax.set(xlabel='time (ms)', ylabel='y',
                   title=(str_info1 + '\n' + str_info2))
            ax.grid()            
            ax.set(xlim=(0, 100))
            
            #I = full_ref_X / (2**10)*3.3 * full_sig_I / (2**12)*3.3
            #ax.plot(full_time, I, '.-m')    
            
            fig.canvas.draw()
            plt.pause(0.1)
    
    # Turn lock-in amp off
    ser.write("\n".encode())
    
    f_log.close()
    ser.close()
    
    print("\nSamples received per buffer: [min, max] = [%d, %d]" % 
          (np.min(samples_received), np.max(samples_received)))
    
    while (1):
        if fDrawPlot:
            plt.pause(0.5)
        else:
            Time.sleep(0.1)
            
        if msvcrt.kbhit() and msvcrt.getch().decode() == chr(27):
            sys.exit(0)