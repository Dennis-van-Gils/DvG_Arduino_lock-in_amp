#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 14:19:05 2018

@author: Dennis_van_Gils
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
fDrawPlot = False

def delayed_query(msg_str: str, ser: serial.Serial):
    #ser.flushInput()
    #ser.flushOutput()
    #ser.flush()
    
    # OFF
    # ---------
    ser.write("off\n".encode())
    #ser.flush()
    #ser.flushOutput()
    #ser.flushInput()

    # OFF?
    # ---------    
    ser.read_until("ok\n".encode())
    ser.flushInput()
    
    # IDN?
    # ---------
    #ser.flush()
    #ser.flushInput()
    #ser.flushOutput()
    
    Time.sleep(0.2)
    ser.flushInput()
    print(ser.inWaiting())
    
    ser.write(msg_str.encode())
    
    # IDN reply
    # ---------
    ans_bytes = ser.read_until("ok\r\n".encode())    
    print(ans_bytes)
    ans_str = ans_bytes.decode('utf8').strip()
    
    # ON
    # ---------
    ser.write("on\n".encode())
    ser.flushOutput()
    
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
    
    #ser.write("on\n".encode())
    
    if fDrawPlot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(0, 0)
        fig.show()
        fig.canvas.draw()

    samples_received = np.array([], dtype=int)

    uber_counter = 0
    while uber_counter < 3:
        uber_counter += 1
        full_time    = np.array([], dtype=int)
        full_ref_out = np.array([], dtype=int)
        full_sig_in  = np.array([], dtype=int)
    
        counter = 0
        N_count = 50
        time_start = 0;
        buffers_received = 0;
        while counter < N_count:
            print("%d %s" % (counter, delayed_query("id?\n", ser)))
            counter += 1
            
            #if msvcrt.kbhit() and msvcrt.getch().decode() == chr(27):
            #    sys.exit(0)
            
            ans_bytes = ser.read_until(EOM)
            ans_bytes = ans_bytes[:-5]      # Remove EOM
            
            if (ans_bytes[:5] == SOM):
                buffers_received += 1
                if time_start == 0:
                    time_start = Time.time()
                
                ans_bytes = ans_bytes[5:]      # Remove SOM
                N_samples = int(len(ans_bytes) / struct.calcsize('LHH'))
                s_byte_time    = 0;
                e_byte_time    = N_samples * struct.calcsize('L');
                s_byte_ref_out = e_byte_time;
                e_byte_ref_out = s_byte_ref_out + N_samples * struct.calcsize('H')
                s_byte_sig_in  = e_byte_ref_out;
                e_byte_sig_in  = s_byte_sig_in + N_samples * struct.calcsize('H')
                time_bytes    = ans_bytes[s_byte_time   :e_byte_time]
                ref_out_bytes = ans_bytes[s_byte_ref_out:e_byte_ref_out]
                sig_in_bytes  = ans_bytes[s_byte_sig_in :e_byte_sig_in]
                try:
                    time    = struct.unpack('<' + 'L'*N_samples, time_bytes)
                    ref_out = struct.unpack('<' + 'H'*N_samples, ref_out_bytes)
                    sig_in  = struct.unpack('<' + 'H'*N_samples, sig_in_bytes)
                except Exception as err:
                    ser.close()
                    f_log.close()
                    raise(err)
                
                samples_received = np.append(samples_received, N_samples)
                full_time    = np.append(full_time   , time)
                full_ref_out = np.append(full_ref_out, ref_out)
                full_sig_in  = np.append(full_sig_in , sig_in)
                
                print("%3d: %d" % (counter, N_samples))
                #print("%10i\t%7.4f" % (time[0], ref_out[0]))
                f_log.write("samples received: %i\n" % N_samples)
                for i in range(N_samples):
                    f_log.write("%i\t%i\t%i\n" % 
                                (time[i], ref_out[i], sig_in[i]))
        
        time_end = Time.time()
        f_log.write("draw\n")
        
        full_time = full_time - full_time[0]
        
        dt = np.diff(full_time)
        #print(np.where(dt != 200))
        Fs = 1/np.mean(dt)*1e6
        str_info = ("Fs = %.2f Hz    dt_min = %d us    dt_max = %d us" % 
                   (Fs, np.min(dt), np.max(dt)))
        str_info += ("    N_buf = %d     %.2f buf/s" % 
                     (buffers_received,
                      buffers_received/(time_end - time_start)))
        print(str_info)
        
        if fDrawPlot:
            ax.cla()
            ax.plot(full_time/1e3, full_ref_out/(2**10)*3.3, 'x-k')
            ax.plot(full_time/1e3, full_sig_in/(2**12)*3.3, 'x-r')
            ax.set(xlabel='time (ms)', ylabel='y', title=str_info)
            ax.grid()
            #ax.set(xlim=(0, 80))
            ax.set(xlim=(0, 100))
            
            #I = full_ref_out / (2**10)*3.3 * full_sig_in / (2**12)*3.3
            #ax.plot(full_time, I, '.-m')    
            
            fig.canvas.draw()
            plt.pause(0.1)
    
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