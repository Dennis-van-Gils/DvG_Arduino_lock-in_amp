#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Arduino lock-in amplifier
Minimal running example for trouble-shooting library
"""
__author__ = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__ = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__ = "03-02-2022"
__version__ = "1.0.0"
# pylint: disable=invalid-name

import os
import sys
from collections import deque
import time as Time

import psutil
import numpy as np
import matplotlib.pyplot as plt

from Alia_protocol_serial import Alia, Waveform

fn_log = "log.txt"
fDrawPlot = True
fVerbose = True

if __name__ == "__main__":
    # Set priority of this process to maximum in the operating system
    print("PID: %s" % os.getpid())
    try:
        proc = psutil.Process(os.getpid())
        if os.name == "nt":
            proc.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows
        else:
            proc.nice(-20)  # Other
    except:  # pylint: disable=bare-except
        print("Warning: Could not set process to high priority.")

    alia = Alia()
    alia.auto_connect()

    if not alia.is_alive:
        print("\nCheck connection and try resetting the Arduino.")
        print("Exiting...\n")
        sys.exit(0)

    alia.begin(
        freq=250,
        V_offset=1.5,
        V_ampl=1,
        waveform=Waveform.Sine,
    )

    if fDrawPlot:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(0, 0)
        fig.show()
        fig.canvas.draw()
    f_log = open(fn_log, "w")

    N_SETS = 3
    N_REPS = 41

    c = alia.config
    N_deque = c.BLOCK_SIZE * N_REPS
    deque_time = deque(maxlen=N_deque)
    deque_ref_X = deque(maxlen=N_deque)
    deque_ref_Y = deque(maxlen=N_deque)
    deque_sig_I = deque(maxlen=N_deque)
    samples_received = np.array([], dtype=int)

    alia.turn_on(reset_timer=True)
    for i_set in range(N_SETS):
        if i_set == 1:
            alia.set_ref(freq=200)
            print("")
        if i_set == 2:
            alia.set_ref(waveform=Waveform.Triangle)
            print("")

        deque_time.clear()
        deque_ref_X.clear()
        deque_ref_Y.clear()
        deque_sig_I.clear()

        tick = 0
        blocks_received = 0
        for i_rep in range(N_REPS):
            (
                success,
                counter,
                time,
                ref_X,
                ref_Y,
                sig_I,
            ) = alia.listen_to_lockin_amp()

            if success:
                blocks_received += 1

                if tick == 0:
                    tick = Time.perf_counter()

                N_samples = len(time)
                if fVerbose:
                    print("%3d: %d" % (i_rep, N_samples))

                # Note: `ref_X` [non-dim] is transformed to `ref_X*` [V]
                # Note: `ref_Y` [non-dim] is transformed to `ref_Y*` [V]
                ref_X = np.multiply(ref_X, c.ref_V_ampl_RMS) + c.ref_V_offset
                ref_Y = np.multiply(ref_Y, c.ref_V_ampl_RMS) + c.ref_V_offset

                f_log.write("samples received: %i\n" % N_samples)
                for i in range(N_samples):
                    f_log.write(
                        "%i\t%.4f\t%.4f\t%.3f\n"
                        % (time[i], ref_X[i], ref_Y[i], sig_I[i])
                    )

                deque_time.extend(time)
                deque_ref_X.extend(ref_X)
                deque_ref_Y.extend(ref_Y)
                deque_sig_I.extend(sig_I)
                samples_received = np.append(samples_received, N_samples)

        f_log.write("draw\n")

        if fVerbose or fDrawPlot:
            np_time = np.array(deque_time)
            np_time = np_time - np_time[0]
            dt = np.diff(np_time)

            Fs = 1 / np.mean(dt) * 1e6
            str_info1 = "Fs = %.2f Hz    dt_min = %d us    dt_max = %d us" % (
                Fs,
                np.min(dt),
                np.max(dt),
            )
            str_info2 = "blocks received = %d     %.2f blocks/s" % (
                blocks_received,
                blocks_received / (Time.perf_counter() - tick),
            )
            print("\n%s    %s\n" % (str_info1, str_info2))

        if fDrawPlot:
            # alia.turn_off()

            ax.cla()
            ax.plot(np_time / 1e3, deque_ref_X, ".-k")
            ax.plot(np_time / 1e3, deque_ref_Y, ".-y")
            ax.plot(np_time / 1e3, deque_sig_I, ".-r")
            ax.set(
                xlabel="time (ms)",
                ylabel="voltage (V)",
                title=(str_info1 + "\n" + str_info2),
            )
            ax.grid()
            ax.set(xlim=(0, 18.8))

            fig.canvas.draw()
            plt.pause(0.1)

            # alia.turn_on()

    # Finish test
    alia.turn_off()
    alia.close()
    f_log.close()

    print(
        "Samples received per block: [min, max] = [%d, %d]"
        % (np.min(samples_received), np.max(samples_received))
    )

    fig.canvas.draw()

    with open(fn_log, "r") as file:
        filedata = file.read()
    filedata = filedata.replace("draw", "")
    filedata = filedata.replace(
        "samples received: %i" % alia.config.BLOCK_SIZE, ""
    )

    with open(fn_log, "w") as file:
        file.write(filedata)

    a = np.loadtxt(fn_log)
    time = np.array(a[:, 0])
    ref_X = np.array(a[:, 1])
    sig_I = np.array(a[:, 2])
    # time = time - time[0]

    time_diff = np.diff(time)
    print("\ntime_diff:")
    print("  median = %i usec" % np.median(time_diff))
    print("  mean   = %i usec" % np.mean(time_diff))
    print("  min    = %i usec" % np.min(time_diff))
    print("  max    = %i usec" % np.max(time_diff))

    time_ = time[:-1]
    time_gaps = time_[abs(time_diff) > 500]
    time_gap_durations = time_diff[abs(time_diff) > 500]
    print("\nnumber of gaps > 500 usec: %i" % len(time_gaps))
    for i in range(len(time_gaps)):
        print(
            "  gap %i @ t = %.3f msec for %.3f msec"
            % (i + 1, time_gaps[i] / 1e3, time_gap_durations[i] / 1e3)
        )

    plt.show(block=True)
