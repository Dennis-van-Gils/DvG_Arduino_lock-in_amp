#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import numpy as np
from scipy.signal import firwin, freqz

def attr_var_unit(g: h5._hl.group.Group,
                  str_var: str,
                  str_unit: str,
                  value):
    g.attrs[str_var]            = value
    g.attrs[str_var + '__unit'] = np.string_(str_unit)

buffer_size = 500
ISR_clock = 200         # [us]
Fs = 1/ISR_clock*1e6    # [Hz]
ref_freq = 110.0        # [Hz]

time  = np.arange(0, 8e6, ISR_clock, dtype=np.int64)
ref_X = np.sin(2*np.pi*ref_freq*time/1e6)
sig_I = np.sin(2*np.pi*ref_freq*time/1e6 - 20/180*np.pi)
sig_I = sig_I + 0.1 * np.random.rand(len(sig_I))

m = (np.array([time, ref_X, sig_I])).transpose()

numtaps   = 20001
cutoff    = [0.9, 49, 51, 99, 101, 149, 151]
window    = 'blackmanharris'
pass_zero = False
firf = firwin(20001, cutoff, window=window, pass_zero=pass_zero, fs=Fs)

if 1:
    filename = "minimal_test_hdf5_plain.txt"
    np.savetxt(filename, m, fmt='%i\t%.5f\t%.5f')

if 1:
    filename = "minimal_test_hdf5_h5py.h5"
    with h5.File(filename, "w", track_order=True) as f:
        grp_0 = f.create_group('GRP_0', track_order=True)
        
        m = grp_0.create_group('settings')
        m.attrs['dd-MM-yyyy'] = np.string_('24-04-2019')
        m.attrs['HH:mm:ss']   = np.string_('11:52:03')
        attr_var_unit(m, 'MCU_sample_rate', 'Hz'     , np.float64(Fs))
        attr_var_unit(m, 'MCU_buffer_size', 'samples', np.uint32 (buffer_size))
        attr_var_unit(m, 'ref_freq'       , 'Hz'     , np.float64(ref_freq))
        attr_var_unit(m, 'ref_V_offset'   , 'V'      , np.float64(1.700))
        attr_var_unit(m, 'ref_V_ampl'     , 'V'      , np.float64(1.414))
        
        n = m.create_group('filter_design_1')
        n.attrs['numtaps'] = np.uint32(numtaps)
        attr_var_unit(n, 'cutoff', 'Hz', np.float64(cutoff))
        n.attrs['window'] = np.string_(window)
        n.attrs['pass_zero'] = np.bool(pass_zero)
        attr_var_unit(n, 'T_settle', 's', np.float64(2.0))
        
        p = {'shape': (1,),
             'chunks': (500,),
             'maxshape': (None,),
             'compression': "gzip"}
        dset_time  = grp_0.create_dataset("time" , dtype='i8', **p)
        dset_ref_X = grp_0.create_dataset("ref_X", dtype='f4', **p)
        dset_sig_I = grp_0.create_dataset("sig_I", dtype='f4', **p)
        
        # Simulate incoming buffers of each buffer_size samples long
        i = 0
        while (i + 1)*buffer_size <= len(time):
            idx_start = i * buffer_size
            idx_end   = (i + 1) * buffer_size
            
            new_len = (i + 1) * buffer_size
            dset_time .resize(new_len, axis=0)
            dset_ref_X.resize(new_len, axis=0)
            dset_sig_I.resize(new_len, axis=0)
            
            dset_time [-buffer_size:] = time[idx_start:idx_end]
            dset_ref_X[-buffer_size:] = ref_X[idx_start:idx_end]
            dset_sig_I[-buffer_size:] = sig_I[idx_start:idx_end]
            i += 1
        
        f.visit(lambda key: print(key))
        
        for k in f.attrs.keys():
            print('{} => {}'.format(k, f.attrs[k]))
            
        f.close()