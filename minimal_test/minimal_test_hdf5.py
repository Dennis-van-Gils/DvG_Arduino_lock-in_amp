#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py as h5
import numpy as np

time  = np.arange(0, 8e6, 100, dtype=np.int64)
ref_X = np.sin(2*np.pi*100*time/1e6)
sig_I = np.sin(2*np.pi*100*time/1e6 - 20/180*np.pi)
sig_I = sig_I + 0.1 * np.random.rand(len(sig_I))

m = (np.array([time, ref_X, sig_I])).transpose()

if 0:
    import matplotlib.pyplot as plt
    plt.plot(time, ref_X)
    plt.plot(time, sig_I, 'r')
    plt.xlim(0, 2e4)
    plt.show()

if 1:
    filename = "minimal_test_hdf5_plain.txt"
    np.savetxt(filename, m, fmt='%i\t%.5f\t%.5f')

if 0:
    from pandas import HDFStore, DataFrame
    filename = "minimal_test_hdf5_hdfstore.h5"
    hdf = HDFStore(filename)
    
    df = DataFrame(m, columns=('time','ref_X','sig_I'), dtype=np.float32)
    hdf.put('d1', df, format='table', data_columns=True)
    hdf.close()

if 1:
    filename = "minimal_test_hdf5_h5py.h5"
    with h5.File(filename, "w") as f:
        #g = f.create_group('Base_Group', track_order=True)
        
        f.attrs['Date'] = '24-04-2019'
        f.attrs['ref_freq'] = '100 Hz'
        dset_time = f.create_dataset("time",
                                     np.empty(1),
                                     chunks=(500,),
                                     maxshape=(None,),
                                     #data=time,
                                     dtype='i8',
                                     compression="gzip")
        dset_ref_X = f.create_dataset("ref_X",
                                      np.empty(1),
                                      chunks=(500,),
                                      maxshape=(None,),
                                      #data=ref_X,
                                      dtype='f4',
                                      compression="gzip")
        dset_sig_I = f.create_dataset("sig_I",
                                      np.empty(1),
                                      chunks=(500,),
                                      maxshape=(None,),
                                      #data=sig_I,
                                      dtype='f4',
                                      compression="gzip")
        
        # Simulate incoming buffers of each buffer_size samples long
        i = 0
        buffer_size = 500
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