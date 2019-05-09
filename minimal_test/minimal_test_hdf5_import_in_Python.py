# -*- coding: utf-8 -*-

import h5py as h5
import numpy as np

filename = "minimal_test_hdf5_h5py.h5"
#with h5.File(filename, "r") as f:
f = h5.File(filename, "r")

f.visit(print)

for key in f.keys():
    print(key)
    
    for subkey in f[key]:
        print("  " + subkey)
    
    