# -*- coding: utf-8 -*-

import timeit

setup = '''
import numpy as np
from numba import njit

np.random.seed(0)
a = np.random.randn(2000)


@njit("float64[:](float64[:])", nogil=True, cache=False)
def round_C_style(array_in: np.ndarray) -> np.ndarray:
    """
    round_C_style([0.1 , 1.45, 1.55, -0.1 , -1.45, -1.55])
    Out[]:  array([0.  , 1.  , 2.  , -0.  , -1.  , -2.  ])
    """
    _abs = np.abs(array_in)
    _trunc = np.trunc(_abs)
    _frac_rounded = np.zeros_like(_abs)
    _frac_rounded[(_abs % 1) >= 0.5] = 1

    return np.sign(array_in) * (_trunc + _frac_rounded)
'''

print(timeit.timeit("round_C_style(a)", setup=setup, number=10000))
