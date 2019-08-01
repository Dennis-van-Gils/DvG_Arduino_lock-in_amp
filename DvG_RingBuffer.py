"""
Class DvG_RingBuffer provides a ring buffer. If, and only if the ring buffer is
completely full, will it return its array data as a contiguous C-style numpy
array at a single fixed memory location per ring buffer instance. It does so
by unwrapping the discontiguous ring buffer array into a second extra 'unwrap'
buffer that is a private member of the ring buffer class. This is advantegeous
for other accelerated computations by, e.g. numpy / sigpy / numba / pyFFTW,
that benefit from being fed with contiguous arrays at the same memory locations
each time again, such that compiler optimizations and data planning are made
possible.

When the ring buffer is not completely full, it will return its data as a
contiguous C-style numpy array, but at different memory locations.

Commonly, 'collections.deque' is used to act as a ring buffer. The benefits of
a deque is that it is thread safe and fast (enough) for most situations.
However, there is an overhead whenever the deque (a list-like container) needs
to be transformed into a numpy array. Class DvG_RingBuffer will outperform a
collections.deque easily (by a factor of ~ 39x). But beware: You have to
implement your own mutex locking/unlocking when using this ring buffer class
in multi-threaded routines.

NOTE: 
The data array that is returned by a full ring buffer is a pass by reference
of the 'unwrap' buffer! It is not a copy. Hence, changing values in the
returned data array is identical to changing values in the 'unwrap' buffer.

Based on: 
https://pypi.org/project/numpy_ringbuffer/
"""
__author__      = "Dennis van Gils"
__authoremail__ = "vangils.dennis@gmail.com"
__url__         = "https://github.com/Dennis-van-Gils/DvG_Arduino_lock-in_amp"
__date__        = "01-08-2019"
__version__     = "1.0.0"

import numpy as np
from collections import Sequence

class RingBuffer(Sequence):
    def __init__(self, capacity, dtype=float, allow_overwrite=True):
        """
        Create a new ring buffer with the given capacity and element type

        Parameters
        ----------
        capacity: int
            The maximum capacity of the ring buffer
        dtype: data-type, optional
            Desired type of buffer elements. Use a type like (float, 2) to
            produce a buffer with shape (N, 2)
        allow_overwrite: bool
            If false, throw an IndexError when trying to append to an alread
            full buffer
        """
        self._arr = np.empty(capacity, dtype)
        self._left_index = 0
        self._right_index = 0
        self._capacity = capacity
        self._allow_overwrite = allow_overwrite
        
        self._unwrap_buffer = np.empty(capacity, dtype, order='C')
        self._is_unwrap_buffer_dirty = False

    def _unwrap(self):
        """ Copy the data from this buffer into unwrapped form """
        
        return np.concatenate((
            self._arr[self._left_index:min(self._right_index, self._capacity)],
            self._arr[:max(self._right_index - self._capacity, 0)]
        ))
        
    def _unwrap_into_buffer(self):
        """ Copy the data from this buffer into unwrapped form """
        if self._is_unwrap_buffer_dirty:
            #print("was dirty")
            np.concatenate((
                self._arr[self._left_index:min(self._right_index, self._capacity)],
                self._arr[:max(self._right_index - self._capacity, 0)]),
                out=self._unwrap_buffer
            )
            self._is_unwrap_buffer_dirty = False
        else:
            #print("was clean")
            pass

    def _fix_indices(self):
        """
        Enforce our invariant that 0 <= self._left_index < self._capacity
        """
        if self._left_index >= self._capacity:
            self._left_index -= self._capacity
            self._right_index -= self._capacity
        elif self._left_index < 0:
            self._left_index += self._capacity
            self._right_index += self._capacity

    @property
    def is_full(self):
        """ True if there is no more space in the buffer """
        return len(self) == self._capacity

    # numpy compatibility
    def __array__(self):
        #print("__array__")
        if self.is_full:
            self._unwrap_into_buffer()
            return self._unwrap_buffer
        else:
            return self._unwrap()
    	
    @property
    def dtype(self):
        return self._arr.dtype

    @property
    def shape(self):
        return (len(self),) + self._arr.shape[1:]

    def clear(self):
        self._left_index = 0
        self._right_index = 0

    # these mirror methods from deque
    @property
    def maxlen(self):
        return self._capacity
        
    def append(self, value):
        if self.is_full:
            if not self._allow_overwrite:
                raise IndexError('append to a full RingBuffer with overwrite disabled')
            elif not len(self):
                return
            else:
                self._left_index += 1

        self._is_unwrap_buffer_dirty = True
        self._arr[self._right_index % self._capacity] = value
        self._right_index += 1
        self._fix_indices()

    def extend(self, values):
        lv = len(values)
        if len(self) + lv > self._capacity:
            if not self._allow_overwrite:
                raise IndexError('extend a RingBuffer such that it would overflow, with overwrite disabled')
            elif not len(self):
                return
            
        self._is_unwrap_buffer_dirty = True
        if lv >= self._capacity:
            # wipe the entire array! - this may not be threadsafe
            self._arr[...] = values[-self._capacity:]
            self._right_index = self._capacity
            self._left_index = 0
            return

        ri = self._right_index % self._capacity
        sl1 = np.s_[ri:min(ri + lv, self._capacity)]
        sl2 = np.s_[:max(ri + lv - self._capacity, 0)]
        self._arr[sl1] = values[:sl1.stop - sl1.start]
        self._arr[sl2] = values[sl1.stop - sl1.start:]
        self._right_index += lv

        self._left_index = max(self._left_index, self._right_index - self._capacity)
        self._fix_indices()

   
    # implement Sequence methods
    def __len__(self):
        return self._right_index - self._left_index

    def __getitem__(self, item):
        # handle simple (b[1]) and basic (b[np.array([1, 2, 3])]) fancy indexing specially
        #print("__get_item__")
        if not isinstance(item, tuple):
            item_arr = np.asarray(item)
            if issubclass(item_arr.dtype.type, np.integer):
                #print("  __int_subclass__")
                item_arr = (item_arr + self._left_index) % self._capacity
                return self._arr[item_arr]

        #print("  __something_else__")
        if self.is_full:
            self._unwrap_into_buffer()
            return self._unwrap_buffer[item]
        else:
            return self._unwrap()[item]

    def __iter__(self):
        #print("__iter__")
        if self.is_full:
            self._unwrap_into_buffer()
            return iter(self._unwrap_buffer)
        else:
            return iter(self._unwrap())

    # Everything else
    def __repr__(self):
        return '<RingBuffer of {!r}>'.format(np.asarray(self))
