"""Provide functionality to store intermediate results in lookup tables (memoize)"""

import pickle
import functools

class Memoize:
    """To be used as a decorator for functions that are memoized."""
    def __init__(self, fn):
        self.fn = fn
        self.memo = {}
    def __call__(self, *args, **kwds):
        if len(self.memo) > 100000:
            self.memo.clear()
        argstr = pickle.dumps(args, 1)+pickle.dumps(kwds, 1)
        if not argstr in self.memo:
            #print("miss") # DEBUG INFO
            self.memo[argstr] = self.fn(*args, **kwds)
        #else:
            #print("hit") # DEBUG INFO
        #print(len(self.memo))
        return self.memo[argstr]
    
    def __get__(self, obj, objtype):
        '''Support instance methods.'''
        return functools.partial(self.__call__, obj)
