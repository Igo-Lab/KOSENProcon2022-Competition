import ctypes

import numpy as np

dp=ctypes.ndpointer(ctypes.c_int16)

lib=np.cdll.LoadLibrary

lib=ctypes.cdll.LoadLibrary("getbox.so")

post=np.array([1,2,3],[4,5,6],dtype=np.c_int16)

post_pp=(post.hairetu["data"][0]+np)

