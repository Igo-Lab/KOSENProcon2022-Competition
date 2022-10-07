import ctypes
import numpy as np
from numpy import ctypeslib


pp=ctypeslib.ndpointer(dtype=ctypes.c_int,ndim=1,flags="C")

dp=ctypes.cdll.LoadLibrary("./test.so")

call=dp.data
call.restype=None
call.argtypes=(pp,ctypes.c_int)

post=np.zeros((2,3),dtype=ctypes.c_int)

post_pp=(post.__array_interface__["data"][0]+np.arange(post.shape[0]) * post.strides[0]).astype(ctypes.c_int)

call(post_pp,1)

print("hai")
