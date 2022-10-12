import ctypes as ct
import numpy as np

myso = ct.cdll.LoadLibrary('./main.so')
pipiarray_PP = np.ctypeslib.ndpointer(dtype=np.uintp,ndim=1,flags="C")

ckass = myso.kass
pipiarray = np.zeros((3, 2), dtype=np.int32)

ckass.restype = None #戻り値の型指定
ckass.argtypes = (pipiarray_PP,ct.c_int) #引数の型指定

pipiarray_P = (pipiarray.__array_interface__["data"][0] + np.arange(pipiarray.shape[0]) * pipiarray.strides[0]).astype(np.uintp)

print("加工前: ")
print(pipiarray)

ckass(pipiarray_P,2)

print("加工後: ")
print(pipiarray)