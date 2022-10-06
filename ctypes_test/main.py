import ctypes as ct
import numpy as np

myso = ct.cdll.LoadLibrary('./main.so')

ckass = myso.kass
pipiarray = np.array([1,2,3],dtype=ct.c_int)

ckass.restype = None #戻り値の型指定
ckass.argtypes = (ct.POINTER(ct.c_int),ct.c_int) #引数の型指定

print("加工前: ")
print(pipiarray)
pipiarray_P = pipiarray.ctypes.data_as(ct.POINTER(ct.c_int))
ckass(pipiarray_P,2)

print("加工後: ")
print(pipiarray)