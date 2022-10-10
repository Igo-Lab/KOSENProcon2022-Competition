from ctypes import *
import numpy as np

myso = cdll.LoadLibrary('./main.so')

ckass = myso.kass
pipiarray = np.array([1,2,3],dtype=c_int)

ckass.restype = None #戻り値の型指定
ckass.argtypes = (POINTER(c_int),c_int) #引数の型指定

print("加工前: ")
print(pipiarray)
pipiarray_P = pipiarray.ctypes.data_as(POINTER(c_int))
ckass(pipiarray_P,2)

print("加工後: ")
print(pipiarray)