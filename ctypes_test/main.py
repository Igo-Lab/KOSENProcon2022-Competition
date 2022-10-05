import ctypes as ct
import numpy as np

myso = ct.cdll.LoadLibrary('./main.so')

ckass = myso.kass
#ここから先、ブックマークに入れてるブログを見て進め。みろ、ちゃんと全部
pipiarray = np.array([1,2,3],dtype=ct.c_int)
ckass.restype = ct.c_int #戻り値の型指定
ckass.argtypes = (ct.POINTER(ct.c_int),ct.c_int) #引数の型指定
print(pipiarray)
ckass(pipiarray.ctypes.data_as(ct.POINTER(ct.c_int),2))