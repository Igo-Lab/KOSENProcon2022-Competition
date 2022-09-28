import ctypes
from ssl import cert_time_to_seconds
import numpy as np
from numpy import ctypeslib


_INT16_PP = ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
_INT16_P = ctypes.POINTER(ctypes.c_int16)
_INT32_PP = ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
_INT32_P = ctypes.POINTER(ctypes.c_int32)

# dll読み込み
resolver_dll = np.ctypeslib.load_library("libresolver.so", ".")

# 関数実体定義
gpu_resolver = resolver_dll.resolver
gpu_resolver.restype = None
gpu_resolver.argtypes = (_INT16_P, _INT16_PP, ctypes.c_int32, _INT32_P, _INT32_PP)

# 引数準備
problem = np.random.rand(5).astype(dtype=np.int16)
src = np.random.rand(5, 5).astype(dtype=np.int16)
src_pp = (
    src.__array_interface__["data"][0] + np.arange(src.shape[0]) * src.strides[0]
).astype(np.uintp)
problem_len = 5
src_length = np.full(5, 5, dtype=np.int32)
result = np.zeros((5, 2), dtype=np.int32)
result_pp = (
    result.__array_interface__["data"][0]
    + np.arange(result.shape[0]) * result.strides[0]
).astype(np.uintp)

# 配列内容表示
print(result)

# 呼び出し
print("python call")

gpu_resolver(
    problem.ctypes.data_as(_INT16_P),
    src_pp,
    problem_len,
    src_length.ctypes.data_as(_INT32_P),
    result_pp,
)

print("python end")

print(result)
