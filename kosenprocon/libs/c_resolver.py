import ctypes

import numpy as np
import numpy.typing as npt
from loguru import logger
from numpy import ctypeslib

from . import constant

# TODO:uintとintに間違いがないかC++側と見比べる
_INT16_PP = ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
_INT16_P = ctypes.POINTER(ctypes.c_int16)
_UINT32_PP = ctypeslib.ndpointer(dtype=np.uintp, ndim=1, flags="C")
_UINT32_P = ctypes.POINTER(ctypes.c_uint32)
_INT32_P = ctypes.POINTER(ctypes.c_int32)
_BOOL_P = ctypes.POINTER(ctypes.c_bool)

c_resolver_dll = np.ctypeslib.load_library("libresolver.so", constant.DLL_DIR)

# DLLの関数
c_memcpy_src2gpu = c_resolver_dll.memcpy_src2gpu
c_memcpy_src2gpu.restype = None
c_memcpy_src2gpu.argtypes = (_INT16_PP, _INT32_P)

c_resolver = c_resolver_dll.resolver
c_resolver.restype = None
c_resolver.argtypes = (_INT16_P, ctypes.c_int32, _BOOL_P, _UINT32_PP)


def memcpy_src2gpu(srcs: npt.NDArray[np.int16], src_lens: npt.NDArray[np.int32]):
    srcs_pp = (
        srcs.__array_interface__["data"][0] + np.arange(srcs.shape[0]) * srcs.strides[0]
    ).astype(np.uintp)

    c_memcpy_src2gpu(srcs_pp, src_lens.ctypes.data_as(_INT32_P))
    logger.debug("Copying Src data to GPU has been done.")


def resolver(
    chunk: npt.NDArray[np.int16],
    chunk_len: int,
    mask: npt.NDArray[np.bool_],
    result: npt.NDArray[np.uint32],
):
    result_pp = (
        result.__array_interface__["data"][0]
        + np.arange(result.shape[0]) * result.strides[0]
    ).astype(np.uintp)

    c_resolver(
        chunk.ctypes.data_as(_INT16_P),
        chunk_len,
        mask.ctypes.data_as(_BOOL_P),
        result_pp,
    )
    logger.debug("Calling C++ Resolver has been done.")
