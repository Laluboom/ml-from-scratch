import ctypes
import os

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'libvector.so'))

lib.dot.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.c_int]
lib.dot.restype = ctypes.c_double

lib.add.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]

lib.S_multiply.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.c_int]

def dot(a,b):
    size = len(a)
    a_arr = (ctypes.c_double * size)(*a)
    b_arr = (ctypes.c_double * size)(*b)
    result = lib.dot(a_arr,b_arr, size)                  
    return result

def add(a,b):
    size = len(a)
    a_arr = (ctypes.c_double * size)(*a)
    b_arr = (ctypes.c_double * size)(*b)
    result = (ctypes.c_double * size)()
    lib.add(a_arr, b_arr, result, size)
    return list(result)

def S_multiply(scalar, a):
    size = len(a)
    a_arr = (ctypes.c_double * size)(*a)
    result = (ctypes.c_double * size)()
    lib.S_multiply(scalar, a_arr, result, size)
    return list(result)                  
                  
