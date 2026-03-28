import ctypes
import os

lib = ctypes.CDLL(os.path.join(os.path.dirname(__file__), '..', 'cpp', 'libvector.so'))

lib.dot.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.c_int]
lib.dot.restype = ctypes.c_double

lib.add.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]

lib.S_multiply.argtypes = [ctypes.c_double, ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),ctypes.c_int]

lib.matrix_vector_multiply.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int, ctypes.c_int]

lib.euclidean_distance.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double), ctypes.c_int]
lib.euclidean_distance.restype = ctypes.c_double

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

def matrix_vector_multiply(matrix, vec):
    """
    multiplies a matrix (2D list) with a vector (1D list)
    """
    rows = len(matrix)
    cols = len(matrix[0]) if rows > 0 else 0
    
    # Flatten the 2D matrix for C++
    flattened_matrix = [item for row in matrix for item in row]
    
    matrix_arr = (ctypes.c_double * len(flattened_matrix))(*flattened_matrix)
    vec_arr = (ctypes.c_double * len(vec))(*vec)
    result = (ctypes.c_double * rows)()
    
    lib.matrix_vector_multiply(matrix_arr, vec_arr, result, rows, cols)
    return list(result)

def euclidean_distance(a, b):
    """
    calculates the euclidean distance between two vectors
    """
    size = len(a)
    a_arr = (ctypes.c_double * size)(*a)
    b_arr = (ctypes.c_double * size)(*b)
    return lib.euclidean_distance(a_arr, b_arr, size)
