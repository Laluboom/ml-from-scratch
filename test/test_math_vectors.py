import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ml_math import CPP_vector

a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

print("Matrix a = ", a , ". Matrix b = ", b) 
print("Dot Product of a and b is ", CPP_vector.dot(a,b))
print("Addition of a and b is ", CPP_vector.add(a,b))
print("Scalar Multiple of a by 2 is ", CPP_vector.S_multiply(2.0,a))
