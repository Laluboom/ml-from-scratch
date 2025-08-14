
#include <vector>
#include <iostream>

extern "C" {
	double dot(const double* a, const double* b, int size) {
		double result = 0.0;
		for (int i = 0; i< size; ++i)
			result += a[i] * b[i];
		return result;
	}
	void add(const double* a, const double* b,double* result, int size) {
		for (int i = 0; i < size; ++i) 
		       result[i] = a[i] + b[i];
	}
	void S_multiply(double scalar, const double* a, double* result, int size) {
	for (int i = 0; i< size; ++i)
		result[i] = scalar * a[i];
	}	
}
