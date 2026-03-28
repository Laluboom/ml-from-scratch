
#include <vector>
#include <iostream>
#include <cmath>

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
	void matrix_vector_multiply(const double* matrix, const double* vec, double* result, int rows, int cols) {
		for (int i = 0; i < rows; ++i) {
			result[i] = 0.0;
			for (int j = 0; j < cols; ++j) {
				result[i] += matrix[i * cols + j] * vec[j];
			}
		}
	}
	double euclidean_distance(const double* a, const double* b, int size) {
		double sum = 0.0;
		for (int i = 0; i < size; ++i) {
			double diff = a[i] - b[i];
			sum += diff * diff;
		}
		return std::sqrt(sum);
	}
}
