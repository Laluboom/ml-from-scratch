# Reference — ml-from-scratch

_Last refreshed: 2026-05-30_

## Purpose
Educational ML library built in three layers: a high-performance C++ backend for linear algebra and distance metrics, a Python ctypes bridge, and a model layer implementing Linear Regression, Logistic Regression, and K-Nearest Neighbours. All heavy computation is offloaded to compiled C++.

## Stack
- C++ (GCC, compiled to shared library `libvector.so`)
- Python 3, `ctypes` (bridge to C++ backend)
- Make (build + test automation)

## Entry Points
```bash
# Compile C++ shared library (only needed once, or after editing cpp/)
make

# Run full test suite (compiles first if needed)
make test

# Remove compiled library
make clean
```

## Key Files
| File | Role |
|------|------|
| `cpp/vector_math.cpp` | C++ source — dot product, matrix-vector multiply, Euclidean distance, vector add, scalar multiply |
| `cpp/libvector.so` | Compiled shared library — loaded at runtime by the Python bridge |
| `ml_math/CPP_vector.py` | ctypes bridge — loads `libvector.so`, exposes C++ functions to Python |
| `ml_math/utils.py` | Python utilities — sigmoid, min-max scaling, standard scaling |
| `models/base.py` | `BaseModel` abstract interface — `fit()` and `predict()` contract |
| `models/linear_regression.py` | Gradient descent regression using C++ dot product + mat-vec multiply |
| `models/logistic_regression.py` | Binary classifier using C++ ops + Python sigmoid |
| `models/knn.py` | K-Nearest Neighbours using C++ Euclidean distance |
| `test/` | 6 test scripts — math vectors, matrix math, vectorisation, linear reg, logistic reg, KNN |
| `Makefile` | Builds `.so`, runs all 6 test scripts sequentially |

## Run Status (2026-05-30)
`make` reported "Nothing to be done" — `libvector.so` was already compiled and up-to-date.

`make test` ran all 5 test scripts and passed:
- `test_math_vectors.py` — dot product, addition, scalar multiply: correct
- `test_matrix_math.py` — matrix-vector multiply `[[1,2],[3,4],[5,6]] · [1,1]` → `[3,7,11]`: correct
- `test_linear_reg.py` — fit 0.12s, weights `[2.004]`, bias `0.986`, predictions within 0.01 of true values
- `test_logistic_reg.py` — binary predictions `[0, 1]` correct
- `test_knn.py` — predictions `[0, 1]` correct

**Stale `requirements.txt`:** lists `yfinance` and `pandas`, but neither is imported anywhere in the source. The project has no external Python dependencies in practice.
