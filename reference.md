# Reference — ml-from-scratch

_Last refreshed: 2026-09-18_

## Purpose
Educational ML library in three layers: a C++ backend for linear algebra and distance metrics, a
Python `ctypes` bridge, and a model layer (Linear Regression, Logistic Regression, KNN). All heavy
arithmetic is offloaded to compiled C++. No external Python packages.

## Stack
- C++ (GCC, `-O3 -Wall -fPIC -shared` → `cpp/libvector.so`)
- Python 3 stdlib only, `ctypes` bridge
- Make for build + test

## Entry Points
```bash
make          # compile cpp/libvector.so
make test     # BROKEN — see Current State
make clean    # remove the .so
```

## Key Files
| File | Role |
|------|------|
| `cpp/vector_math.cpp` | dot, add, scalar multiply, matrix-vector multiply, Euclidean distance |
| `cpp/libvector.so` | compiled library — **committed to git**, arch-specific, drifts from source |
| `ml_math/CPP_vector.py` | ctypes wrappers; no `__init__.py`, works as a namespace package |
| `ml_math/utils.py` | sigmoid, min-max scaler, standard scaler (nothing currently calls the scalers) |
| `models/base.py` | `BaseModel` ABC — `fit()` / `predict()` |
| `models/linear_regression.py` | gradient-descent MSE regression |
| `models/logistic_regression.py` | binary classifier, `predict_proba` + `predict` |
| `models/knn.py` | majority-vote KNN over C++ Euclidean distance |
| `Makefile` | builds the `.so`; `test` target references files that do not exist |

## Current State (verified 2026-09-18)

**The package does not import.** `models/linear_regression.py:10` runs
`logging.basicConfig(filename="test/LinearReg_debug.log")` at import time; `test/` does not exist,
so `import models` raises `FileNotFoundError`. `models/__init__.py` pulls in that module, so every
model is affected. This is a one-import-line fix and it blocks everything else.

**There are no tests in the repository.** `.gitignore:27` ignores `test/`, `git ls-files` lists no
test files, and the directory is absent from disk. `make test` invokes five scripts that do not
exist. The previous version of this file reported a passing `make test` run with per-test results —
that was not reproducible against the tracked tree and has been removed.

**Two confirmed correctness bugs** beyond the import failure:
- `CPP_vector.matrix_vector_multiply` does not check the vector length against the matrix column
  count; C++ reads past the end of the buffer and returns nondeterministic garbage. Reachable from
  `predict()` on either regression model.
- `dot` / `add` / `euclidean_distance` size their buffers from the first argument only and silently
  zero-pad a shorter second argument, so KNN answers dimension-mismatched queries without error.

`ml_math/utils.py:8` `sigmoid` also raises `OverflowError` below about `x = -710`.

Full detail, with reproductions, in `todo.md`.

## Activity
Last substantive code commit was `a75feb6` (2026-03-28), which added KNN, logistic regression, the
scalers, and the `BaseModel` interface. Everything since has been scaffold and notes churn. The
architecture is sound and the C++ layer is correct; the Python boundary around it is what needs work.
