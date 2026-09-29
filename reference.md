# Reference — ml-from-scratch

_Last refreshed: 2026-09-29_

## Purpose
Educational ML library in three layers: a C++ backend for linear algebra and distance metrics, a
Python `ctypes` bridge, and a model layer (Linear Regression, Logistic Regression, KNN). All heavy
arithmetic is *intended* to be offloaded to compiled C++ — see Current State for how that is going.
No external Python packages.

## Stack
- C++ (GCC, `-O3 -Wall -fPIC -shared` → `cpp/libvector.so`)
- Python 3 stdlib only (tested on 3.14), `ctypes` bridge
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
| `cpp/libvector.so` | compiled library — **committed to git**, arch-specific; currently in sync with the source |
| `ml_math/CPP_vector.py` | ctypes wrappers; no `__init__.py`, works as a namespace package |
| `ml_math/utils.py` | sigmoid, min-max scaler, standard scaler (nothing currently calls the scalers) |
| `models/base.py` | `BaseModel` ABC — `fit()` / `predict()` |
| `models/linear_regression.py` | gradient-descent MSE regression |
| `models/logistic_regression.py` | binary classifier, `predict_proba` + `predict` |
| `models/knn.py` | majority-vote KNN over C++ Euclidean distance |
| `Makefile` | builds the `.so`; `test` target references five files that do not exist |
| `docs/kaggle-learning-notes/` | imported EDA/modelling reading list; stub, paths inside it don't resolve |

## Current State (verified 2026-09-29)

**The package does not import.** `models/linear_regression.py:10` runs
`logging.basicConfig(filename="test/LinearReg_debug.log")` at import time; `test/` does not exist,
so `import models` raises `FileNotFoundError`. `models/__init__.py` pulls in that module, so every
model is affected. Unchanged since the 2026-09-18 review flagged it. Six-line fix, blocks everything.

**The C++ backend is currently slower than pure Python.** Every `CPP_vector` wrapper re-marshals its
inputs on each call, and that dominates the arithmetic. Measured at 2000×20, 20 reps:
`matrix_vector_multiply` 0.260 s vs 0.129 s for the pure-Python equivalent, and 0.001 s if the
matrix is marshalled once up front. In the KNN path, `euclidean_distance` per training row takes
0.542 s against 0.028 s for stdlib `math.dist` — 19x slower. `README.md:41` claims the opposite.
The C++ itself is correct and fast; the boundary is crossed at the wrong granularity.

**There are no tests in the repository.** `.gitignore:27` ignores `test/`, `git ls-files` lists no
test files, and the directory is absent from disk. `make test` invokes five scripts that do not exist.

**Confirmed correctness bugs** beyond the import failure:
- `matrix_vector_multiply` does not check the vector length against the matrix column count; C++
  reads past the end of the buffer and returns nondeterministic garbage. Reachable from `predict()`
  on either regression model.
- `dot` / `add` / `euclidean_distance` size their buffers from the first argument only and silently
  zero-pad a shorter second argument, so KNN answers dimension-mismatched queries without error.
- `ml_math/utils.py:8` `sigmoid` raises `OverflowError` below about `x = -710`, reachable from
  `LogisticRegression.fit` with unscaled features.
- Both scalers (`utils.py:10`, `:48`) recompute statistics from whatever matrix they are handed and
  discard them, so training scaling cannot be applied to held-out data.

Full detail, with reproductions and measurements, in `todo.md`.

## Activity
Last substantive code commit was `a75feb6` (2026-03-28) — six months ago — which added KNN, logistic
regression, the scalers, and the `BaseModel` interface. Everything since is scaffold, automated
review notes, and the `docs/kaggle-learning-notes/` import (`b625f98`, 2026-09-21). The architecture
is sound and the C++ layer is correct; the Python boundary around it is what needs work, and nobody
has worked on it.
