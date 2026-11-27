# Reference — ml-from-scratch

_Last refreshed: 2026-11-27_

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
make          # compile cpp/libvector.so (note: rewrites a git-tracked binary)
make test     # BROKEN — runs five files that do not exist
make clean    # remove the .so
```
There is no packaging. Imports resolve only when the repo root is on `sys.path` — i.e. a REPL or
`python3 -c` **from the repo root**. Running a script from anywhere else gives
`ModuleNotFoundError: No module named 'models'`.

## Key Files
| File | Role |
|------|------|
| `cpp/vector_math.cpp` | dot, add, scalar multiply, matrix-vector multiply, Euclidean distance |
| `cpp/libvector.so` | compiled library — **committed to git**, arch-specific; currently in sync |
| `ml_math/CPP_vector.py` | ctypes wrappers; no length validation; no `__init__.py` |
| `ml_math/utils.py` | sigmoid, min-max scaler, standard scaler. **No loss or metric functions** |
| `models/base.py` | `BaseModel` ABC — `fit()` / `predict()`, inits `weights=[]`, `bias=0.0` |
| `models/linear_regression.py` | gradient-descent MSE regression (MSE itself is never computed) |
| `models/logistic_regression.py` | binary classifier, `predict_proba` + `predict` |
| `models/knn.py` | majority-vote KNN over C++ Euclidean distance |
| `Makefile` | builds the `.so`; `test` target references five nonexistent files |
| `docs/kaggle-learning-notes/` | imported EDA/modelling reading list; stub, internal paths broken |

## Current State (verified 2026-11-27)

The package **imports cleanly** — the September blocker was fixed in `b42f6c5` (PR #1). That is the
only code change since 2026-03-28.

**Neither model can be trained on raw unscaled data.** Linear regression fails *silently*: on two
features of scale 1 and 50000 it diverges at epoch 2, reaches `inf` by ~epoch 40, and `fit()` returns
normally with `weights=[nan, nan]`. Logistic regression fails *loudly*: `sigmoid` at `utils.py:8`
raises `OverflowError: math range error` below about `x = -710`, which one feature of scale 1000 is
enough to trigger. Standard-scaling the same input makes linear regression fit exactly, so the
gradient maths is correct — scaling is the missing step, and `utils.py`'s scalers can't supply it for
held-out data (below).

**Nothing in the codebase computes a loss.** `grep -rn mse` matches only the comments at
`linear_regression.py:3-4`. No MSE, no accuracy, no score, no loss history — so a diverging run and a
converging run look identical from outside, which is why the NaN above goes unnoticed.

**There are no tests.** `.gitignore:27` ignores `test/`; `git ls-files` returns 20 files and zero
tests; the directory is absent from disk. `make test` fails immediately.

**The C++ backend is slower than pure Python**, unchanged since September. At 2000×20, 20 reps:
`matrix_vector_multiply` 0.259 s vs 0.114 s pure Python (2.3x slower); KNN's per-row
`euclidean_distance` 0.530 s vs 0.025 s for stdlib `math.dist` (21x slower). Every wrapper re-marshals
its inputs per call and that dominates the arithmetic. The C++ is correct and fast; the boundary is
crossed at the wrong granularity. `README.md:41` claims the opposite.

**Other confirmed correctness bugs:**
- `CPP_vector.py` never validates lengths across arguments. `predict()` on an unfitted model
  (`weights == []`) returns nondeterministic values read from unallocated memory instead of raising.
  Ragged matrices misalign silently; a short second argument to `dot`/`add`/`euclidean_distance` is
  zero-padded, so KNN answers dimension-mismatched queries with a confident label.
- Both scalers (`utils.py:10`, `:48`) recompute statistics from whatever matrix they are handed and
  discard them, so training scaling cannot be applied to a test set.
- The `except TypeError` fallback in `LinearRegression.fit:20-24` prints a reassuring message and
  then crashes two lines later.

Full detail, with reproductions and measurements, in `todo.md`.

## Activity
Last human code commit: `a75feb6` (2026-03-28) — eight months ago — which added KNN, logistic
regression, the scalers, and `BaseModel`. The only `.py` change since is the agent's six-line import
fix (`b42f6c5`, 2026-09-21). Everything else is scaffold, automated review notes, and the
`docs/kaggle-learning-notes/` import. The three-layer architecture is sound and the C++ layer is
correct; the Python boundary around it is what needs work, and nobody is working on it.
