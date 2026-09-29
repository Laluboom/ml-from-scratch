# Daily brief — ml-from-scratch (2026-09-29)

## What I looked at

The whole thing; it is only 337 lines of Python plus 37 of C++. Read all five model files, both
`ml_math` modules, `vector_math.cpp`, the Makefile, README, `.gitignore`, and the previous
`todo.md` / `reference.md`. Ran `git log`, `git log --stat`, and `git ls-files`. Then actually
executed the library: reproduced the import failure, the out-of-bounds read, the zero-padding, the
sigmoid overflow, and the scaler behaviour, and benchmarked the ctypes boundary. Not a web project,
so no browser check.

## The uncomfortable part first

**Nothing from the 2026-09-18 review was done.** The import failure is byte-for-byte identical
eleven days later. More to the point: the last commit that touched a `.py` or `.cpp` file is
`a75feb6`, dated **2026-03-28** — six months ago. Every commit since is scaffold, automated review
notes, or the Kaggle docs import. This repo sits in `01_active_projects` and is not being worked on.

`import models` still raises `FileNotFoundError`, because `models/linear_regression.py:10` calls
`logging.basicConfig(filename="test/LinearReg_debug.log")` at module import time and `test/` is
gitignored and absent. `models/__init__.py` pulls that module in, so every model is unreachable —
you cannot even `from models.knn import KNN`. Six lines.

## The new finding: the C++ backend is currently slower than pure Python

This is the one worth your attention, and I only found it by measuring. The README's premise is that
heavy arithmetic is offloaded to compiled C++ for speed. As written, that is backwards, because
every `CPP_vector` wrapper rebuilds its C buffers from Python lists on each call and the marshalling
costs more than the arithmetic. At 2000 samples × 20 features, 20 reps:

```
marshal X only        : 0.249 s     <- 96% of the call is just copying
mvm as written        : 0.260 s
pure python mvm       : 0.129 s     <- 2x faster than the C++ path
mvm X pre-marshalled  : 0.001 s     <- 260x faster than as-written
```

And in the KNN path, where `_predict_single` crosses the boundary once per training row:
`0.542 s` via `CPP_vector.euclidean_distance` against `0.028 s` for stdlib `math.dist`, identical
results. **19x slower.** `README.md:41` says the distance calculations are "offloaded to C++ to
handle larger datasets efficiently."

The C++ itself is fine — it is tight, correct, and `-O3`. The problem is purely that the boundary is
crossed at the wrong granularity. And the cheapest instance of it is glaring: `LinearRegression.fit`
and `LogisticRegression.fit` call `matrix_vector_multiply(X, ...)` *inside* the epoch loop, so at the
default `epochs=1000` the same unchanging matrix is re-flattened and re-copied a thousand times.
Hoisting that out is most of the cost of training.

## What I'm proposing

Five tasks, ranked in `todo.md`. Two are new this round.

1. **`[QUICK WIN ~15min]`** Delete the import-time `basicConfig`. Nothing else is testable until
   this lands. Worth saying explicitly: do *not* fix it by creating `test/` — that hides the symptom
   and leaves a library that writes files into your CWD on import.
2. **`[IMPROVEMENT]`** Marshal `X` once per `fit()` instead of once per epoch — a small `CMatrix`
   holder in `CPP_vector.py` plus a hoist in both `fit()` methods. This is the finding above.
3. **`[BUG]`** Length checks in all five ctypes wrappers. Carried forward and re-verified:
   `matrix_vector_multiply([[0,0,1],[0,0,2]], [0.0])` returns different uninitialised memory on
   consecutive calls, and `euclidean_distance([3,4,99],[0,0])` silently zero-pads to 99.126.
4. **`[TEST]`** Un-gitignore `test/`, recreate it, `pytest` in the Makefile. The single most valuable
   test is a `LinearRegression` fit-then-predict with mismatched column counts — that is exactly the
   area of the last real code commit ("fixed matrix math and added linear regression model fixes")
   and it catches the out-of-bounds read, the shape gap, and any `matrix_vector_multiply` regression
   together.
5. **`[DESIGN]`** Both scalers recompute their statistics from whatever matrix you hand them and
   throw them away, so you cannot apply training scaling to held-out data:
   `min_max_scaler([[5.0],[20.0]])` gives `[[0.0],[1.0]]` where the training scale would give
   `[[0.5],[2.0]]`. Ranked now because the newest thing in the repo is
   `docs/kaggle-learning-notes/`, whose entire agenda is honest validation and which lists leakage as
   the mistake to avoid. Nothing calls either scaler yet, so fixing the API is free today.

Parked with full evidence: the `sigmoid` overflow below `x ≈ -710` (15 min, fold into task 5), the
`TypeError` fallback in `fit` that prints a reassuring message and then crashes anyway four lines
later, the three eagerly-formatted `logging.debug` calls inside the per-sample loop, and pushing the
KNN distance loop into C++ as a single `all_distances` call (its own hour, depends on task 2).

## Verdict

Good bones, abandoned in place. The layering is sensible, the C++ is correct, and the learning
intent is real — but the package does not import, there are no tests, and the performance premise is
currently inverted. Roughly two focused hours across tasks 1–3 would make it work *and* make the
C++ claim true. If those two hours are not going to happen, the honest move is to re-tier this out
of `01_active_projects` rather than generate another review of the same five bugs in ten days.
