# TODOs

Ranked. Everything below was confirmed by running the code on 2026-09-18, not guessed.

---

### 1. `[QUICK WIN ~15min]` Make the package importable again

`models/linear_regression.py:10-15` calls `logging.basicConfig(filename="test/LinearReg_debug.log")`
at **module import time**. `test/` is gitignored (`.gitignore:27`) and does not exist on disk, so the
handler cannot open its file and the import dies:

```
FileNotFoundError: [Errno 2] No such file or directory: '.../test/LinearReg_debug.log'
```

`models/__init__.py:4` imports `linear_regression`, so this takes down **every** model — even
`from models.knn import KNN` fails. `import models` does not work from a clean checkout, from the
repo root, or from anywhere else.

Delete the `basicConfig` block. If per-run debug logging is still wanted, use
`logger = logging.getLogger(__name__)` at module level and let the *caller* attach a handler —
a library should never configure the root logger or create files on import.

This is 6 lines and it is the difference between the project working and not working. Do it first.

---

### 2. `[BUG]` Validate vector dimensions at the ctypes boundary

`ml_math/CPP_vector.py` sizes its C buffers from one argument and never checks the other. Two
distinct failure modes, both silent:

**Out-of-bounds read** — `matrix_vector_multiply` (`CPP_vector.py:40-55`) builds `vec_arr` with
`len(vec)` elements, but `vector_math.cpp:25` loops `j < cols` where `cols = len(matrix[0])`.
When `len(vec) < cols` the C++ reads past the end of the allocation. Observed:

```python
V.matrix_vector_multiply([[0,0,1],[0,0,2]], [0.0])
# -> [8.3214395e-317, 1.6642879e-316]   and the values change on every call
```

Reachable from the model layer: `LinearRegression.predict(X)` / `LogisticRegression.predict(X)`
never check that `X`'s column count matches `len(self.weights)`. Fit on 2 features, predict on 3,
and you get numbers out of unallocated memory with no exception.

**Silent zero-padding** — `dot`, `add`, `euclidean_distance` (`:18-31`, `:57-64`) all do
`size = len(a)` and then `(c_double * size)(*b)`. A shorter `b` gets zero-filled:

```python
V.euclidean_distance([3,4,99], [0,0])   # -> 99.126, i.e. b became [0,0,0]
```

So `KNN.predict` with a 3-D query against 2-D training data returns a confident label and no
warning (verified: distances `[15.588, 9.11]`, label `1`). A longer `b` instead throws
`IndexError: invalid index` from inside ctypes — technically correct, completely unhelpful.

Add an explicit length check to each wrapper raising `ValueError` with both lengths named.
~30 min for all five functions.

---

### 3. `[TEST]` Get the test suite back under version control

`.gitignore:27` ignores `test/`. `git ls-files` returns **zero** test files, and the directory is
not on disk either. `Makefile:21-25` invokes five scripts that do not exist, so `make test` — the
command the README tells people to run — fails immediately for everyone.

Drop `test/` from `.gitignore`, recreate the suite, commit it. Then swap the five hand-listed
`@python3 test/...` lines for `pytest test/` (an old TODO item, still valid): the current form
stops at the first failure and the `@` prefix hides which command died.

**The one test worth writing first:** a multi-feature `LinearRegression` fit-then-predict where
the prediction matrix has a *different* column count than the training matrix. The last real code
commit (`a75feb6` "fixed matrix math and added linear regression model fixes") was exactly this
area, and that single test catches issue #2's out-of-bounds read, the shape-mismatch gap, and any
future regression in `matrix_vector_multiply` at once.

---

### 4. `[BUG]` `sigmoid` overflows on moderately large negative input

`ml_math/utils.py:8` is the textbook `1 / (1 + math.exp(-x))`, which raises
`OverflowError: math range error` once `x <= ~-710`:

```
sigmoid(-700) -> 9.86e-305      sigmoid(-710) -> OverflowError
```

`LogisticRegression.fit` (`logistic_regression.py:26`) feeds `w·x + b` straight in. With
unscaled features — which is the default, since nothing calls the scalers in `utils.py` — a
handful of epochs is enough to get there, and training dies mid-loop with a bare traceback.

Use the branched stable form (`exp(x)/(1+exp(x))` for negative `x`) so it saturates to 0.0 and
1.0 instead of throwing. ~15 min including a test at ±1000.

---

### 5. `[DESIGN]` Fix the dead fallback and the logging in the training loop

Both in `models/linear_regression.py`:

**The `TypeError` recovery at `:25-29` does not recover.** For a 1-D `X` it prints a reassuring
message, sets `n_features = 1`, and then crashes anyway four lines later at `:41`, because
`matrix_vector_multiply` also calls `len(matrix[0])`:

```
Recieved TypeError due to python not being able to recognize all features...
TypeError: object of type 'float' has no len()
```

Either normalise 1-D input to `[[x] for x in X]` up front and make the fallback real, or drop the
`try/except` and raise a clear `ValueError`. The current version is strictly worse than no
handler — it converts a clear error into a misleading message followed by the same error.
(Also: `print` → logger, and "Recieved" is misspelled.)

**Three `logging.debug` calls sit inside the per-sample loop** (`:38`, `:45-47`). At the default
1000 epochs that is `3 × n_samples × 1000` formatted log records — over 3M writes for a
1000-row dataset, all f-strings evaluated eagerly whether or not the level is enabled. Move the
per-epoch line out of the sample loop and delete the per-sample ones, or guard with
`if logger.isEnabledFor(logging.DEBUG)`.

---

## Parked

Still worthwhile, but not while the package does not import:

- **Naive Bayes** (`models/naive_bayes.py` + test, wired into the Makefile) — no C++ changes needed.
- **More C++ distance metrics** — Manhattan and cosine in `vector_math.cpp`, which would make
  `KNN.k` a genuine hyperparameter sweep.
- **Feedforward net with one hidden layer** — reuses `dot` and `matrix_vector_multiply` as-is.
- **C++ vs pure-Python benchmark** at 100 / 1K / 10K / 100K elements, to make the "why C++" case visible.
- **Packaging** (`pyproject.toml`, `pip install -e .`). Note `ml_math/` has no `__init__.py` and
  currently works only as an implicit namespace package — worth fixing as part of this.
- **Stop committing `cpp/libvector.so`.** It is a tracked, architecture-specific binary that will
  drift from `vector_math.cpp` with no signal. `make` already builds it.
