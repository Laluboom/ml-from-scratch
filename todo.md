# TODOs

Ranked. Everything below was re-confirmed by running the code on 2026-09-29.

> **Nothing from the 2026-09-18 list was actioned.** The import failure below is byte-for-byte the
> same bug, 11 days later. The last commit that touched a `.py` or `.cpp` file is `a75feb6`
> (2026-03-28) — **six months ago**. Everything since is scaffold, review notes, and the Kaggle
> docs import. If this repo is no longer being worked on, that is a fine answer, but say so and
> move it out of `01_active_projects` rather than letting the review churn continue.

---

### 1. `[QUICK WIN ~15min]` Make the package importable again

`models/linear_regression.py:10-15` calls `logging.basicConfig(filename="test/LinearReg_debug.log")`
at **module import time**. `test/` is gitignored (`.gitignore:27`) and does not exist on disk, so the
handler cannot open its file and the import dies:

```
FileNotFoundError: [Errno 2] No such file or directory: '.../test/LinearReg_debug.log'
```

`models/__init__.py:4` imports `linear_regression`, so this takes down **every** model — even
`from models.knn import KNN` fails. Re-verified today: `import models` does not work from a clean
checkout, from the repo root, or from anywhere else.

Delete the `basicConfig` block. If per-run debug logging is still wanted, use
`logger = logging.getLogger(__name__)` at module level and let the *caller* attach a handler —
a library should never configure the root logger or create files on import.

**Do not "fix" this by creating `test/`.** That makes the symptom go away and leaves a library that
writes files into your CWD on import. While you are in the file: `logging` is imported and never
used in `models/logistic_regression.py:4` — drop it.

This is 6 lines and it is the difference between the project working and not working. Do it first;
nothing else on this list is testable until it lands.

---

### 2. `[IMPROVEMENT]` Marshal `X` once per `fit()`, not once per epoch

**The C++ backend is currently a net performance loss.** Every wrapper in `CPP_vector.py` rebuilds
its C buffers from Python lists on every call, and that marshalling costs more than the arithmetic
it exists to accelerate. Measured today at 2000 samples × 20 features, 20 reps:

```
marshal X only        : 0.249 s     <- 96% of the call
mvm as written        : 0.260 s
pure python mvm       : 0.129 s     <- 2x FASTER than the C++ path
mvm X pre-marshalled  : 0.001 s     <- 260x faster than as-written
```

`LinearRegression.fit:41` and `LogisticRegression.fit:25` call `matrix_vector_multiply(X, ...)`
inside the epoch loop. `X` never changes across epochs, so at the default `epochs=1000` the same
matrix is re-flattened into a Python list and re-copied into a fresh `c_double` array **1000
times**. That one hoist is essentially the entire cost of training.

Scope for this hour:
- In `ml_math/CPP_vector.py`, add a small holder that marshals a matrix once — e.g. a `CMatrix`
  class storing `(arr, rows, cols)` with a `matvec(vec)` method that only marshals `vec`.
- Keep the existing `matrix_vector_multiply(matrix, vec)` as a thin one-shot wrapper so nothing
  else breaks.
- Hoist `Xc = CMatrix(X)` above the epoch loop in both `fit()` methods and call `Xc.matvec(...)`.

Note the gradient loops (`linear_regression.py:44-53`, `logistic_regression.py:31-35`) stay pure
Python and O(n·d), so the backward pass will then dominate. That is expected and fine — it is the
next thing to move into C++, not a reason to skip this.

---

### 3. `[BUG]` Validate vector dimensions at the ctypes boundary

`ml_math/CPP_vector.py` sizes its C buffers from one argument and never checks the other. Two
distinct failure modes, both silent, both re-confirmed today.

**Out-of-bounds read** — `matrix_vector_multiply` (`CPP_vector.py:40-55`) builds `vec_arr` with
`len(vec)` elements, but `vector_math.cpp:25` loops `j < cols` where `cols = len(matrix[0])`.
When `len(vec) < cols` the C++ reads past the end of the allocation:

```python
V.matrix_vector_multiply([[0,0,1],[0,0,2]], [0.0])
# -> [8.3214395e-317, 1.6642879e-316]
# -> [8.321439e-317,  1.6642878e-316]   different values on the very next call
```

Reachable from the model layer: `LinearRegression.predict(X)` / `LogisticRegression.predict(X)`
never check that `X`'s column count matches `len(self.weights)`. Fit on 2 features, predict on 3,
and you get numbers out of unallocated memory with no exception.

**Silent zero-padding** — `dot`, `add`, `euclidean_distance` (`:18-31`, `:57-64`) all do
`size = len(a)` and then `(c_double * size)(*b)`. A shorter `b` gets zero-filled:

```python
V.euclidean_distance([3,4,99], [0,0])   # -> 99.126, i.e. b silently became [0,0,0]
```

So `KNN.predict` with a 3-D query against 2-D training data returns a confident label and no
warning. A longer `b` instead throws `IndexError: invalid index` from inside ctypes — technically
an error, completely unhelpful.

Add an explicit length check to each wrapper raising `ValueError` with both lengths named. Fold it
into the `CMatrix` work from task 2 while you are in the file. ~30 min for all five functions.

---

### 4. `[TEST]` Get the test suite back under version control

`.gitignore:27` ignores `test/`. `git ls-files` returns **zero** test files and the directory is
not on disk either. `Makefile:21-25` invokes five scripts that do not exist, so `make test` — the
command the README tells people to run — fails immediately for everyone.

Drop `test/` from `.gitignore`, recreate the suite, commit it. Then replace the five hand-listed
`@python3 test/...` lines with `pytest test/`: the current form stops at the first failure and the
`@` prefix hides which command died.

**The one test worth writing first:** a multi-feature `LinearRegression` fit-then-predict where the
prediction matrix has a *different* column count than the training matrix. The last real code
commit (`a75feb6`, "fixed matrix math and added linear regression model fixes") was exactly this
area, and that single test catches task 3's out-of-bounds read, the shape-mismatch gap, and any
future regression in `matrix_vector_multiply` at once.

Second test, because task 2 will touch this code: assert `CMatrix(X).matvec(w)` equals
`matrix_vector_multiply(X, w)` for a non-square `X`. Cheap, and it is the only thing standing
between a marshalling optimisation and a silently wrong training loop.

---

### 5. `[DESIGN]` Scalers cannot be applied to held-out data

`min_max_scaler` (`ml_math/utils.py:10`) and `standard_scaler` (`:48`) take a matrix, compute the
column statistics from *that* matrix, scale, and throw the statistics away. There is no way to
apply the training scaling to a test set. Verified:

```python
min_max_scaler([[0.0],[10.0]])   # train -> [[0.0], [1.0]]
min_max_scaler([[5.0],[20.0]])   # test  -> [[0.0], [1.0]]
```

The test point `5.0` should map to `0.5` under the training scale and `20.0` to `2.0`. Instead both
rows are re-normalised against themselves, so every held-out set silently gets a different feature
space than the model was trained in. Any accuracy number produced this way is meaningless.

This is the API shape problem, not a formula problem — the arithmetic in both functions is correct
(population std at `:72` matches sklearn). Split each into a fit and a transform: return the
`(min, max)` / `(mean, std)` per column alongside the scaled data, or introduce a tiny
`MinMaxScaler` / `StandardScaler` object with `fit`, `transform`, `fit_transform`.

Worth ranking now rather than later: the newest thing in this repo is
`docs/kaggle-learning-notes/main_question.md`, whose whole premise is honest validation on held-out
Kaggle data — and it lists leakage and "weak validation" (`:26`) as the mistakes to avoid. These
two functions are a leakage bug waiting for its first user. Nothing currently calls either scaler,
so the change is free to make today.

---

## Parked

Still real, not worth a slot this round:

- **`sigmoid` overflows on large negative input** (`ml_math/utils.py:8`). `sigmoid(-700)` is fine at
  `9.86e-305`; `sigmoid(-710)` raises `OverflowError: math range error`. `LogisticRegression.fit:26`
  feeds `w·x + b` straight in, so with unscaled features training dies mid-loop with a bare
  traceback. Use the branched stable form (`exp(x)/(1+exp(x))` for negative `x`) so it saturates to
  0.0 instead of throwing. **~15 min — fold it into whichever task you touch `utils.py` for**
  (task 5 is the natural one).
- **The `TypeError` recovery in `LinearRegression.fit:25-29` does not recover.** For a 1-D `X` it
  prints a reassuring message, sets `n_features = 1`, then crashes four lines later at `:41`
  because `matrix_vector_multiply` also calls `len(matrix[0])`. Either normalise 1-D input to
  `[[x] for x in X]` up front and make the fallback real, or drop the `try/except` and raise a clear
  `ValueError`. Currently strictly worse than no handler. (Also `print` → logger; "Recieved" is
  misspelled.)
- **Three `logging.debug` calls inside the per-sample loop** (`linear_regression.py:38`, `:45-47`).
  At 1000 epochs that is `3 × n_samples × 1000` records with eagerly-evaluated f-strings. Move the
  per-epoch line out of the sample loop and delete the per-sample ones. Largely moot once task 1
  removes the file handler, but the f-string cost stays.
- **Push the KNN distance loop into C++.** `KNN._predict_single:31` calls across the ctypes boundary
  once per training row, re-marshalling the query vector every time. Measured at 2000×20: 0.542 s
  versus 0.028 s for stdlib `math.dist` — **19x slower**, and `README.md:41`'s claim that distances
  are "offloaded to C++ to handle larger datasets efficiently" is measurably backwards. The fix is
  one `all_distances(matrix, query, rows, cols, out)` function in `vector_math.cpp` plus a wrapper,
  so the boundary is crossed once per query instead of once per row. Its own hour (C++ change +
  rebuild + test), and it depends on task 2's `CMatrix`.
- **Naive Bayes** (`models/naive_bayes.py` + test, wired into the Makefile) — no C++ changes needed.
- **More C++ distance metrics** — Manhattan and cosine in `vector_math.cpp`, which would make
  `KNN.k` a genuine hyperparameter sweep.
- **Feedforward net with one hidden layer** — reuses `dot` and `matrix_vector_multiply` as-is.
- **Packaging** (`pyproject.toml`, `pip install -e .`). Note `ml_math/` has no `__init__.py` and
  currently works only as an implicit namespace package — worth fixing as part of this.
- **Stop committing `cpp/libvector.so`.** Tracked, architecture-specific binary. It is currently
  *in sync* — both it and `vector_math.cpp` were last written by `a75feb6` — so this is latent, not
  broken. It becomes a real problem the first time the `.cpp` changes without a rebuild. `make`
  already builds it.
- **`docs/kaggle-learning-notes/` is a stub.** `main_question.md:80-83` points at `kaggle_titanic/`,
  `prediction_modeling/`, and `research_reports/run_logs/`, none of which exist here; `wiki/INDEX.md`
  still names the repo it was folded in from (`kaggle_strategies_models`) and lists no pages. Either
  fix the paths or state plainly that it is an imported reading list, not project structure.
