# TODOs

Ranked. Everything below was re-run against the code on 2026-11-27.

> **Since the 2026-09-29 review, exactly one thing changed: task 1 landed** (PR #1, `b42f6c5`) —
> `import models` works again. Tasks 2–5 from that list are untouched and all four still reproduce.
>
> **The last code commit written by a human is `a75feb6` (2026-03-28) — eight months ago.** The only
> `.py` change since then is the six-line import fix the doer agent made in September. This repo is
> currently agent-maintained, not human-active. The architecture is genuinely sound and the C++ is
> correct, so it is worth finishing — but if nobody intends to pick it up, demote it out of
> `01_active_projects` and stop the review churn. That is a decision, not a task; make it either way.

---

### 1. `[BUG]` `fit()` silently returns NaN weights — make the loss observable

**Neither model can be trained on raw unscaled data, and the linear one does not tell you.** Four
rows, two features on ordinary real-world scales (a count and a salary):

```python
X = [[1.0, 50000.0], [2.0, 60000.0], [3.0, 70000.0], [4.0, 80000.0]]
y = [1.0, 2.0, 3.0, 4.0]
m = LinearRegression()          # defaults: lr=0.01, epochs=1000
m.fit(X, y)                     # returns normally
m.weights                       # -> [nan, nan]
m.predict(X)                    # -> [nan, nan, nan, nan]
```

No exception, no warning. It diverges on **epoch 2** and gains ~8 orders of magnitude per epoch:

```
epochs=1   weights=['0.15',     '3.5e+03']
epochs=2   weights=['-1.22e+07','-3.04e+11']
epochs=6   weights=['-7.02e+38','-1.74e+43']
epochs=20  weights=['-9.99e+149','-2.48e+154']
```

It hits `inf` around epoch 40, then `inf - inf` makes it NaN, and the remaining ~960 epochs run for
nothing. Standard-scale the same `X` first and it fits perfectly (`weights=[0.559, 0.559]`,
`bias=2.5`, predictions exactly `[1,2,3,4]`) — so the gradient maths at `linear_regression.py:46-52`
is right. The problem is that **a diverging run and a converging run are indistinguishable from the
outside.**

Root cause of the blindness: `grep -rn mse` finds MSE only in the comments at
`linear_regression.py:3-4`. **The loss function the whole README is built around is never computed
anywhere in the codebase** — no `mse`, no `score`, no accuracy, no loss history. `fit()` runs exactly
`epochs` iterations and reports nothing.

Scope for this hour:
- Add `mse(y_true, y_pred)` to `ml_math/utils.py` (it is three lines and it is conspicuously absent).
- In `LinearRegression.fit`, compute the loss per epoch from the `y_preds` it **already has** at
  `linear_regression.py:36-37` — zero extra cost — and append to `self.loss_history`.
- If the loss is not finite, `raise ValueError` naming the epoch and suggesting a lower `lr` or
  scaled features. Failing loudly at epoch 2 beats 1000 epochs of NaN.
- Replace the three per-sample `logger.debug` calls (`:40-42`, `3 × n_samples × epochs` eagerly
  formatted f-strings) with one per-epoch line that logs the loss. That is what you actually wanted
  to see in a debug log.

Do this first. It is the difference between a library you can trust and one you cannot.

---

### 2. `[QUICK WIN ~15min]` Numerically stable `sigmoid` — logistic regression currently crashes

`ml_math/utils.py:8` is the textbook unstable form, `1 / (1 + math.exp(-x))`. `sigmoid(-700)` is fine
(`9.86e-305`); `sigmoid(-710)` raises `OverflowError: math range error`. `LogisticRegression.fit:26`
feeds `w·x + b` straight in, so on perfectly ordinary input it dies mid-training:

```python
X = [[0.0,1000.0],[1.0,2000.0],[2.0,3000.0],[3.0,4000.0]]
y = [0,0,1,1]
LogisticRegression().fit(X, y)
# OverflowError: math range error
```

One feature of scale 1000 is enough. Use the branched form — `1/(1+exp(-x))` for `x >= 0`, and
`exp(x)/(1+exp(x))` for `x < 0` — so it saturates to `0.0` instead of throwing. `math.exp` of a
negative number cannot overflow, so both branches are safe.

Smallest change in this list with the most visible result: logistic regression goes from *crashing*
to *training* on unscaled data. Promoted from "Parked" because task 1 proved that the unscaled-input
path is the main thing broken here, and this is the other half of it.

---

### 3. `[BUG]` Validate lengths at the ctypes boundary — `predict()` before `fit()` reads unallocated memory

`ml_math/CPP_vector.py` sizes its C buffers from one argument and never checks the other. The most
user-facing consequence, new this review: **`predict()` on an unfitted model returns garbage out of
unallocated memory instead of raising.** `BaseModel.__init__` sets `self.weights = []`
(`base.py:9`), so `cols=3` but `vec` has length 0:

```python
m = LinearRegression()                              # never fitted
m.predict([[1.0,2.0,3.0],[4.0,5.0,6.0]])
# -> [2.49643184e-316, 4.9928637e-316]
# -> [2.4964317e-316,  4.9928634e-316]   different values on the next call
```

Three distinct failure modes, all silent, all re-confirmed today:

- **Out-of-bounds read** — `matrix_vector_multiply` (`:40-55`) builds `vec_arr` from `len(vec)` but
  `vector_math.cpp:25` loops `j < cols` where `cols = len(matrix[0])`.
  `V.matrix_vector_multiply([[0,0,1],[0,0,2]], [0.0])` → `[8.32e-317, 1.66e-316]`, nondeterministic.
- **Ragged matrices misalign silently** — `cols` comes from row 0 only, but the flatten at `:48`
  ignores `cols`. `V.matrix_vector_multiply([[1,1,1],[5,5],[1,1,1]], [1,1,1])` → `[3.0, 11.0, 2.0]`;
  the correct answer is `[3.0, 10.0, 3.0]`. Row 1 read `[5,5,1]`, row 2 ran off the end.
- **Silent zero-padding** — `dot`, `add`, `euclidean_distance` (`:18-31`, `:57-64`) do
  `size = len(a)` then `(c_double * size)(*b)`, so a shorter `b` is zero-filled:
  `V.euclidean_distance([3,4,99], [0,0])` → `99.126`. So `KNN.predict` answers a 3-D query against
  2-D training data with a confident label: `k.predict([[1.0,1.0,999.0]])` → `['a']`, no warning.

Add an explicit check to each of the five wrappers raising `ValueError` with both lengths named, plus
a rectangularity check in `matrix_vector_multiply`. ~30 min. While you are there, have `predict()`
raise if `self.weights` is empty — "call fit() first" is a better error than a float from the heap.

---

### 4. `[TEST]` Get a test suite back under version control

`.gitignore:27` ignores `test/`. `git ls-files` lists **20 files and not one test**, and `test/` is
absent from disk. `Makefile:21-25` invokes five scripts that do not exist, so `make test` — the
command `README.md:62` tells people to run — fails for everyone.

Drop `test/` from `.gitignore`, add a `test/` directory, commit it. Then replace the five hand-listed
`@python3 test/...` lines with `pytest test/`: the current form stops at the first failure and the
`@` prefix hides which command died.

**The one test worth writing first:** `import models` — a bare smoke test, one line. The most recent
bug-shaped commit is `b42f6c5` ("Make the package importable again"), where a module-level
`logging.basicConfig` into a nonexistent directory broke *every* model for months. A one-line import
test would have caught it the day it landed, and it is the cheapest regression guard in the repo.

Then, in order: a `fit()`-on-unscaled-data test asserting the loss is finite (locks in task 1), and
a shape-mismatch test asserting `ValueError` rather than heap garbage (locks in task 3).

---

### 5. `[DESIGN]` Scalers cannot be applied to held-out data

`min_max_scaler` (`ml_math/utils.py:10`) and `standard_scaler` (`:48`) take a matrix, compute the
column statistics from *that* matrix, scale, and throw the statistics away. There is no way to apply
the training scaling to a test set. Verified again today:

```python
min_max_scaler([[0.0],[10.0]])   # train -> [[0.0], [1.0]]
min_max_scaler([[5.0],[20.0]])   # test  -> [[0.0], [1.0]]
```

The test point `5.0` should map to `0.5` under the training scale and `20.0` to `2.0`. Instead each
set is re-normalised against itself, so held-out data silently lands in a different feature space
than the model was trained in, and any accuracy number from it is meaningless.

This is an API-shape problem, not a formula problem — the arithmetic is correct (population std at
`:72` matches sklearn). Split each into fit and transform: return the `(min, max)` / `(mean, std)`
per column alongside the scaled data, or add small `MinMaxScaler` / `StandardScaler` objects with
`fit` / `transform` / `fit_transform`.

Now coupled to tasks 1 and 2, which is why it stays ranked: scaling **is** the fix for the
divergence and the overflow, so these functions are about to go from "nothing calls them" to
"on the critical path". Also worth noting
`docs/kaggle-learning-notes/main_question.md` is built entirely on honest validation of held-out
Kaggle data and lists leakage and "weak validation" as the mistakes to avoid — these two functions
are that exact bug waiting for its first user.

---

## Parked

Still real, not worth a slot this round:

- **The C++ backend is a net performance loss, and `README.md:41` claims the opposite.** Re-measured
  today at 2000 samples × 20 features, 20 reps — unchanged since September:
  ```
  mvm as written  : 0.259 s      pure python mvm : 0.114 s   <- 2.3x FASTER
  knn dist in C++ : 0.530 s      stdlib math.dist: 0.025 s   <- 21x FASTER
  ```
  Every wrapper re-marshals its inputs on each call and the marshalling dominates the arithmetic.
  `X` never changes across epochs, yet `LinearRegression.fit:36` and `LogisticRegression.fit:25`
  re-flatten and re-copy it 1000 times. The fix is a `CMatrix` holder that marshals once, plus an
  `all_distances(matrix, query, ...)` in `vector_math.cpp` so KNN crosses the boundary once per query
  instead of once per training row. Was ranked #2 last review; **deliberately demoted** — a model
  that returns NaN does not need to be faster. Pick this up once 1–3 are done. Its own hour.
- **The `TypeError` recovery in `LinearRegression.fit:20-24` does not recover.** For a 1-D `X` it
  prints a reassuring message, sets `n_features = 1`, then dies at `:36` because
  `matrix_vector_multiply` also calls `len(matrix[0])`. An empty `X` raises `IndexError`, which the
  `except TypeError` does not catch either. Either normalise 1-D input to `[[x] for x in X]` up front
  and make the fallback real, or drop the `try/except` and raise a clear `ValueError`. Currently
  strictly worse than no handler. (Also `print` → logger; "Recieved" is misspelled.)
- **`logging` is imported and never used** in `models/logistic_regression.py:4`. One-line delete.
- **`make` dirties the working tree.** `cpp/libvector.so` is tracked (`git ls-files`) and
  `Makefile:11-14` rewrites it, so the first documented build command leaves `git status` dirty with
  an architecture-specific binary. Gitignore the `.so` and keep only `vector_math.cpp`; `make`
  already rebuilds it. (It is currently in sync — both files last changed in `a75feb6` — so this is
  latent, not broken.)
- **No packaging, and `ml_math/` has no `__init__.py`.** It only resolves as an implicit namespace
  package, so imports work from the repo root and nowhere else — running any script from outside the
  directory gives `ModuleNotFoundError: No module named 'models'`. A `pyproject.toml` plus
  `pip install -e .` fixes both; also needed before `pytest` is pleasant.
- **KNN design mismatches.** It inherits `BaseModel`, which allocates `weights`/`bias` that KNN never
  uses (`base.py:8-10`); `k > len(X_train)` silently uses fewer neighbours; `predict` before `fit`
  raises `IndexError` from `Counter([]).most_common(1)` at `knn.py:41`; and ties in `most_common`
  resolve by insertion order, which makes even-`k` results arbitrary-but-deterministic.
- **Naive Bayes** (`models/naive_bayes.py` + test, wired into the Makefile) — no C++ changes needed.
- **More C++ distance metrics** — Manhattan and cosine in `vector_math.cpp`, which would make
  `KNN.k` a genuine hyperparameter sweep.
- **Feedforward net with one hidden layer** — reuses `dot` and `matrix_vector_multiply` as-is.
- **`docs/kaggle-learning-notes/` is still a stub.** `main_question.md:80-83` points at
  `kaggle_titanic/`, `prediction_modeling/`, and `research_reports/run_logs/`, none of which exist
  here; `wiki/INDEX.md` still names the repo it was folded in from (`kaggle_strategies_models`) and
  lists no pages. Either fix the paths or state plainly that it is an imported reading list, not
  project structure.
