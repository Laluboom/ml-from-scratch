# Daily brief — ml-from-scratch (2026-11-27)

## What I looked at

Full git history (and history filtered to `*.py` / `*.cpp` / `Makefile`), the README, the previous
`todo.md` and `reference.md`, and every source file: `cpp/vector_math.cpp`, both `ml_math` modules,
and all four files in `models/`. Then I ran the library — trained both regressions on realistic
unscaled input, exercised the ctypes wrappers at mismatched dimensions, re-timed the C++ bridge
against pure Python, and checked `git ls-files` for tests. Not a web project, so no browser check.

I did **not** run `make`: `cpp/libvector.so` is git-tracked and the build rewrites it, which would
have dirtied a tracked file. (That is itself a small finding — see Parked.)

## State since last review

The September blocker is fixed. PR #1 (`b42f6c5`) removed the module-level `logging.basicConfig` and
`import models` works cleanly now. That is the good news, and it is all of it: tasks 2–5 from the
2026-09-29 list are untouched and all four still reproduce verbatim.

Worth stating plainly: **the last code commit written by a human is `a75feb6`, 2026-03-28 — eight
months ago.** The only `.py` change since is the six-line fix the doer agent made in September. This
repo is agent-maintained, not human-active. The architecture is genuinely good and the C++ is
correct, so finishing it is worthwhile — but if nobody intends to, demote it out of
`01_active_projects` rather than letting the review churn continue. Either answer is fine; the
non-answer is the expensive one.

## The main finding

**Neither model can be trained on raw unscaled data, and the linear one does not tell you.**

Four rows, two features on ordinary scales (a count and a salary), library defaults:

```python
X = [[1.0, 50000.0], [2.0, 60000.0], [3.0, 70000.0], [4.0, 80000.0]]
LinearRegression().fit(X, [1.0, 2.0, 3.0, 4.0])   # returns normally
# weights -> [nan, nan]      predict(X) -> [nan, nan, nan, nan]
```

It diverges on epoch 2, gains roughly eight orders of magnitude per epoch, hits `inf` by about epoch
40, and then `inf - inf` turns everything to NaN — after which ~960 epochs run for nothing. No
exception, no warning. Standard-scale the same `X` and it fits exactly (`weights=[0.559, 0.559]`,
`bias=2.5`, predictions `[1,2,3,4]`), so the gradient maths at `linear_regression.py:46-52` is right.

The reason this is invisible is the more interesting half. `grep -rniE 'mse|loss|score|accuracy'`
across every `.py` and `.cpp` in the repo matches **only the comments** at `linear_regression.py:3-5`.
There is no MSE function, no accuracy, no score, no loss history. The loss function the entire README
is built around is never computed, so a diverging run and a converging run are indistinguishable from
outside. That is task 1: compute the loss from the `y_preds` `fit()` already has (free), record it,
and raise on a non-finite value. Failing loudly at epoch 2 beats 1000 epochs of NaN.

Logistic regression has the same root cause with the opposite symptom — it fails loudly. `sigmoid` at
`utils.py:8` is the unstable `1/(1+exp(-x))` form, and one feature of scale 1000 is enough to kill it:
`LogisticRegression().fit([[0,1000],[1,2000],[2,3000],[3,4000]], [0,0,1,1])` raises
`OverflowError: math range error` mid-training. That is the quick win — a three-line branched sigmoid
takes logistic regression from crashing to training, which is the most visible fifteen minutes
available here. I promoted it out of "Parked", where the last review left it, because task 1 showed
the unscaled-input path is the main thing actually broken.

## Also proposing

The ctypes boundary bug (previously task 3) picked up a much better reproduction this round:
`predict()` on an **unfitted** model returns garbage from unallocated memory rather than raising,
because `BaseModel` initialises `weights = []` and nothing checks it against the matrix column count.
Values differ on each call. Ragged matrices also misalign silently —
`matrix_vector_multiply([[1,1,1],[5,5],[1,1,1]], [1,1,1])` returns `[3.0, 11.0, 2.0]` where the right
answer is `[3.0, 10.0, 3.0]`. Validating five wrappers is about thirty minutes.

Then tests, where I have a sharper answer than last time: the single most valuable test is now
`import models`, one line. The most recent bug-shaped commit is exactly the import failure PR #1
fixed, which broke every model for months and would have been caught the day it landed. There are
currently 20 tracked files and zero tests, and `make test` invokes five scripts that do not exist.
Finally the scaler redesign stays ranked, and is now coupled to tasks 1 and 2 — scaling *is* the fix
for the divergence and the overflow, so `utils.py`'s scalers are about to move from "nothing calls
them" onto the critical path, and they currently cannot transform held-out data at all.

I demoted last review's #2 — the C++ bridge being 2.3x slower than pure Python for matrix-vector and
21x slower for KNN distances. The numbers are unchanged and the finding is real, but a model that
returns NaN does not need to be faster first.
