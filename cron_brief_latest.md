# Daily review — ml-from-scratch — 2026-09-18

## What I looked at

Full read of the tracked tree: `cpp/vector_math.cpp`, both `ml_math/` modules, all four files in
`models/`, the Makefile, README, and the previous `reference.md` / `todo.md`. Then I actually ran
the code — importing the package, driving `LinearRegression` and the ctypes wrappers directly, and
probing the Python→C++ boundary with deliberately mismatched shapes. Not a web project, so no
browser check.

## What I found

**The package does not import.** `models/linear_regression.py:10` calls
`logging.basicConfig(filename="test/LinearReg_debug.log")` at module scope. `test/` was gitignored
back in `e022bcc` and no longer exists, so opening the log file raises `FileNotFoundError` before
any class is defined. `models/__init__.py:4` imports that module, which means *every* model is
unreachable — even `from models.knn import KNN`. This reproduces from the repo root, not just a
clean clone. It is six lines of fix and it gates everything else in the project.

**There are no tests.** `.gitignore:27` ignores `test/`; `git ls-files` returns zero test files and
the directory is gone from disk. `Makefile:21-25` still invokes five scripts that do not exist, so
the `make test` that README.md:57 advertises fails for anyone who tries it. Worth flagging: the
previous `reference.md` reported a passing `make test` with per-script results including
`test_linear_reg.py` — a script the Makefile never even calls. I removed that section rather than
carry a false green signal forward.

**The deep finding is the ctypes boundary.** `ml_math/CPP_vector.py` sizes its C buffers from one
argument and never validates the other, and this breaks two different ways. `matrix_vector_multiply`
builds `vec_arr` from `len(vec)` while `vector_math.cpp:25` loops to `cols = len(matrix[0])`, so an
undersized vector makes the C++ read past the allocation — I got
`[8.3214395e-317, 1.6642879e-316]` back, drifting on every call, straight out of unallocated memory
with no exception. That path is live from `predict()` on either regression model, since neither
checks `X`'s width against `len(self.weights)`. Separately, `dot`/`add`/`euclidean_distance` do
`size = len(a)` and then zero-pad a shorter `b`: `euclidean_distance([3,4,99],[0,0])` returns
99.126, so `KNN` answers a dimension-mismatched query with a confident label and stays quiet about
it. The reverse case throws `IndexError: invalid index` from inside ctypes, which tells the user
nothing. For a library whose entire pitch is "the math is correct and fast", silently wrong
distances are the worst possible failure mode.

Two smaller confirmed bugs: `utils.py:8` `sigmoid` raises `OverflowError` below about `x = -710`,
which logistic regression on unscaled features will hit; and the `TypeError` fallback at
`linear_regression.py:25-29` is dead code — it prints a reassuring recovery message and then crashes
four lines later with the same `TypeError`, which is strictly worse than having no handler.

The C++ itself is clean. All five kernels are correct; every problem is in the Python wrapping.

## What I am proposing

Ranked in `todo.md`. The quick win is deleting the `basicConfig` block — smallest possible change,
and it takes the project from "does not run" to "runs". Then dimension checks on the five wrappers,
then getting `test/` back under version control with a multi-feature fit-then-predict shape-mismatch
test as the first one written (that single test covers the out-of-bounds read *and* the area of the
last real code commit, `a75feb6` "fixed matrix math and added linear regression model fixes"). Then
the sigmoid fix and the linear-regression cleanup.

I dropped the old "fix requirements.txt" item (done) and demoted Naive Bayes, extra distance
metrics, the neural net, benchmarking and packaging to a Parked section — all still worth doing, but
not before the package imports. The pytest item folded into the test task. Noted in Parked that
`ml_math/` has no `__init__.py` and that `cpp/libvector.so` is a tracked binary that will silently
drift from its source.

## Verdict

Good bones, currently non-functional. The three-layer design is genuinely well chosen and the C++ is
correct, but the repo has been in scaffold-and-notes churn since March while the actual package sat
broken. A focused day — import fix, boundary validation, tests back in git — would make this a
project worth showing people. Right now a visitor cloning it cannot run a single line.
