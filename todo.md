# TODOs

1. **Implement Naive Bayes classifier** — create `models/naive_bayes.py` inheriting from `models/base.py`. It should follow the same `fit(X, y)` / `predict(X)` interface as `linear_regression.py` and `logistic_regression.py`. The C++ backend doesn't need changes — Naive Bayes relies on probability arithmetic that Python handles fine. Add a corresponding `test/test_naive_bayes.py` and wire it into the `Makefile` `test` target.

2. **Add automated test runner via pytest** — the `Makefile` currently calls each test script individually with `@python3 test/...`. This means a failure in one script halts the rest silently. Replace with `pytest test/` so all tests run, failures are reported together, and exit codes propagate correctly. Add `pytest` to `requirements.txt` once it exists.

---

## Future Ideas

3. **Add more C++ distance metrics** — `vector_math.cpp` currently implements Euclidean distance. Manhattan distance and cosine similarity are natural additions: Manhattan is useful for high-dimensional sparse data, cosine for text/embedding similarity. Both would make KNN more flexible and add new backend test cases to `test/test_math_vectors.py`.

4. **Neural network layer from scratch** — the architecture (C++ math → Python model) is well suited for a simple feedforward net. A single hidden layer with sigmoid activation could be built using the existing `dot_product` and `matrix_vector_multiply` functions. Would be a natural Phase 3 project after Naive Bayes.

5. **Benchmark C++ vs pure Python** — `test/validate_vectorization.py` exists but the timing output isn't captured anywhere. Add a proper benchmark script that runs the same operation at increasing array sizes (100, 1K, 10K, 100K elements) and plots C++ vs Python speed. Would make the "why C++?" motivation concrete and visible.

6. **Extend Linear Regression to multiple features** — the current implementation handles single-feature input (`X` is a 1D list). The C++ `matrix_vector_multiply` already supports arbitrary dimensions — update `linear_regression.py` to accept a 2D feature matrix and add a multi-feature test case.

7. **Package as an installable library** — add a `setup.py` or `pyproject.toml` so the project can be installed with `pip install -e .` and imported as `from ml_scratch.models import LinearRegression`. Would make it usable from other projects and is a good exercise in Python packaging.
