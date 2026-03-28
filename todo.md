# Project Roadmap & TODO List

This list tracks the development of the "ML From Scratch" project, focusing on expanding model variety and strengthening the mathematical foundation.

## Phase 1: Mathematical Foundation & Optimization
- [x] **Add Matrix-Vector Multiplication**
- [x] **Implement Data Normalization Utilities**
- [x] **Vectorize Linear Regression Predictions**
- [x] **Automate C++ Compilation**

## Phase 2: Expanding Machine Learning Models
- [x] **Implement Logistic Regression**
- [x] **Implement K-Nearest Neighbors (KNN)**
  - **Explanation:** A simple, non-parametric model that classifies data based on how close it is to existing points. It relies heavily on distance calculations, which can be optimized in C++.
  - **Action:** Create `models/knn.py` and implement a distance-based classification logic.
- [ ] **Implement Naive Bayes Classifier**
  - **Explanation:** A probabilistic model based on Bayes' Theorem. It is very fast and works well for simple classification tasks like spam detection.
  - **Action:** Create `models/naive_bayes.py` to calculate feature probabilities.

## Phase 3: Architecture & Validation
- [x] **Create a Base Model Interface**
- [ ] **Expand Backend Unit Tests**
  - **Explanation:** As we add more C++ functions (like Matrix multiplication), we must ensure they are 100% accurate before using them in models.
  - **Action:** Add specific test cases in `test/` for every new C++ function implemented.
- [ ] **Automated Test Runner**
  - **Explanation:** Instead of running tests one by one, we need a way to check the whole project at once.
  - **Action:** Configure `pytest` or a simple shell script to run all files in the `test/` folder and report any failures.
