# ML From Scratch

A educational machine learning library built from the ground up, combining Python's high-level logic with a high-performance C++ mathematical backend.

## 🏗 Architecture Overview

The project is structured in three distinct layers to balance ease of use with computational efficiency:

1.  **C++ Backend (`cpp/`)**: High-performance implementations of core linear algebra and distance metrics using standard C++.
2.  **Math Bridge (`ml_math/`)**: Uses Python's `ctypes` library to load the compiled C++ shared library (`libvector.so`) and provide a seamless Python interface.
3.  **Model Layer (`models/`)**: High-level machine learning algorithms that inherit from a common `BaseModel` interface, utilizing the backend for all heavy computations.

## 🧮 Mathematical Foundation

### C++ Operations (Performance Critical)
*   **Dot Product**: Calculates $\sum (a_i \times b_i)$, used for linear combinations of weights and features.
*   **Matrix-Vector Multiplication**: Efficiently processes batch predictions by multiplying a matrix of samples by a weight vector.
*   **Euclidean Distance**: Calculates $\sqrt{\sum (a_i - b_i)^2}$, the core metric for similarity-based models like KNN.
*   **Vector Addition & Scalar Multiplication**: Fundamental operations used for updating model parameters during Gradient Descent.

### Python Utilities
*   **Sigmoid Function**: $S(x) = \frac{1}{1 + e^{-x}}$, maps any real value to a probability between 0 and 1.
*   **Data Scaling**: Includes Min-Max and Standard Scaling to ensure numerical stability and faster convergence during training.

## 🤖 Implemented Models

### 1. Linear Regression
Predicts a continuous target value $y$ based on input features $X$.
*   **Hypothesis**: $y = W \cdot X + b$
*   **Loss Function**: Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values.
*   **Optimization**: Gradient Descent, iteratively adjusting weights and bias to minimize the MSE.

### 2. Logistic Regression
A binary classification model that predicts the probability of an input belonging to a specific class.
*   **Hypothesis**: $y = \sigma(W \cdot X + b)$, where $\sigma$ is the Sigmoid function.
*   **Mechanism**: Uses Gradient Descent to optimize weights based on the probability error (Binary Cross-Entropy principles).

### 3. K-Nearest Neighbors (KNN)
A non-parametric, instance-based learning algorithm for classification.
*   **Mechanism**: Stores all training cases. For a new sample, it calculates the **Euclidean Distance** to all stored points, identifies the $K$ closest neighbors, and assigns the class through a majority vote.
*   **Performance**: The distance calculations are offloaded to C++ to handle larger datasets efficiently.

## 🚀 Getting Started

### Prerequisites
*   GCC (for C++ compilation)
*   Python 3.x

### Build and Test
The project includes a `Makefile` to automate the workflow:

```bash
# Compile the C++ shared library
make

# Run the complete test suite (includes backend and model validation)
make test

# Clean build artifacts
make clean
```

## 📁 Project Structure
*   `cpp/`: C++ source and compiled shared libraries.
*   `ml_math/`: The Python-C++ bridge and math utilities.
*   `models/`: Implementation of ML algorithms (`BaseModel`, `Linear`, `Logistic`, `KNN`).
*   `test/`: Unit tests and validation scripts.
*   `todo.md`: Project roadmap and future goals.
