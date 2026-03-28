from ml_math import CPP_vector
from ml_math.utils import sigmoid
from .base import BaseModel
import logging

class LogisticRegression(BaseModel):
    """
    Logistic Regression model for binary classification.
    Uses Gradient Descent for optimization.
    """
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__()
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0]) if n_samples > 0 else 0
        
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Vectorized linear prediction
            linear_model = CPP_vector.matrix_vector_multiply(X, self.weights)
            y_predicted = [sigmoid(lm + self.bias) for lm in linear_model]

            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                error = y_predicted[i] - y[i]
                for j in range(n_features):
                    dw[j] += X[i][j] * error
                db += error

            # Update parameters
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n_samples)
            self.bias -= self.lr * (db / n_samples)

    def predict_proba(self, X):
        """
        Returns the probability of the positive class.
        """
        linear_model = CPP_vector.matrix_vector_multiply(X, self.weights)
        return [sigmoid(lm + self.bias) for lm in linear_model]

    def predict(self, X):
        """
        Returns binary class labels (0 or 1).
        """
        probabilities = self.predict_proba(X)
        return [1 if p >= 0.5 else 0 for p in probabilities]
