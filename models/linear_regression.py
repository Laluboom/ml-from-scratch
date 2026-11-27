# Linear Regression -> y(predicted value) = w(weight) * x(input feature) + b(bias)
# Machine Learning: Minimize error between true y and calculated y by learning best weight bias pair
# Mean Squared Error Loss Function -> measure average squared diffrence of true y and calculated y. Prioritizes larger errors than multiple smaller ones therefore convinient for optimizations. Used to determine model accuracy for evaluating learning process and as a metric for comparision. 
# Gradient Descent Amendments -> Updates weights and bias for next iteration using previous MSE(mean square error) score. Weights: w'/b'(new weight or bias) = w/b(old weight or bias) - a(learning rate) * dL(partial derivate of loss function with respect to weight or bias)
# learning rate factors -> a = 0.01 to 0.1 for same order of magnitute scaling between features and low-dimensional dataset, Used when features are normalised and have not too many or too less datapoints for learning. Small datasets(100) prefer 0.001 to 0.0001 while larger(50,000) ones use 0.01 to 0.1. Always proportionally increase the learning rate to the batch size(samples of dataset taken before parameters are updated). monitor loss curve: divergent and unstable loss curve - decrease alpha, crawling and still loss curve - increase alpha
import math

from ml_math import CPP_vector
from ml_math.utils import mse
from .base import BaseModel
import logging

logger = logging.getLogger(__name__)

class LinearRegression(BaseModel):
    def __init__(self, learning_rate=0.01, epochs=1000):
        super().__init__()
        self.lr = learning_rate
        self.epochs = epochs

    def fit(self, X, y):
        n_samples = len(X)
        try:        
            n_features = len(X[0])
        except TypeError as e:
            n_features = 1
            print("Recieved TypeError due to python not being able to recognize all features in input value X using len(X[0]). Assumed 1 feature for the model prediction")
        
        self.weights = [0.0] * n_features
        self.bias = 0.0
        self.loss_history = []

        for epoch in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0

            # Vectorized prediction for all samples
            y_preds = CPP_vector.matrix_vector_multiply(X, self.weights)
            y_preds = [yp + self.bias for yp in y_preds]

            loss = mse(y, y_preds)
            if not math.isfinite(loss):
                raise ValueError(
                    f"Loss diverged to {loss} at epoch {epoch}. "
                    "Try a lower learning rate or scale the input features."
                )
            self.loss_history.append(loss)
            logger.debug(f"Epoch {epoch}: loss={loss}")

            for i in range(n_samples):
                error = y[i] - y_preds[i]

                for j in range(n_features):
                    dw[j] += -2 * X[i][j] * error
                db += -2 * error

            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n_samples)
            self.bias -= self.lr * (db / n_samples)
            
    def predict(self,X):
        y_preds = CPP_vector.matrix_vector_multiply(X, self.weights)
        return [y_hat + self.bias for y_hat in y_preds]
