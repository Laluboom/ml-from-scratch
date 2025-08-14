# Linear Regression -> y(predicted value) = w(weight) * x(input feature) + b(bias)
# Machine Learning: Minimize error between true y and calculated y by learning best weight bias pair
# Mean Squared Error Loss Function -> measure average squared diffrence of true y and calculated y. Prioritizes larger errors than multiple smaller ones therefore convinient for optimizations. Used to determine model accuracy for evaluating learning process and as a metric for comparision. 
# Gradient Descent Amendments -> Updates weights and bias for next iteration using previous MSE(mean square error) score. Weights: w'/b'(new weight or bias) = w/b(old weight or bias) - a(learning rate) * dL(partial derivate of loss function with respect to weight or bias)
# learning rate factors -> a = 0.01 to 0.1 for same order of magnitute scaling between features and low-dimensional dataset, Used when features are normalised and have not too many or too less datapoints for learning. Small datasets(100) prefer 0.001 to 0.0001 while larger(50,000) ones use 0.01 to 0.1. Always proportionally increase the learning rate to the batch size(samples of dataset taken before parameters are updated). monitor loss curve: divergent and unstable loss curve - decrease alpha, crawling and still loss curve - increase alpha

from ml_math import CPP_vector

class LinearRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = []
        self.bias = 0.0

    def fit(self, X, y):
        n_samples = len(X)
        n_features = len(X[0])
        
        self.weight = 0.0 * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            dw = [0.0] * n_features
            db = 0.0

            for i in range(n_samples):
                y_pred = CPP_vector.dot(self.weights, X[i]) + self.bias
                error = y[i] - y_pred

                for j in range(n_features):
                    dw[j] += -2 * X[i][j] * error
                db += -2 * error
            
            for j in range(n_features):
                self.weights[j] -= self.lr * (dw[j] / n_samples)
            self.bias -= self.lr * (db / n_samples)
            
    def predict(self,X):
        y_pred = []
        for x in X:
            y_hat = CPP_vector.dot(self.weights, x) + self.bias
            y_pred.append(y_hat)
        return y_pred
