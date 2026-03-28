from ml_math import CPP_vector
from .base import BaseModel
from collections import Counter

class KNN(BaseModel):
    """
    K-Nearest Neighbors classifier.
    Non-parametric model that classifies based on majority vote of neighbors.
    """
    def __init__(self, k=3):
        super().__init__()
        self.k = k
        self.X_train = []
        self.y_train = []

    def fit(self, X, y):
        """
        KNN 'learning' is just storing the training data.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict labels for a list of samples.
        """
        return [self._predict_single(x) for x in X]

    def _predict_single(self, x):
        # Calculate distances between x and all training samples
        distances = [CPP_vector.euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Get indices of k nearest samples
        k_indices = sorted(range(len(distances)), key=lambda i: distances[i])[:self.k]
        
        # Get labels of k nearest samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]
