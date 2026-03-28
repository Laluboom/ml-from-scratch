from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract Base Class for all machine learning models.
    Ensures a consistent API with fit and predict methods.
    """
    def __init__(self):
        self.weights = []
        self.bias = 0.0

    @abstractmethod
    def fit(self, X, y):
        """
        Train the model using features X and targets y.
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Make predictions for features X.
        """
        pass
