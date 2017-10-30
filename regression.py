import numpy as np

class LinearRegression:
    x = None
    theta = None
    y = None

    def fit(self, x, y):
        temp = np.linalg.inv(np.dot(x.T, x))
        self.theta = np.dot(np.dot(temp, x.T), y)

    def predict(self, x):
        return np.dot(x, self.theta)

    def score(self, x, y):
        error = self.predict(x) - y
        return (error**2).sum()

class RidgeRegression(LinearRegression):
    alpha = None

    def __init__(self, alpha=0.1):
        self.alpha = alpha

    def fit(self, input, output):
        pass

