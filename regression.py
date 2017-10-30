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
        xTx = np.dot(input.T, input)
        I = np.eye(len(xTx))
        self.theta = np.dot(np.dot(np.linalg.inv(xTx + self.alpha*I), input.T), output)

    # for scikit-learn
    def get_params(self, deep=True):
        return {'alpha':self.alpha}

    # (OPTION) the coefficient of determination R^2.
    # see sklearn.linear_model.Ridge().score()
    # http://goo.gl/v93tNM
    def score2(self, input, output):
        u = ((output - self.predict(input)) ** 2).sum()
        v = ((output - output.mean()) ** 2).sum()
        return (1 - u/v)
