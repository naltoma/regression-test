from sklearn import datasets
boston = datasets.load_boston()
x = boston.data
y = boston.target

import numpy as np
ones = np.ones((len(x),1))
ex_x = np.c_[ones, x]

import regression
alpha = 0.1
model = regression.RidgeRegression(alpha=alpha)

#from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, ex_x, y, cv=10, n_jobs=-1)
print("*** Ridge(alpha=%0.2f) ***" % alpha)
print("scores=", scores)
print("mean score = %f (+/- %0.2f)" % (scores.mean(), scores.std()*2))
