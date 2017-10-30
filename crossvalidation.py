from sklearn import datasets
boston = datasets.load_boston()
x = boston.data
y = boston.target
half = int(len(x)/2)

import regression
model = regression.RidgeRegression(alpha=0.1)

# case 1: learn on the first half, test on the last half
model.fit(x[:half], y[:half])
score = model.score(x[half:], y[half:])

# case 2: learn on the last half, test on the first half
model.fit(x[half:], y[half:])
score += model.score(x[:half], y[:half])

print("RidgeRegression(alpha=0.1) score =", score)
#-> RidgeRegression(alpha=0.1) score = 78656.6246552
#-> RidgeRegression(alpha=1.0) score = 42334.1689238
