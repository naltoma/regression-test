# regression-test

## Class design / How to use
```python
# from numpy as np
# X = np.array([[1,4],[1,8],[1,13],[1,17]])
# Y = np.array([7, 10, 11, 14])
>>> import datasets
>>> X, Y = datasets.load_linear_example1()
>>> import regression
>>> model = regression.LinearRegression()
>>> model.fit(X, Y)
>>> model.theta
array([ 5.30412371,  0.49484536])
>>> model.predict(X)
array([  7.28350515,   9.2628866 ,  11.7371134 ,  13.71649485])
>>> model.score(X, Y)  # RSS
1.2474226804123705
```

## See also
- [PDF](https://ie.u-ryukyu.ac.jp/~tnal/2017/info4/dm/2017info4dm-w4.pdf), pp.18-33.

<hr>

## Class design 2
```python
# from numpy as np
# X = np.array([[1,4],[1,8],[1,13],[1,17]])
# Y = np.array([4.0, 0.0, 3.0, 2.0])
>>> import datasets
>>> X, Y = datasets.load_nonlinear_example1()
>>> ex_X = datasets.polynominal3_features(X)
>>> import regression
>>> model = regression.RidgeRegression(alpha=0.5)
>>> model = regression.RidgeRegression() #default: alpha=0.1
>>> model.fit(ex_X, Y)
>>> model.theta
array([ 3.54259714, -1.24971967, -0.68925104,  0.23695052])
>>> model.predict(ex_X)
array([ 3.54259714,  0.1817578 ,  2.24085012,  2.68053522])
>>> model.score(ex_X, Y)  # RSS
1.2816900115950909
```

## See also
- [PDF](https://ie.u-ryukyu.ac.jp/~tnal/2017/info4/dm/2017info4dm-w5.pdf), pp.9-30.
