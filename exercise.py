import numpy as np
import datasets
import regression

# 学習データの準備
X, Y = datasets.load_nonlinear_example1()
ex_X = datasets.polynomial3_features(X)

# モデル描画用のデータ（値域）を準備。
# 　学習したモデル自体は、入力を与えると予測値を出力する。
# 　このモデルを病が売るため、値域として 0〜4 を指定。
samples = np.arange(0, 4, 0.1)
x_samples = np.c_[ np.ones(len(samples)), samples ]
ex_x_samples = datasets.polynomial3_features(x_samples)

# モデル描画用のデータ（モデル出力値）を準備。
# 　alphaを変更する都度異なる予測を行うモデルを獲得できるので、
# 　毎回予測結果を predicted = [[alpha=0時の予測], [alhpa=0.1時の予測],,,] として保存する。
predicted = []
alphas = [0, 0.1, 0.5, 1.0, 10.0]
for alpha in alphas:
    model = regression.RidgeRegression(alpha=alpha)
    model.fit(ex_X, Y)
    predicted.append( model.predict(ex_x_samples) )

# グラフに凡例名を指定する際には、
# case 1) plot 時に label を指定する（plt.plot(x, y, label="hogehoge"）か、
# case 2) plot 時の object を保存しておき、legend時に指定する（plt.legend(obj, label)）か、
# の2通りのやり方がある。下記は case 1 での例。
import matplotlib.pyplot as plt
plt.scatter(X[:,1], Y)
lines = []
titles = []
for i in range(len(alphas)):
    label = "alpha = " + str(alphas[i])
    plt.plot(samples, predicted[i], label=label)

plt.legend(loc="lower right")
plt.show()

