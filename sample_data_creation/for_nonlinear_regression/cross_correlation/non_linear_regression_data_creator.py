# purpose: 非線形回帰分析用のサンプルデータを自動的に作成する
# author: Katsuhiro Morishita
# created: 2017-07-20
# license: MIT
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


w = 5 # 独立変数の数
wp = 0 # 非独立変数の数
n = 1000 # レコード数
scale = 10 # 変数の大きさの目安というか

# 独立変数の平均と標準偏差を決める。
means = [random.uniform(0, scale) for _ in range(w)] # 平均
stds = [random.uniform(1, scale / 10) for _ in range(w)] # 標準偏差


def ndarray2str(val):
	""" ndarray型の変数を文字列に変換する
	val: ndarray 1次元配列を仮定
	"""
	val = [str(x) for x in val]
	return ",".join(val)


# 独立したデータを作る
df = pd.DataFrame()
for i in range(w):
	val = np.random.randn(n)
	val = val * stds[i] + means[i]
	df[i] = val

# 独立していない変数も作る
for _ in range(wp):
	c = random.randint(0, len(df.columns)-1) # 引用元のインデックス
	r = 3 * random.uniform(-1, 1)            # 係数
	noise = 0.05 * random.uniform(0, scale) * np.random.randn(n) # ノイズ（説明変数毎にノイズの大きさを変える）
	df[len(df.columns)] = df[c] * r + noise  # dataframeに追加

# 目的変数を作る
cof = np.random.randn(len(df.columns)) # 係数を作成する
open("cof.csv", "w").write(ndarray2str(cof))
print("cof", cof)
exponents = [np.random.randint(1,4) for _ in range(len(df.columns))] # 1 to 3の指数を作成する
open("exp.csv", "w").write(ndarray2str(exponents))
print("exp", exponents)
y = np.array([0]*n)
for i in range(len(df.columns)):
	y = y + cof[i] * (df[i]**exponents[i])
noise = 0.005 * scale * np.random.randn(n) # ノイズ
df["noisy y"] = y + noise # noiseなしだと、cofがかなり正確に推定される
df.columns = ["V{0}".format(x+1) for x in range(len(df.columns))] # 列名を改名する

# 保存する
df.to_csv("non_linear_regression_data.csv", index=False)

# グラフの作成
sns.pairplot(df, size=1.0, vars=list(df.columns), plot_kws={"s":5}) # （層別もできる。hue="key"という引数を追加すれば)
plt.savefig("graph.png")
plt.show()

