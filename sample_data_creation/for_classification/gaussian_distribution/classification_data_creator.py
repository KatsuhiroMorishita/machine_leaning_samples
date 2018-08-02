# purpose: 識別（分類）問題用のサンプルデータを自動的に作成する
# author: Katsuhiro Morishita
# created: 2017-08-08
# license: MIT
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns


val_num = 5 # 独立変数の数
n = 300 # 各クラスのレコード数
scale = 10 # 変数の大きさの目安というか
cls_num = 3 # クラス数

# 各クラスの各次元における平均と標準偏差を決める。
means = []
stds = []
for _ in range(cls_num):
	means.append([random.uniform(0, scale) for _ in range(val_num)]) # 平均
	stds.append([random.uniform(0.5, scale / 5) for _ in range(val_num)]) # 標準偏差


def ndarray2str(val):
	""" ndarray型の変数を文字列に変換する
	val: ndarray 1次元配列を仮定
	"""
	val = [str(x) for x in val]
	return ",".join(val)


# 独立したデータを作る
val_names = ["V{0}".format(i) for i in range(val_num)]
df = pd.DataFrame()
for i in range(val_num):
	values = []
	for k in range(cls_num):
		mean = means[k][i]
		std = stds[k][i]
		values += [np.random.normal(mean, std) for j in range(n)]
	df[val_names[i]] = values

# 正解ラベルを作成・追加
label = []
for k in range(cls_num):
	label += ["class_{0}".format(k)] * n
df["class_label"] = label
print(df)


# 保存する
df.to_csv("classification_data.csv", index=False)


# パラメータも保存しておく
class_names = ["class_{0}".format(k) for k in range(cls_num)]
_means = np.array(means).T
df2 = pd.DataFrame(_means, columns=class_names, index=val_names)
print("means", df2)
df2.to_csv("cof_means.csv", index=False)

class_names = ["class_{0}".format(k) for k in range(cls_num)]
_stds = np.array(stds).T
df3 = pd.DataFrame(_stds, columns=class_names, index=val_names)
print("stds", df3)
df3.to_csv("cof_stds.csv", index=False)


# グラフの作成
sns.pairplot(df, size=1.0, vars=list(df.columns[:-1]), plot_kws={"s":5}, hue="class_label") # （層別もできる。hue="key"という引数を追加すれば)
plt.savefig("graph.png")
plt.show()
