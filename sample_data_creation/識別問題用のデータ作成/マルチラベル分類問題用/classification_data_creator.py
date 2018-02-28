# purpose: 識別（分類）問題用のサンプルデータを自動的に作成する
# memo: 
# author: Katsuhiro Morishita
# created: 2017-08-15
# license: MIT
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns




class creator:
	""" 空間上にガウス分布する1つの集団を作る
	"""
	def __init__(self, name=None, dim=5, scale=10):
		"""
		val_num: int, 次元数
		scale: float, 変数の大きさの目安というか
		"""
		self._name = str(name)
		self._dim = dim
		self._scale = scale
		self._means = [random.uniform(0, scale) for _ in range(dim)]
		self._stds = [random.uniform(0.5, scale / 5) for _ in range(dim)]
		self._val_names = ["V{0}".format(i) for i in range(self._dim)]

	def create(self, n):
		"""
		n: int, 作成するレコード数
		"""
		df = pd.DataFrame()
		for i in range(self._dim):
			mean = self._means[i]
			std = self._stds[i]
			values = [np.random.normal(mean, std) for j in range(n)]
			df[self._val_names[i]] = values
		return df

	def save_param(self):
		_id = ""
		if self._name is None:
			_id = random.randrange(100000)
		else:
			_id = self._name

		df = pd.DataFrame([self._means], columns=self._val_names, index=[_id])
		print("means", df)
		df.to_csv("cof_means_{0}.csv".format(_id))

		df = pd.DataFrame([self._stds], columns=self._val_names, index=[_id])
		print("stds", df)
		df.to_csv("cof_stds_{0}.csv".format(_id))

	

def main():
	# クラスターを作るオブジェクトを作成
	cluster_size = 10 # クラスターの数（変数名悪い？
	dim = 5
	clusters = [creator("cluster_{0}".format(i), dim=dim) for i in range(cluster_size)]
	for mem in clusters:
		mem.save_param()

	# クラスターとラベルを紐付けるための正解を作る
	th = 0.1
	series_set = set()
	vol = [0] * cluster_size
	while 0 in vol: # 使われていないクラスターが無くなった時点でループを抜ける
		series = []
		for i in range(cluster_size):
			a = 0
			if np.random.rand() < th: # 一定の確率で1を出す
				a = 1
			series.append(a)
		if not(1 <= sum(series) <= 3): # あまりにも大きかったら
			continue
		series_set.add(tuple(series)) # タプル化した上でsetに入れることで同じパターンの多重登録を防止
		df = pd.DataFrame(list(series_set)) # クラスター毎の集計を行うためにDataFrameに入れる
		print(df)
		vol = [sum(df[k]) for k in range(len(df.columns))] # 使われないクラスターが出ないか確認するために、クラスター毎に使われた数を集計
	df = pd.DataFrame(list(series_set))
	correct = {}
	for i in range(cluster_size):
		correct[i] = tuple(df[i].values)
	print(correct) # ラベルのパターン数はlen(correct)


	# クラスターのデータを作りつつ、正解と紐付けながら教師データを作成
	df = pd.DataFrame()
	size = 300 # 1つのクラスターに属するデータ数
	class_names = ["class_{0}".format(k) for k in range(len(correct[0]))]
	for i in range(cluster_size):
		cluster = clusters[i]
		df1 = pd.DataFrame([correct[i]] * size, columns=class_names) # 正解を必要なレコード分だけ作る
		df2 = cluster.create(size) # クラスターに属するデータを作成
		df3 = pd.concat([df2, df1], axis=1) # 横に結合
		df = df.append(df3, ignore_index=True) # 全体に結合
		#df = pd.concat([df, df2])

	# 教師データを保存する
	df.to_csv("classification_data.csv", index=False)

	# グラフの作成（ちょっと重い）
	labels = [] # クラスター毎に色を付けたいので、クラスターごとのラベルを作成
	for i in range(cluster_size):
		labels += ["g{0}".format(i)] * size
	df4 = df.iloc[:,:dim]
	df4["group_label"] = labels
	sns.pairplot(df4, size=1.0, vars=list(df4.columns[:-1]), plot_kws={"s":5}, hue="group_label") # （層別もできる。hue="key"という引数を追加すれば)
	plt.savefig("graph.png") # 一応保存もしておく
	plt.show()



if __name__ == "__main__":
	main()
