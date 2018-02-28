#----------------------------------------
# purpose: ランダムフォレストによる回帰のテストスクリプト　学習編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に正解（数値）が入っていること。
# created: 2017-08-01
#----------------------------------------
import pandas
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor as mc

# データの読み込み
data = pandas.read_csv("data.csv")
#print(data)
x = (data.iloc[:, :-1]).values # 最終列以外を取得
y = (data.iloc[:, -1:]).values # 最終列を取得
y = np.ravel(y) # transform 2次元 to 1次元 ぽいこと

# 学習
clf = mc()               # 学習器
clf.fit(x, y)
result = clf.score(x, y) # 学習データに対する、適合率

# 学習結果を保存
with open('entry.pickle', 'wb') as f:
	pickle.dump(clf, f)

# 1個だけテスト
test = clf.predict([x[0]])
print(test)

# 額数データに対する適合率
print(result)
print(clf.feature_importances_)	# 各特徴量に対する寄与度を求める