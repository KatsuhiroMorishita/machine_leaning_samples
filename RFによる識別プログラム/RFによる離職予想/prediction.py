#----------------------------------------
# purpose: ランダムフォレストによる弁別器テストスクリプト　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に正解ラベルが入っていること。
# created: 2017-07-05
#----------------------------------------
import pandas
import pickle
import numpy as np


# データの読み込み
data = pandas.read_csv("HR_comma_sep_test.csv")
#print(data)
trainFeature = (data.iloc[:, :-1]).values # transform to ndarray
trainLabel = (data.iloc[:, -1:]).values
trainLabel = np.ravel(trainLabel) # transform 2次元 to 1次元 ぽいこと

# 機械学習器を復元
clf = None
with open('entry.pickle', 'rb') as f:  # 学習成果を読み出す
    clf = pickle.load(f)               # オブジェクト復元

# テスト用のデータを保存
with open("test_result.csv", "w") as fw:
    test = clf.predict(trainFeature)
    test = [str(x) for x in test] # 要素を文字列に変更（ラベルが数値だった場合に有効）
    print(test)
    fw.write("{0}\n".format("\n".join(test)))
