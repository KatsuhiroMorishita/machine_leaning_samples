#----------------------------------------
# purpose: ランダムフォレストによる弁別器テストスクリプト　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に正解ラベルが入っていること。
#       正解ラベルは整数もしくは文字列であること。
# created: 2017-07-05
#----------------------------------------
import pandas
import pickle
import numpy as np


# データの読み込み
data = pandas.read_csv("iris_test.csv")
#print(data)
s = len(data.columns) # 列数の取得
x = (data.iloc[:, :-1]).values # transform to ndarray

# 機械学習器を復元
clf = None
with open('entry.pickle', 'rb') as f:  # 学習成果を読み出す
    clf = pickle.load(f)               # オブジェクト復元

# 検証用のデータの予測と結果の保存
with open("test_result.csv", "w") as fw:
    test = clf.predict(x)
    test = [str(x) for x in test] # 要素を文字列に変更（ラベルが数値だった場合に有効）
    print(test)
    fw.write("{0}\n".format("\n".join(test)))

