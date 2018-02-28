# purpose: kerasによる弁別器テストスクリプト　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に層名（文字列でクラス名、または整数で0-nの連番）が入っていること。
# created: 2017-07-08
import pandas
import pickle
import numpy as np
from sklearn import preprocessing # 次元毎の正規化に使う
from keras.models import model_from_json


# データの読み込み
data = pandas.read_csv("iris_test.csv")
#print(data)
x = (data.iloc[:, :-1]).values # transform to ndarray
x = preprocessing.scale(x)     # 次元毎に正規化する

# 機械学習器を復元
model = model_from_json(open('model', 'r').read())
model.load_weights('param.hdf5')

# 出力をラベルに変換する辞書を復元
label_dict = None
with open('label_dict.pickle', 'rb') as f:
    label_dict = pickle.load(f)        # オブジェクト復元

# テスト用のデータを保存
with open("test_result.csv", "w") as fw:
    test = model.predict_classes(x)
    test = [str(label_dict[x]) for x in test] # 出力をラベルに変換（ラベルが文字列だった場合に便利だし、str.join(list)が使えて便利）
    print(test)
    fw.write("{0}\n".format("\n".join(test)))
