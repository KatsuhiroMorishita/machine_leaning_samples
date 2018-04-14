# purpose: kerasによるiris識別プログラム　学習編
# memo: 読み込むデータは、1行目に列名があり、最終列に正解となる層名（文字列でクラス名、または整数で0-nの連番）が入っていること。
# author: Katsuhiro MORISHITA　森下功啓
# created: 2017-07-08
import numpy as np
import pandas
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn import preprocessing # 次元毎の正規化に使う
from sklearn.feature_extraction import DictVectorizer # 判別問題における文字列による正解ラベルをベクトル化する



# データの読み込み
df = pandas.read_csv("iris_learning_str_label.csv")
df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True) # ランダムに並べ替える（効果高い）
s = len(df.columns)
x = (df.iloc[:, :-1]).values # transform to ndarray
x = preprocessing.scale(x) # 次元毎に正規化する
y = (df.iloc[:, -1:]).values
print("x", x)
print("y", y)


# 正解ラベルを01のリストを作成
labels = np.ravel(y) # 出力をラベルに変換するための布石
if y.dtype == "object":  # ラベルが文字列かチェック
    vec = DictVectorizer()
    y = vec.fit_transform([{"class":mem[0]} for mem in y]).toarray() # 判別問題における文字列による正解ラベルをベクトル化する
else:
    y = np_utils.to_categorical(y)
print("y", y)
label_dict = {list(y[i]).index(y[i].max()):labels[i] for i in range(len(labels))} # 出力をラベルに変換する辞書
with open('label_dict.pickle', 'wb') as f:
    pickle.dump(label_dict, f)

# 学習器の準備
model = Sequential()
model.add(Dense(2, input_shape=(s-1, ))) # 入力層は全結合層で入力がs-1次元のベクトルで、出力先のユニット数が2。
model.add(Activation('relu')) # 中間層の活性化関数がReLU
model.add(Dense(len(y[0]))) # 出力層のユニット数
model.add(Activation('relu')) # 中間層の活性化関数がReLU
model.add(Activation('softmax')) # 出力が合計すると1になる
model.compile(optimizer='adam',
      loss='categorical_crossentropy', # binary_crossentropy
      metrics=['accuracy'])

# 学習
batch_size = 5
model.fit(x, y, epochs=500, batch_size=batch_size, verbose=1) # nb_epochは古い引数の書き方なので、epochsを使う@2017-07

# 学習のチェック
result = model.predict_classes(x, batch_size=batch_size, verbose=0) # クラス推定
print("result1: ", result)
for mem in result:
    print("label convert test,", mem, label_dict[mem])
result = model.predict(x, batch_size=batch_size, verbose=0) # 各ユニットの出力を見る
print("result2: ", result)
result = model.predict_proba(x, batch_size=batch_size, verbose=0) # 確率を出す（softmaxを使っているので、predictと同じ出力になる）
print("result3: ", result)


# 学習結果を保存
print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json())
model.save_weights('param.hdf5')

