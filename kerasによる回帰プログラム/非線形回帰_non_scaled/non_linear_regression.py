#----------------------------------------
# purpose: kerasによる非線形回帰問題の学習テスト
# memo: fit()の引数のvalidation_dataを使えばfitを小刻みに行わなくても良いことに気がついたが、
#       fitを抜ける度に学習データにに何かするならそのまま使えるかもということで、このまま保存する。
# author: Katsuhiro MORISHITA　森下功啓
# created: 2017-07-20
#----------------------------------------
import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, advanced_activations, Activation
import matplotlib.pyplot as plt


def read_data(fname, ratio=0.8):
    """ データの読み込み
    ratio: float, 学習に使うデータの割合
    """
    df = pandas.read_csv(fname)
    df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True) # ランダムに並べ替える（効果高い）
    s = len(df.columns)
    x = (df.iloc[:, :-1]).values # ndarrayに変換
    y = (df.iloc[:, -1:]).values # 最後の列が正解データ
    print("x", x)
    print("y", y)
    p = int(ratio * len(df))
    x_train = x[:p] # 学習に使うデータ
    y_train = y[:p]
    x_test = x[p:] # 検証に使うデータ
    y_test = y[p:]
    return x_train, y_train, x_test, y_test, s

# データ読み込み
x_train, y_train, x_test, y_test, s = read_data("regression_learning.csv")

# 学習器の準備
model = Sequential()
model.add(Dense(15, input_shape=(s-1, ), use_bias=True)) # 定数項をbiasで表現する
model.add(advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dense(10))
model.add(advanced_activations.LeakyReLU(alpha=0.3))
model.add(Dense(1))
opt = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 学習係数を大きめに取っている
model.compile(optimizer=opt,
      loss='mean_squared_error',
      metrics=['mae'])

# 学習しつつ、lossの確認を行う
times = 20 # 学習ループを回す回数
epochs = 10 # 1回の学習ループでの、1つのデータ当たりの学習回数
batch_size = 50
loss_train = []
loss_test = []
epoch = 0
for i in range(times):
    # 学習
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1) # ephochsは誤差の収束状況を見て調整のこと

    # 学習状況を確認
    loss, accuracy = model.evaluate(x_train, y_train, batch_size=batch_size, verbose=0) # 学習データに対するlossを計算
    loss_train.append(loss)
    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size, verbose=0) # 検証データに対するlossを計算
    loss_test.append(loss)

# 学習結果を保存
print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json())
model.save_weights('param.hdf5')

# 検証
test = model.predict(x_test)
plt.scatter(y_test, test, c="b", marker="^") # グラフを描画する
plt.show()

# 誤差をグラフで保存
plt.clf()
plt.plot(np.arange(times) * epochs, loss_train, c="b", label="loss of train")
plt.plot(np.arange(times) * epochs, loss_test, c="r", label="loss of test")
plt.legend() # 凡例の表示
plt.xlabel("ephochs")
plt.ylabel("loss")
plt.savefig("loss.png")

