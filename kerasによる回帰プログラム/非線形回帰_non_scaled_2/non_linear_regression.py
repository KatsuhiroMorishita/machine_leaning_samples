#----------------------------------------
# purpose: kerasによる非線形回帰問題の学習テスト
# memo: この例では、validation_splitを使えばデータ読み込み時に分割する必要はないと思うのだが、
#       時系列データを使った学習をする際にはそれでは難があるのでそのままとした。
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

# 学習
epochs = 200 # 1つのデータ当たりの学習回数
batch_size = 50
history = model.fit(x_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1, 
    validation_split=0.1,
    validation_data=(x_test, y_test), # validation_dataをセットするとvalidation_splitは無視される
    shuffle=True,
    ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）

# 学習結果を保存
print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json())
model.save_weights('param.hdf5')

# 検証データに対する予測と正解の散布図を作成する
test = model.predict(x_test)
plt.scatter(y_test, test, c="b", marker="^") 
plt.show()

# 学習のlossの変化をplot
def plot_history(history):
    """ 損失の履歴をプロット
    from http://www.procrasist.com/entry/2017/01/07/154441
    """
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"^-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.grid(which="both")
    plt.yscale("log") # ケースバイケースでコメントアウト
    plt.show()

#print(history.history)
plot_history(history)
