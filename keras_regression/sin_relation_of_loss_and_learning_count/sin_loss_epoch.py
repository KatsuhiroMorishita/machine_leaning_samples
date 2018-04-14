#----------------------------------------
# purpose: kerasによるsinの学習を通して、epochやlossの変化を観察する図を作成する（説明用の図をつくる）
# author: Katsuhiro MORISHITA　森下功啓
# created: 2017-07-10
#----------------------------------------
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, advanced_activations, Activation
import matplotlib.pyplot as plt


# データの読み込み
x = np.arange(0, 2*np.pi, np.pi/10)
y = np.sin(x)
#y = 5*x*x*x
print("x", x)
print("y", y)

# 学習器の準備
model = Sequential()
model.add(Dense(15, input_shape=(1, ), use_bias=True)) # 定数項をbiasで表現する
model.add(Activation("sigmoid"))
model.add(Dense(1))
opt = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 学習係数を大きめに取っている
model.compile(optimizer=opt,
      loss='mean_squared_error',
      metrics=['mae']) # mean_absolute_percentage_error 


times = 10
epochs = 120
loss_array = []
mae_array = []
for i in range(times):
    # 学習
    model.fit(x, y, epochs=epochs, batch_size=10, verbose=1) # ephochsは誤差の収束状況を見て調整のこと

    # 学習結果を保存
    print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
    open("model_{0}".format(i), "w").write(model.to_json())
    model.save_weights('param_{0}.hdf5'.format(i))

    # 学習状況を確認
    loss, mae = model.evaluate(x, y, verbose=0)
    loss_array.append(loss)
    mae_array.append(mae)

    # 学習状況の検証のためのグラフの作成と保存
    test = model.predict(x)
    plt.clf()
    plt.plot(x, test, c="b")
    plt.plot(x, y, c="r")
    plt.savefig("graph_{0}.png".format(i))

# 誤差と精度をグラフで保存
plt.clf()
plt.plot(np.arange(times) * epochs, loss_array, c="b", label="loss")
plt.plot(np.arange(times) * epochs, mae_array, c="r", label="mae")
plt.legend() # 凡例の表示
plt.xlabel("ephochs")
plt.ylabel("loss or mae")
plt.savefig("loss_mae.png")
