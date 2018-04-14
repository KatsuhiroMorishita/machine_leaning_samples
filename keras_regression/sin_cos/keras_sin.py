#----------------------------------------
# purpose: kerasによるsinの学習のテスト
# author: Katsuhiro MORISHITA　森下功啓
# created: 2017-07-10
#----------------------------------------
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, advanced_activations, Activation
import matplotlib.pyplot as plt


# 教師データの作成
x = np.arange(0, 2*np.pi, np.pi/10)
y = np.sin(x)
#y = 5*x*x*x # 一応、こういう関数でもOK 5x^3
print("x", x)
print("y", y)

# 学習器の準備
model = Sequential()
model.add(Dense(15, input_shape=(1, ), use_bias=True)) # 定数項をbiasで表現する
#model.add(advanced_activations.LeakyReLU(alpha=0.3))
model.add(Activation("sigmoid")) # LeakyReLUよりも良さげ
model.add(Dense(1))
opt = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 学習係数を大きめに取っている
model.compile(optimizer=opt,
      loss='mean_squared_error',
      metrics=['mae'])

# 学習
model.fit(x, y, epochs=2000, batch_size=10, verbose=1) # ephochsは誤差の収束状況を見て調整のこと

# 学習結果を保存
print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json())
model.save_weights('param.hdf5')

# 検証
test = model.predict(x)
plt.plot(x, test, c="b") # グラフを描画する
plt.plot(x, y, c="r") # グラフを描画する
plt.show()
