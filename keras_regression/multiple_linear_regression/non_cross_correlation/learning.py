#----------------------------------------
# purpose: kerasによる重回帰テストスクリプト　学習編
# このスクリプトは説明変数の各スケールに大きな違いがない場合に有効です。
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に解が入っていること。
# created: 2017-07-08
#----------------------------------------
import numpy as np
import pandas
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout


# データの読み込み
df = pandas.read_csv("regression_learning.csv")
df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True) # ランダムに並べ替える（効果高い）
s = len(df.columns)
x = (df.iloc[:, :-1]).values # ndarrayに変換
y = (df.iloc[:, -1:]).values
print("x", x)
print("y", y)

# 学習器の準備
model = Sequential()
model.add(Dense(1, input_shape=(s-1, ), use_bias=True)) # 定数項をbiasで表現する。活性化関数は指定しなければliner(f(x)=x)となる。
opt = keras.optimizers.Adam(lr=0.05, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 学習係数を大きめに取っている
model.compile(optimizer=opt,
      loss='mean_squared_error',
      metrics=['mae'])

# 学習
model.fit(x, y, epochs=2000, batch_size=200, verbose=1) # 次元数が大きければ、バッチサイズを大きくした方がいいかもしれない。ephochsは誤差の収束状況を見て調整のこと

# 学習結果を保存
print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json())
model.save_weights('param.hdf5')

# 結合係数を保存
weights = model.get_weights() # レイヤーを指定するには、model.layers[0].get_weights()の様に書く
print(weights)
open("weights.txt", "w").write(str(weights))
