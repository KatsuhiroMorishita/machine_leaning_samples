# purpose: 活性化関数mishの実装テスト
# author: Katsuhiro MORISHITA　森下功啓
# created: 2020-02-22
# lisence: MIT. If you use this program in your study, you should write shaji in your paper.
import keras
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import signal
import sys
import os

# GPUを使わないようにする設定（不要ならコメントアウト）
import tensorflow as tf
import keras.backend as K
K.tensorflow_backend.set_session(tf.Session(config=tf.ConfigProto(device_count = {'GPU': 0})))

# 標準外ライブラリ
import image_preprocessing as ip
import mish_keras



def build_model_simple(input_shape, output_dim, data_format):
    """ 転移学習を利用しないモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    """
    # モデルの作成

    mish = mish_keras.Mish()  # 活性化関数mishのオブジェクトを作成

    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", data_format=data_format, input_shape=input_shape))  # カーネル数32, カーネルサイズ(3,3), input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d
    model.add(Activation(mish))
    model.add(Conv2D(24, (3, 3)))
    model.add(Activation(mish))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation(mish))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation(mish))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation(mish))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))                      # 出力層のユニット数は2
    model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))
    opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 最適化器のセット。lrは学習係数
    model.compile(optimizer=opt,             # コンパイル
          loss='categorical_crossentropy',   # 損失関数は、判別問題なのでcategorical_crossentropyを使う
          metrics=['accuracy'])
    print(model.summary())

    return model


def plot_history(history):
    """ 損失の履歴を図示する
    from http://www.procrasist.com/entry/2017/01/07/154441
    """
    x = history.epoch
    y1 = history.history['loss']
    y2 = history.history['val_loss']
    plt.plot(x, y1, "o-", label="loss")
    plt.plot(x, y2, "^-", label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.grid()
    plt.yscale("log") # ケースバイケースでコメントアウト
    plt.show()




# 中断対策
def handler(signal, frame):
    print('exit: ctrl+c or ctrl+z')
    tensorflow_backend.clear_session()


def main():
    # 調整することの多いパラメータを集めた
    image_shape = (32, 32)   # 画像サイズ
    epochs = 5               # 1つのデータ当たりの学習回数
    batch_size = 10          # 学習係数を更新するために使う教師データ数
    
    # 教師データの読み込みと、モデルの構築
    data_format = "channels_last"
    dir_names_dict = {"yellow":["sample_image_flower/1_train"], 
                      "white":["sample_image_flower/2_train"]} 
    param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":image_shape, "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":ip.preprocessing2}
    x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim, test_file_names = ip.load_save_images(ip.read_images1, param, validation_rate=0.2)
    model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成

    # 諸々を確認のために表示
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(weights_dict)
    print(label_dict)
    print(y_train, y_test_o)

    # 中断対策
    signal.signal(signal.SIGINT, handler)

    # 学習
    # callbackもGeneratorを使わないパターン
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=1,
        class_weight=weights_dict,
        validation_data=(x_test, y_test),
        shuffle=True,
        batch_size=batch_size
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）


    # 学習結果を保存
    print(model.summary())      # レイヤー情報を表示(上で表示させると流れるので)
    model.save('model.hdf5')    # 獲得した結合係数を保存
    plot_history(history)       # lossの変化をグラフで表示


if __name__ == "__main__":
    main()


