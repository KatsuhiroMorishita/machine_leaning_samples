# purpose: kerasによる花の画像を利用したCNNのテスト　学習編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 
# created: 2018-02-17
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import np_utils
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import os


data_format = "channels_last"


def read_image(dir_name, data_format="channels_last", size=(32, 32), mode="RGB", resize_filter=Image.NEAREST):
    """ 指定されたフォルダ内の画像をリストとして返す
    dir_name: str, フォルダ名、又はフォルダへのパス
    data_format: str, データ構造を指定
    size: tuple<int, int>, 読み込んだ画像のリサイズ後のサイズ
    mode: str, 読み込んだ後の画像の変換モード
    resize_filter: int, Image.NEARESTなど、リサイズに使うフィルターの種類。処理速度が早いやつは粗い
    """
    image_list = []
    files = os.listdir(dir_name)     # ディレクトリ内部のファイル一覧を取得
    print(files)

    for file in files:
        root, ext = os.path.splitext(file)  # 拡張子を取得
        if ext != ".jpg":
            continue

        path = os.path.join(dir_name, file)             # ディレクトリ名とファイル名を結合して、パスを作成
        image = Image.open(path).resize(size, resample=resize_filter)   # 画像の読み込み
        image = image.resize(size, resample=resize_filter)              # 画像のリサイズ
        image = image.convert(mode)                     # 画像のモード変換。　mode=="LA":透過チャンネルも考慮して、グレースケール化, mode=="RGB":Aを無視
        image = np.array(image)                         # ndarray型の多次元配列に変換
        if mode == "RGB" and data_format == "channels_first":
            image = image.transpose(2, 0, 1)            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        image_list.append(image)                        # 出来上がった配列をimage_listに追加  
    
    return image_list


def preprocessing(imgs):
    """ 画像の前処理
    必要なら呼び出して下さい。
    
    imgs: ndarray, 画像が複数入っている多次元配列
    """
    image_list = []
    
    for img in imgs:
        img2 = (img - np.mean(img)) / np.std(img) / 4 + 0.5   # 平均0.5, 標準偏差を0.25にする
        img2[img2 > 1.0] = 1.0                 # 0-1からはみ出た部分が存在するとImageDataGeneratorに怒られるので、調整
        img2[img2 < 0.0] = 0.0
        
        #img2 = img / 255
        image_list.append(img2)
        
    image_list = np.array(image_list)     # ndarrayに変換
    return image_list


# 画像を読み込む
img1 = read_image('1_train')
img2 = read_image('2_train')
x = np.array(img1 + img2)  # リストを結合
x = preprocessing(x)       # 画像の前処理
y = np.array([0] * len(img1) + [1] * len(img2))  # 正解ラベルを作成
y = np_utils.to_categorical(y)                   # 正解ラベルをone-hot-encoding形式に変換
print(x.shape)
print(y)
#exit()


# モデルの作成
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", data_format=data_format, input_shape=x.shape[1:]))  # カーネル数32, カーネルサイズ(3,3), input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d
model.add(Activation('relu'))
model.add(Conv2D(24, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))                      # 出力層のユニット数は2
model.add(Activation('sigmoid'))
model.add(Activation('softmax'))
opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 最適化器のセット。lrは学習係数
model.compile(optimizer=opt,             # コンパイル
      loss='categorical_crossentropy',   # 損失関数は、判別問題なのでcategorical_crossentropyを使う
      metrics=['accuracy'])
print(model.summary())


# 学習
epochs = 10                 # 1つのデータ当たりの学習回数
batch_size = 8              # 学習係数を更新するために使う教師データ数
history = model.fit(x, y, 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1, 
    validation_split=0.1,
    #validation_data=(x_test, y_test), # validation_dataをセットするとvalidation_splitは無視される
    shuffle=True,           # 学習毎にデータをシャッフルする
    ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）


def plot_history(history):
    """ 損失の履歴を図示する
    from http://www.procrasist.com/entry/2017/01/07/154441
    """
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"^-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.grid()
    plt.yscale("log") # ケースバイケースでコメントアウト
    plt.show()


# 学習結果を保存
print(model.summary())                    # レイヤー情報を表示(上で表示させると流れるので)
open("model", "w").write(model.to_json()) # モデル情報の保存
model.save_weights('param.hdf5')          # 獲得した結合係数を保存
plot_history(history)                     # lossの変化をグラフで表示


