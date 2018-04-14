# purpose: kerasによる花の画像を利用したCNNのテスト　学習編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 学習を効率行うためのcallbackについて追記している。
# created: 2018-03-20
import keras
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import sys
import os


data_format = "channels_last"   # 画像データの並び方は、[x, y, ch]みたいな感じを想定。（xとyは逆かも？）


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


def split(arr1, arr2, rate):
    """ 引数で受け取ったリストをrateの割合でランダムに抽出・分割する
    """
    if len(arr1) != len(arr2):
        return

    arr1_1, arr2_1 = list(arr1), list(arr2)  # popを使いたいので、listに変換
    arr1_2, arr2_2 = [], []                  # 抽出したものを格納する

    times = int(rate * len(arr1_1))
    for _ in range(times):
        i = np.random.randint(0, len(arr1_1))  # 乱数で抽出する要素番号を作成
        arr1_2.append(arr1_1.pop(i))
        arr2_2.append(arr2_1.pop(i))

    return np.array(arr1_1), np.array(arr2_1), np.array(arr1_2), np.array(arr2_2)

# 関数の動作テスト
"""
a = [1,2,3,4,5,6,7,8,9,10]
b = [11,12,13,14,15,16,17,18,19,20]
print(split(a, b, 0.2))
exit()
"""


def one_hotencoding(data=[]):
    """ one-hotencodingを行う
    data: list<ndarray>, 1次元のndarrayを格納したリスト
    """
    ans = []
    for mem in data:
        if len(mem) > 0:
            val = np_utils.to_categorical(mem)
            ans.append(val)
    return ans


def read_images(dir_names, data_format="channels_last", size=(32, 32), mode="RGB", resize_filter=Image.NEAREST, preprocess_func=None):
    """ リストで複数指定されたフォルダ内にある画像を読み込んで、リストとして返す
    通常は教師データの読み込みを想定している。

    dir_names: list<str>, フォルダ名又はフォルダへのパスを格納したリスト
    data_format: str, データ構造を指定
    size: tuple<int, int>, 読み込んだ画像のリサイズ後のサイズ
    mode: str, 読み込んだ後の画像の変換モード
    resize_filter: int, Image.NEARESTなど、リサイズに使うフィルターの種類。処理速度が早いやつは粗い
    preprocess_func: func, 前処理を行う関数
    """
    x, y = [], []       # 読み込んだ画像データと正解ラベル（整数）を格納する
    label_dict = {}     # 番号からフォルダ名を返す辞書
    weights = []        # 学習の重み

    for i in range(len(dir_names)):    # 貰ったフォルダ名の数だけループを回す
        name = dir_names[i]
        label_dict[i] = name           # 番号からフォルダ名を取得できる辞書を作成（予測段階で役に立つ）
        imgs = read_image(name, data_format=data_format, size=size, mode=mode, resize_filter=resize_filter)
        label = [i] * len(imgs)
        x = x + imgs
        y = y + label
        weights.append(len(imgs))

    # クラスごとの重みの計算と、重みの辞書の作成（教師データ数の偏りを是正する）
    weights = np.array(weights)
    weights = np.max(weights) / weights
    weights_dict = {i:weights[i] for i in range(len(weights))}

    # 画像の前処理
    if preprocess_func is not None:
        x = preprocess_func(x)

    return np.array(x), np.array(y), weights_dict, label_dict



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



def build_model(input_shape, output_dim, data_format):
    """ 機械学習のモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", data_format=data_format, input_shape=input_shape))  # カーネル数32, カーネルサイズ(3,3), input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d
    model.add(Activation('relu'))
    model.add(Conv2D(24, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), padding='same')) # sameを付けないと、サイズが小さくなる
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_dim))    # 出力層のユニット数はoutput_dim個
    model.add(Activation('sigmoid'))
    model.add(Activation('softmax'))
    opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 最適化器のセット。lrは学習係数
    model.compile(optimizer=opt,             # コンパイル
        loss='categorical_crossentropy',   # 損失関数は、判別問題なのでcategorical_crossentropyを使う
        metrics=['accuracy'])
    print(model.summary())
    return model


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



def save_validation_table(prediction, correct_data, label_dict):
    """ 学習に使わなかった検証データに対する予測と正解ラベルを使って、スレットスコアの表的なものを作って保存する
    prediction: ndarray, 1次元配列を想定。予測されたラベルが格納されている事を想定
    correct_data: ndrray, 1又は2次元配列を想定。正解ラベルが格納されている事を想定
    label_dict: pandas.DataFrame, 整数のkeyでラベルを取り出す辞書を想定
    """
    prediction = np.array([label_dict[x] for x in prediction]) # 予測されたクラスをラベルに変換
    correct_data = np.ravel(correct_data) # 1次元配列に変換
    #print(prediction, correct_data)

    # 行と列の名称のリストを作成
    keys = list(label_dict.keys())
    names = [label_dict[x] for x in range(min(keys), max(keys)+1)]
    #print(keys, names)

    # 結果を集計する
    df1 = pd.DataFrame(index=names, columns=names)  # 列名と行名がラベルのDataFrameを作成
    df1 = df1.fillna(0)               # とりあえず、0で全部埋める
    for i in range(len(prediction)):  # 行名と列名を指定しながら1を足す
        v1 = prediction[i]
        v2 = correct_data[i]
        #print("--v--", v1, v2)
        df1.loc[[v1],[v2]] += 1       # 行名と列名でセルを指定できる
    print("--件数でカウントした分割表--")
    print(df1)
    df1.to_csv("validation_table1.csv")

    # 正解ラベルを使って正規化する
    df2 = df1.copy()
    amount = [len(np.where(correct_data==x)[0]) for x in names] # 正解ラベルをカウント
    #amount = [len(np.where(prediction==x)[0]) for x in names] # 予測値をカウント
    print(amount)
    for i in range(len(df1)):
        df2.iloc[:,i] = df2.iloc[:,i] / amount[i] # 列単位で割る
        #df2.iloc[i,:] = df2.iloc[i,:] / amount[i] # 行単位で割る
    print("--割合で表した分割表--")
    print(df2)
    df2.to_csv("validation_table2.csv")



def restore(files):
    """ 保存されているファイルを読み込んでリストで返す
    files: list<str>, ファイル名がリストに格納されている事を想定
    """
    for fname in files:
        if os.path.exists(fname) == False:
            return

    ans = []
    for fname in files:
        if "npy" in fname:
            ans.append(np.load(fname))
        elif "pickle" in fname:
            with open(fname, 'rb') as f:
                ans.append(pickle.load(f))
    return ans


def load():
    """ 画像の読み込みと教師データの作成と保存を行う
    """
    # 画像を読み込む
    x, y, weights_dict, label_dict = read_images(['1_train', '2_train'], preprocess_func=preprocessing)
    x_train, y_train_o, x_test, y_test_o = split(x, y, 0.1)         # データを学習用と検証用に分割
    y_train, y_test = one_hotencoding(data=[y_train_o, y_test_o])   # 正解ラベルをone-hotencoding

    # 型の変換
    x_train, x_test = x_train.astype(np.float16), x_test.astype(np.float16)
    y_train_o, y_test_o = y_train_o.astype(np.int8), y_test_o.astype(np.int8)

    # 保存
    np.save('x_train.npy', x_train)
    np.save('y_train_o.npy', y_train_o)
    np.save('x_test.npy', x_test)
    np.save('y_test_o.npy', y_test_o)
    with open('weights_dict.pickle', 'wb') as f: # 再利用のために、ファイルに保存しておく
        pickle.dump(weights_dict, f)
    with open('label_dict.pickle', 'wb') as f: # 再利用のために、ファイルに保存しておく
        pickle.dump(label_dict, f)

    return x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test



def main():
    x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test = [None] * 8
    model = None

    # 教師データを無限に用意するオブジェクトを作成
    datagen = ImageDataGenerator(
        #samplewise_center = True,              # 平均をサンプル毎に0
        #samplewise_std_normalization = True,   # 標準偏差をサンプル毎に1に正規化
        #zca_whitening = True,                  # 計算に最も時間がかかる。普段はコメントアウトで良いかも
        rotation_range = 60,                    # 回転角度[degree]
        zoom_range=0.5,                         # 拡大縮小率
        fill_mode='nearest',                    # 引き伸ばしたときの外側の埋め方
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        rescale=1,                              # 
        width_shift_range=0.3,                  # 横方向のシフト率
        height_shift_range=0.3)                 # 縦方向のシフト率
    #datagen.fit(x_train)                        # zca用に、教師データの統計量を内部的に求める

    
    # 教師データの読み込みと、モデルの構築。必要なら、callbackで保存していた結合係数を読み込む
    load_flag = False
    if len(sys.argv) > 1 and sys.argv[1] == "retry":
        obj = restore(['x_train.npy', 'y_train_o.npy', 'x_test.npy', 'y_test_o.npy', 'weights_dict.pickle', 'label_dict.pickle'])
        if os.path.exists("cb_model.hdf5") and obj is not None:
            # 保存してたファイルからデータを復元
            x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict = obj
            y_train, y_test = one_hotencoding(data=[y_train_o, y_test_o])   # 正解ラベルをone-hotencoding

            # モデルを再構築
            print("--load 'cb_model.hdf5'--")
            model = load_model('cb_model.hdf5')
        else:
            print("--failure for restore--")
            load_flag = True
    else:
        load_flag = True

    if load_flag:  # 画像読み込みからモデルの構築までを実施
        x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test = load()
        model = build_model(input_shape=x_train.shape[1:], output_dim=len(label_dict), data_format=data_format)   # モデルの作成
    

    # 諸々を確認のために表示
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(weights_dict)
    print(label_dict)
    print(y_train, y_test_o)

    # 学習
    epochs = 30                 # 1つのデータ当たりの学習回数
    batch_size = 16             # 学習係数を更新するために使う教師データ数
    cb_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')  # 学習を適当なタイミングで止める仕掛け
    cb_save = keras.callbacks.ModelCheckpoint("cb_model.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5) # 学習中に

    history = model.fit_generator(   # ImageDataGeneratorを使った学習
        datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),  # シャッフルは順序によらない学習のために重要
        epochs=epochs,
        steps_per_epoch=x_train.shape[0],
        verbose=1,
        class_weight=weights_dict,
        callbacks=[cb_stop, cb_save],
        validation_data=(x_test, y_test)  # ここにジェネレータを渡すことも出来る
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）

    # 学習成果のチェックとして、検証データに対して分割表を作成する
    result = model.predict_classes(x_test, batch_size=batch_size, verbose=0) # クラス推定, 0-nの整数が得られる
    print("test result: ", result)
    correct_data = [label_dict[num] for num in y_test_o]  # ラベルに変換
    save_validation_table(result, correct_data, label_dict)

    # 学習結果を保存
    print(model.summary())      # レイヤー情報を表示(上で表示させると流れるので)
    model.save('model.hdf5')    # 獲得した結合係数を保存
    plot_history(history)       # lossの変化をグラフで表示



if __name__ == "__main__":
    main()


