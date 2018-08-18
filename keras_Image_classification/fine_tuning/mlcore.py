# purpose: kerasによる画像識別の関数をまとめた
# main()に利用方法のサンプルを置いているので、参考にしてください。
# author: Katsuhiro MORISHITA　森下功啓
# created: 2018-08-12
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import sys
import os






def read_name_dict(fname, skiprows=[], key_valule=[0, 1], delimiter=","):
    """ ファイル名とクラスIDなどが入ったファイルから、特定の列を辞書に加工して返す
    fname: str, ファイル名
    skiprows: list<int>, 読み飛ばす行番号を格納したリスト
    key_valule: list<int>, keyとvalueのそれぞれの列番号を格納したリスト
    """
    df = pd.read_csv(fname, delimiter, header=None, skiprows=skiprows)
    
    name_dict = {}
    for i in range(len(df)):
        name_dict[df.iloc[i, key_valule[0]]] = df.iloc[i, key_valule[1]]
    return name_dict



def read_image(param):
    """ 指定されたフォルダ内の画像をリストとして返す
    
    param: dict, 計算に必要なパラメータを収めた辞書
    """
    dir_name = param["dir_name"]            # dir_name: str, フォルダ名、又はフォルダへのパス
    data_format = param["data_format"]      # data_format: str, データ構造を指定
    size = param["size"]                    # size: tuple<int, int>, 読み込んだ画像のリサイズ後のサイズ。 width, heightの順。
    mode = param["mode"]                    # mode: str, 読み込んだ後の画像の変換モード
    resize_filter = param["resize_filter"]  # resize_filter: int, Image.NEARESTなど、リサイズに使うフィルターの種類。処理速度が早いやつは粗い
    image_list = []                  # 読み込んだ画像データを格納するlist
    name_list = []                   # 読み込んだファイルのファイル名
    files = os.listdir(dir_name)     # ディレクトリ内部のファイル一覧を取得
    print("--files--", files[:20])

    for file in files:
        root, ext = os.path.splitext(file)  # 拡張子を取得
        if ext != ".jpg" and ext != ".bmp" and ext != ".png":
            continue

        path = os.path.join(dir_name, file)             # ディレクトリ名とファイル名を結合して、パスを作成
        image = Image.open(path).resize(size, resample=resize_filter)   # 画像の読み込み
        image = image.resize(size, resample=resize_filter)              # 画像のリサイズ
        image = image.convert(mode)                     # 画像のモード変換。　mode=="LA":透過チャンネルも考慮して、グレースケール化, mode=="RGB":Aを無視
        image = np.array(image)                         # ndarray型の多次元配列に変換
        image = image.astype(np.float16)                # 型の変換（整数に変換すると、0-1にスケーリングシた際に、0や1になるのでfloatに変換）
        if mode == "RGB" and data_format == "channels_first":
            image = image.transpose(2, 0, 1)            # 配列を変換し、[[Redの配列],[Greenの配列],[Blueの配列]] のような形にする。
        image_list.append(image)                        # 出来上がった配列をimage_listに追加  
        name_list.append(file)
    
    return image_list, name_list


def split(arr1, arr2, rate):
    """ 引数で受け取ったlistをrateの割合でランダムに抽出・分割する（副作用に注意）
    arr1, arr2: list<ndarray or list or int>, 画像や整数が入ったリストを期待している
    """
    if len(arr1) != len(arr2):
        raise ValueError("length of arr1 and arr2 is not equal.")

    arr1_1, arr2_1 = arr1, arr2  # arr1, arr2を代入すると、副作用覚悟で使用メモリを少し減らす。副作用が嫌いなら、→　list(arr1), list(arr2)　を代入
    arr1_2, arr2_2 = [], []       # 抽出したものを格納する

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
    （2018-08-12: クラスの数の割にクラス毎のサンプル数が少ないことが原因でdata内の各要素におけるクラスIDの欠落が生じないように、ロジックを書き換えた）
    data: list<ndarray>, 1次元のndarrayを格納したリスト
    """
    fusion = []   # 一旦、全部結合させる
    for mem in data:
        fusion += list(mem)
    fusion_onehot = np_utils.to_categorical(fusion)  # 全部を一緒にしてからone-hotencoding

    ans = []    # fusion_onehotを個々に切り出す
    s = 0
    for length in [len(mem) for mem in data]:
        ans.append(fusion_onehot[s:s + length])
        s += length
    return ans




def read_images1(param):
    """ 辞書で指定されたフォルダ内にある画像を読み込んで、リストとして返す
    クラス毎にフォルダ名又はフォルダへのパスを格納した辞書が、param内に"dir_names_dict"をキーとして保存されていることを期待している。
    フォルダ名がそのままクラス名でも、この関数で処理すること。
    param: dict, 計算に必要なパラメータを収めた辞書
    """
    dir_names_dict = param["dir_names_dict"]     # dict<str:list<str>>, クラス毎にフォルダ名又はフォルダへのパスを格納した辞書。例：{"A":["dir_A1", "dir_A2"], "B":["dir_B"]}
    x, y = [], []    # 読み込んだ画像データと正解ラベル（整数）を格納するリスト
    file_names = []  # 読み込んだ画像のファイル名のリスト
    size_dict = {}   # データの数をクラス毎に格納する辞書

    class_name_list = sorted(dir_names_dict.keys())  # この時点ではstr。ソートすることで、local_id（プログラム中で割り振るクラス番号）とクラス名がずれない
    label_dict = {i:class_name_list[i]  for i in range(len(class_name_list))}   # local_idからクラス名を引くための辞書。ここでのlocal_idはこの学習内で通用するローカルなID。（予測段階で役に立つ）
    label_dict_inv = {class_name_list[i]:i  for i in range(len(class_name_list))}   # 逆に、クラス名から番号を引く辞書
    output_dim = len(label_dict)        # 出力層に必要なユニット数（出力の次元数）
    label_dict[len(label_dict)] = "ND"  # 分類不能に備えて、NDを追加


    for class_name in class_name_list:    # 貰った辞書内のクラス数だけループを回す
        for dir_name in dir_names_dict[class_name]:   # クラス毎に、フォルダ名が格納されたリストから1つずつフォルダ名を取り出してループ
            param["dir_name"] = dir_name
            imgs, _file_names = read_image(param)    # file_namesは使わない
            if len(imgs) == 0:
                continue

            local_id = label_dict_inv[class_name]   # local_idはint型
            label_local = [local_id] * len(imgs)    # フォルダ内の画像は全て同じクラスに属するものとして処理
            x += imgs
            y += label_local
            file_names += _file_names
            if local_id in size_dict:    # クラス毎にその数をカウント
                size_dict[local_id] += len(imgs)
            else:
                size_dict[local_id] = len(imgs)

    # クラスごとの重みの計算と、重みの辞書の作成（教師データ数の偏りを是正する）
    size_keys = sorted(size_dict.keys())
    size_list = [size_dict[k] for k in size_keys]
    print("size_dict: ", size_dict)
    print("size list: ", size_list)
    weights = np.array(size_list)
    weights = np.max(weights) / weights
    weights_dict = {i:weights[i] for i in size_keys}

    return x, y, weights_dict, label_dict, output_dim, file_names



def read_images2(param):
    """ リストで指定されたフォルダ内にある画像を読み込んで、リストとして返す
    ファイル名とクラス名（整数か文字列）を紐づけた辞書が、param内に"name_dict"をキーとして保存されていることを期待している。
    param: dict, 計算に必要なパラメータを収めた辞書
    """
    dir_names_list = param["dir_names_list"]    # list<str>, フォルダ名又はフォルダへのパスを格納したリスト
    name_dict = param["name_dict"]              # dict<key: str, value: int or str>, ファイル名をクラスIDに変換する辞書
    x, y = [], []    # 読み込んだ画像データと正解ラベル（整数）を格納するリスト
    file_names = []
    size_dict = {}   # データの数をクラス毎に格納する辞書

    class_name_list = sorted(list(set(name_dict.values())))      # この時点ではintかstrのどちらか。ソートすることで、local_id（プログラム中で割り振るクラス番号）とクラス名がずれない
    label_dict = {i:class_name_list[i]  for i in range(len(class_name_list))}   # local_idからクラス名を引くための辞書。ここでのlocal_idはこの学習内で通用するローカルなID。（予測段階で役に立つ）
    label_dict_inv = {class_name_list[i]:i  for i in range(len(class_name_list))}   # 逆に、クラス名から番号を引く辞書
    output_dim = len(label_dict)        # 出力層に必要なユニット数（出力の次元数）
    label_dict[len(label_dict)] = "ND"  # 分類不能に備えて、NDを追加

    for dir_name in dir_names_list:    # 貰ったフォルダ名の数だけループを回す
        param["dir_name"] = dir_name
        imgs, _file_names = read_image(param)
        if len(imgs) == 0:
            continue

        label_raw = [name_dict[name]  for name in _file_names]         # ファイル名からラベルのリスト（クラス名のlist）を作る
        label_local = [label_dict_inv[raw_id] for raw_id in label_raw]  # 学習に使うlocal_idに置換
        print("--label--", label_local[:20])
        x += imgs
        y += label_local
        file_names += _file_names
        for local_id in label_local:    # クラス毎にその数をカウント
            if local_id in size_dict:
                size_dict[local_id] += 1
            else:
                size_dict[local_id] = 1

    # クラスごとの重みの計算と、重みの辞書の作成（教師データ数の偏りを是正する）
    size_keys = sorted(size_dict.keys())
    size_list = [size_dict[k] for k in size_keys]
    print("size_dict: ", size_dict)
    print("size list: ", size_list)
    weights = np.array(size_list)
    weights = np.max(weights) / weights
    weights_dict = {i:weights[i] for i in size_keys}

    return x, y, weights_dict, label_dict, output_dim, file_names




def preprocessing(imgs):
    """ 画像の前処理
    必要なら呼び出して下さい。
    （処理時間が長い・・・）
    
    imgs: ndarray or list<ndarray>, 画像が複数入っている多次元配列
    """
    image_list = []
    
    for img in imgs:
        _img = img.astype(np.float32)    # float16のままではnp.mean()がオーバーフローする
        img2 = (_img - np.mean(_img)) / np.std(_img) / 4 + 0.5   # 平均0.5, 標準偏差を0.25にする
        img2[img2 > 1.0] = 1.0                 # 0-1からはみ出た部分が存在するとImageDataGeneratorに怒られるので、調整
        img2[img2 < 0.0] = 0.0
        img2 = img2.astype(np.float16)
        image_list.append(img2)
        
    return np.array(image_list)



def build_model(input_shape, output_dim, data_format):
    """ 機械学習のモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    """
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    #base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)

    top_model = Sequential()
    top_model.add(Conv2D(32, (3, 3), padding="same", input_shape=base_model.output_shape[1:]))
    #top_model.add(Activation('relu'))
    top_model.add(Conv2D(32, (3, 3), padding="same"))
    #top_model.add(Activation('relu'))
    top_model.add(Conv2D(32, (3, 3), padding="same"))
    #top_model.add(Conv2D(32, (3, 3), padding="same"))
    #top_model.add(Activation('relu'))
    top_model.add(MaxPooling2D(pool_size=(3, 3)))
    top_model.add(Flatten())
    top_model.add(Dense(100, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(output_dim))    # 出力層のユニット数はoutput_dim個
    top_model.add(Activation('sigmoid'))
    top_model.add(Activation('softmax'))

    # fix weights of base_model
    for layer in base_model.layers:
        layer.trainable = False    # Falseで更新しない

    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9),    # コンパイル
        loss='categorical_crossentropy',   # 損失関数は、判別問題なのでcategorical_crossentropyを使う
        metrics=['accuracy'])
    print(model.summary())
    return model



def build_model_simple(input_shape, output_dim, data_format):
    """ 機械学習のモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    """
    # モデルの作成
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding="same", data_format=data_format, input_shape=input_shape))  # カーネル数32, カーネルサイズ(3,3), input_shapeは1層目なので必要。https://keras.io/ja/layers/convolutional/#conv2d
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
    model.add(Dense(output_dim))                      # 出力層のユニット数は2
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



def save_validation_table(predicted_classse, correct_classse, label_dict):
    """ 学習に使わなかった検証データに対する予測と正解ラベルを使って、スレットスコアの表的なものを作って保存する
    predicted_classse: list or ndarray, 1次元配列を想定。予測されたラベルが格納されている事を想定
    correct_classse: list or ndrray, 1又は2次元配列を想定。正解ラベルが格納されている事を想定
    label_dict: pandas.DataFrame, 整数のkeyでラベルを取り出す辞書を想定
    """
    correct_classse = np.ravel(correct_classse) # 1次元配列に変換
    #print(predicted_classse, correct_classse)

    # 行と列の名称のリストを作成
    keys = list(label_dict.keys())
    keys = sorted(keys)
    names = [label_dict[x] for x in range(min(keys), max(keys)+1)]
    #print(keys, names)

    # 結果を集計する
    df1 = pd.DataFrame(index=names, columns=names)  # 列名と行名がラベルのDataFrameを作成
    df1 = df1.fillna(0)               # とりあえず、0で全部埋める
    for i in range(len(predicted_classse)):  # 行名と列名を指定しながら1を足す
        v1 = predicted_classse[i]
        v2 = correct_classse[i]
        #print("--v--", v1, v2)
        df1.loc[[v1],[v2]] += 1       # 行名と列名でセルを指定できる
    print("--件数でカウントした分割表--")
    print(df1)
    df1.to_csv("validation_table1.csv")

    # 正解ラベルを使って正規化する
    df2 = df1.copy()
    amount = [len(np.where(correct_classse==x)[0]) for x in names] # 正解ラベルをカウント
    #amount = [len(np.where(predicted_classse==x)[0]) for x in names] # 予測値をカウント
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


def load(read_func, param, validation_rate=0.1):
    """ 画像の読み込みと教師データの作成と保存を行う
    read_func: function, 画像を読み込む画像
    param: dict<str: obj>, read_funcに渡すパラメータ 
    validation_rate: float, 検証に使うデータの割合
    """
    # 画像を読み込む
    x, y, weights_dict, label_dict, output_dim, file_names = read_func(param)
    x_train, y_train_o, x_test, y_test_o = split(x, y, validation_rate)  # データを学習用と検証用に分割

    if "preprocess_func" in param:   # 必要なら前処理
        preprocess_func = param["preprocess_func"]  # func, 前処理を行う関数
        x_train = preprocess_func(x_train)
        x_test = preprocess_func(x_test)
    y_train, y_test = one_hotencoding(data=[y_train_o, y_test_o])   # 正解ラベルをone-hotencoding。分割表を作りたいので、splitの後でone-hotencodingを行う

    # 保存
    np.save('x_train.npy', x_train)
    np.save('y_train_o.npy', y_train_o)
    np.save('x_test.npy', x_test)
    np.save('y_test_o.npy', y_test_o)
    with open('weights_dict.pickle', 'wb') as f:  # 再利用のために、ファイルに保存しておく
        pickle.dump(weights_dict, f)
    with open('label_dict.pickle', 'wb') as f:  # 再利用のために、ファイルに保存しておく
        pickle.dump(label_dict, f)
    with open('param.pickle', 'wb') as f:  # 再利用のために、ファイルに保存しておく
        pickle.dump(param, f)

    return x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim



def main():
    x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = [None] * 9
    model = None

    # 教師データを無限に用意するオブジェクトを作成
    datagen = ImageDataGenerator(
        #samplewise_center = True,              # 平均をサンプル毎に0
        #samplewise_std_normalization = True,   # 標準偏差をサンプル毎に1に正規化
        #zca_whitening = True,                  # 計算に最も時間がかかる。普段はコメントアウトで良いかも
        rotation_range = 30,                    # 回転角度[degree]
        zoom_range=0.5,                         # 拡大縮小率、[1-zoom_range, 1+zoom_range]
        fill_mode='nearest',                    # 引き伸ばしたときの外側の埋め方
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        rescale=1,                              # 
        width_shift_range=0.2,                  # 横方向のシフト率
        height_shift_range=0.2)                 # 縦方向のシフト率
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
        data_format = "channels_last"
        
        # pattern 1, flower
        dir_names_dict = {"yellow":["sample_image_flower/1_train"], 
                          "white":["sample_image_flower/2_train"]} 
        param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
        x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = load(read_images1, param, validation_rate=0.2)
        model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成
        
        # pattern 1, animal
        #dir_names_dict = {"cat":["sample_image_animal/cat"], 
        #                  "dog":["sample_image_animal/dog"]} 
        #param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
        #x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = load(read_images1, param, validation_rate=0.2)
        #model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成

        # pattern 2, animal
        #dir_names_list = ["sample_image_animal/cat", "sample_image_animal/dog"]
        #name_dict = read_name_dict("sample_image_animal/file_list.csv")
        #param = {"dir_names_list":dir_names_list, "name_dict":name_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
        #x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = load(read_images2, param, validation_rate=0.2)
        #model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成
    

    # 諸々を確認のために表示
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(weights_dict)
    print(label_dict)
    print(y_train, y_test_o)

    # 学習
    epochs = 10               # 1つのデータ当たりの学習回数
    batch_size = 50             # 学習係数を更新するために使う教師データ数
    cb_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')  # 学習を適当なタイミングで止める仕掛け
    cb_save = keras.callbacks.ModelCheckpoint("cb_model.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5)  # 学習中に最高の成績が出るたびに保存

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
    th = 0.4  # 尤度の閾値
    result_raw = model.predict(x_test, batch_size=batch_size, verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
    result_list = [len(arr) if np.max(arr) < th else arr.argmax() for arr in result_raw]  # 最大尤度を持つインデックスのlistを作る。ただし、最大尤度<thの場合は、"ND"扱いとする
    predicted_classes = np.array([label_dict[class_id] for class_id in result_list])   # 予測されたclass_local_idをラベルに変換
    print("test result: ", predicted_classes)
    correct_classse = [label_dict[num] for num in y_test_o]  # 正解class_idをラベルに変換
    save_validation_table(predicted_classes, correct_classse, label_dict)

    # 学習結果を保存
    print(model.summary())      # レイヤー情報を表示(上で表示させると流れるので)
    model.save('model.hdf5')    # 獲得した結合係数を保存
    plot_history(history)       # lossの変化をグラフで表示


if __name__ == "__main__":
    main()


