# purpose: kerasによる画像識別の関数をまとめた
# main()に利用方法のサンプルを置いているので、参考にしてください。
# author: Katsuhiro MORISHITA　森下功啓
# created: 2018-08-12
import keras
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.applications.xception import Xception
from matplotlib import pylab as plt
from PIL import Image
import numpy as np
import pickle
import pandas as pd
import sys
import os

import image_preprocessing as ip



def build_model(input_shape, output_dim, data_format):
    """ 機械学習のモデルを作成する
    入力は画像、出力はラベルという構造を想定しています。
    """
    base_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    #base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)

    top_model = Sequential()   # 追加する層
    top_model.add(Conv2D(32, (3, 3), padding="same", input_shape=base_model.output_shape[1:]))
    top_model.add(Conv2D(32, (3, 3), padding="same"))
    top_model.add(Conv2D(32, (3, 3), padding="same"))
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



def check_validation(th, model, x_test, y_test_o, label_dict, batch_size):
    """ 学習成果のチェックとして、検証データに対して分割表を作成する
    th: float, 尤度の閾値
    """
    result_raw = model.predict(x_test, batch_size=batch_size, verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
    result_list = [len(arr) if np.max(arr) < th else arr.argmax() for arr in result_raw]  # 最大尤度を持つインデックスのlistを作る。ただし、最大尤度<thの場合は、"ND"扱いとする
    predicted_classes = np.array([label_dict[class_id] for class_id in result_list])   # 予測されたclass_local_idをラベルに変換
    print("test result: ", predicted_classes)
    correct_classse = [label_dict[num] for num in y_test_o]  # 正解class_idをラベルに変換
    save_validation_table(predicted_classes, correct_classse, label_dict)



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



def reload():
    """ 保存済みの画像やモデルを読み込む
    """
    obj = restore(['x_train.npy', 'y_train.npy', 'y_train_o.npy', 'x_test.npy', 'y_test.npy', 'y_test_o.npy', 'weights_dict.pickle', 'label_dict.pickle'])
    if os.path.exists("cb_model.hdf5") and obj is not None:
        # 保存してたファイルからデータを復元
        x_train, y_train, y_train_o, x_test, y_test, y_test_o, weights_dict, label_dict = obj

        # モデルを再構築
        print("--load 'cb_model.hdf5'--")
        model = load_model('cb_model.hdf5')

        return x_train, y_train, y_train_o, x_test, y_test, y_test_o, weights_dict, label_dict, model
    else:
        print("--failure for restore--")
        exit()





def main():
    # 調整することの多いパラメータを集めた
    image_shape = (32, 32)   # 画像サイズ
    epochs = 5               # 1つのデータ当たりの学習回数
    batch_size = 10          # 学習係数を更新するために使う教師データ数
    initial_epoch = 0        # 再開時のエポック数。途中から学習を再開する場合は、0以外を指定しないとhistryのグラフの横軸が0空になる
    if epochs <= initial_epoch:  # 矛盾があればエラー
        raise ValueError("epochs <= initial_epoch")

    # 教師データを無限に用意するオブジェクトを作成
    """
    datagen = ImageDataGenerator(       # kerasのImageDataGenerator
        #samplewise_center = True,              # 平均をサンプル毎に0
        #samplewise_std_normalization = True,   # 標準偏差をサンプル毎に1に正規化
        #zca_whitening = True,                  # 計算に最も時間がかかる。普段はコメントアウトで良いかも
        rotation_range = 30,                    # 回転角度[degree]
        zoom_range=0.5,                         # 拡大縮小率、[1-zoom_range, 1+zoom_range]
        #fill_mode='nearest',                    # 引き伸ばしたときの外側の埋め方
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        #rescale=1,                              # fit()などで引数xに、更に掛ける係数があれば1以外を設定
        width_shift_range=0.2,                  # 横方向のシフト率
        height_shift_range=0.2)                 # 縦方向のシフト率
    #datagen.fit(x_train)                        # zca用に、教師データの統計量を内部的に求める
    """

    #"""
    datagen = ip.MyImageDataGenerator(       # 自作のImageDataGenerator
        rotation_range = 45,                    # 回転角度[degree]
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        #crop=True,
        #random_erasing=True,
        mixup = 0.5,                            # 画像の混合確率
        shape=image_shape)                      # 出力する画像のサイズ
    #"""
    
    # 教師データの読み込みと、モデルの構築。必要なら、callbackで保存していた結合係数を読み込む
    if len(sys.argv) > 1 and sys.argv[1] == "retry":
        x_train, y_train, y_train_o, x_test, y_test, y_test_o, weights_dict, label_dict, model = reload()
    else:
        data_format = "channels_last"
        
        # pattern 1, flower
        dir_names_dict = {"yellow":["sample_image_flower/1_train"], 
                          "white":["sample_image_flower/2_train"]} 
        param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":image_shape, "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":ip.preprocessing2}
        x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = ip.load_save_images(ip.read_images1, param, validation_rate=0.2)
        model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成
        
        # pattern 1, animal
        #dir_names_dict = {"cat":["sample_image_animal/cat"], 
        #                  "dog":["sample_image_animal/dog"]} 
        #param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":image_shape, "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing2}
        #x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = ip.load_save_images(ip.read_images1, param, validation_rate=0.2)
        #model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成

        # pattern 2, animal
        #dir_names_list = ["sample_image_animal/cat", "sample_image_animal/dog"]
        #name_dict = read_name_dict("sample_image_animal/file_list.csv")
        #param = {"dir_names_list":dir_names_list, "name_dict":name_dict, "data_format":data_format, "size":image_shape, "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing2}
        #x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = ip.load_save_images(ip.read_images2, param, validation_rate=0.2)
        #model = build_model_simple(input_shape=x_train.shape[1:], output_dim=output_dim, data_format=data_format)   # モデルの作成
    

    # 諸々を確認のために表示
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    print(weights_dict)
    print(label_dict)
    print(y_train, y_test_o)

    # 学習
    cb_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='auto')  # 学習を適当なタイミングで止める仕掛け
    cb_save = keras.callbacks.ModelCheckpoint("cb_model.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=5)  # 学習中に最高の成績が出るたびに保存

    #"""
    history = model.fit_generator(   # ImageDataGeneratorを使った学習
        datagen.flow(x_train, y_train, batch_size=batch_size, shuffle=True),  # シャッフルは順序によらない学習のために重要
        epochs=epochs,
        steps_per_epoch=int(x_train.shape[0] / batch_size),
        verbose=1,
        class_weight=weights_dict,
        callbacks=[cb_stop, cb_save],
        validation_data=(x_test, y_test),  # ここにジェネレータを渡すことも出来る
        initial_epoch=initial_epoch
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）
    #"""
    
    """
    # Generatorを使わないパターン
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        verbose=1,
        class_weight=weights_dict,
        validation_data=(x_test, y_test),
        callbacks=[cb_stop, cb_save],
        shuffle=True,
        batch_size=batch_size
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）
    #"""

    # 学習成果のチェックとして、検証データに対して分割表を作成する
    check_validation(0.4, model, x_test, y_test_o, label_dict, batch_size)

    # 学習結果を保存
    print(model.summary())      # レイヤー情報を表示(上で表示させると流れるので)
    model.save('model.hdf5')    # 獲得した結合係数を保存
    plot_history(history)       # lossの変化をグラフで表示


if __name__ == "__main__":
    main()


