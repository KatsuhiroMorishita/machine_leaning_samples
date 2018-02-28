# purpose: kerasによる花の画像を利用したCNNのテスト　　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 学習に全く利用していない画像を使って、学習成果を確認する。
# created: 2018-02-17
from sklearn import preprocessing # 次元毎の正規化に使う
from keras.models import model_from_json
from keras.utils import np_utils
from PIL import Image
import pandas as pd
import numpy as np
import pickle
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
    arr1_2, arr2_2 = [], []

    times = int(rate * len(arr1))
    for _ in range(times):
        i = np.random.randint(0, len(arr1))
        arr1_2.append(arr1.pop(i))
        arr2_2.append(arr2.pop(i))

    return arr1, arr2, arr1_2, arr2_2

# 関数の動作テスト
"""
a = [1,2,3,4,5,6,7,8,9,10]
b = [11,12,13,14,15,16,17,18,19,20]
print(split(a, b, 0.2))
exit()
"""


def read_images(dir_names, data_format="channels_last", size=(32, 32), mode="RGB", resize_filter=Image.NEAREST, preprocess_func=None, validation_rate=0.1):
    """ リストで複数指定されたフォルダ内にある画像を読み込んで、リストとして返す
    通常は教師データの読み込みを想定している。

    dir_names: list<str>, フォルダ名又はフォルダへのパスを格納したリスト
    data_format: str, データ構造を指定
    size: tuple<int, int>, 読み込んだ画像のリサイズ後のサイズ
    mode: str, 読み込んだ後の画像の変換モード
    resize_filter: int, Image.NEARESTなど、リサイズに使うフィルターの種類。処理速度が早いやつは粗い
    preprocess_func: func, 前処理を行う関数
    validation_rate: float, 検証用に使うデータの割合。0-1.0
    """
    x_train, y_train = [], []  # 学習用のデータを格納する
    x_test, y_test = [], []    # 検証用のデータを格納する
    y_train_onehot = []        # y_trainをone-hotencodingしたオブジェクト
    y_test_onehot = []         # y_testをone-hotencodingしたオブジェクト
    label_dict = {}            # 番号からフォルダ名を返す辞書
    weights = []               # 学習の重み

    for i in range(len(dir_names)):    # 貰ったフォルダ名の数だけループを回す
        name = dir_names[i]
        label_dict[i] = name           # 番号からフォルダ名を取得できる辞書を作成（予測段階で役に立つ）
        imgs = read_image(name, data_format=data_format, size=size, mode=mode, resize_filter=resize_filter)
        label = [i] * len(imgs)
        x1, y1, x2, y2 = split(imgs, label, validation_rate)   # 教師データを学習用と検証用に分割
        x_train = x_train + x1
        y_train = y_train + y1
        x_test = x_test + x2
        y_test = y_test + y2
        weights.append(len(imgs))

    # クラスごとの重みの計算と、重みの辞書の作成（教師データ数の偏りを是正する）
    weights = np.array(weights)
    weights = np.max(weights) / weights
    weights_dict = {i:weights[i] for i in range(len(weights))}

    # 画像の前処理
    if preprocess_func is not None:
        x_train = preprocess_func(x_train)
        x_test = preprocess_func(x_test)

    # 正解ラベルをone-hotencoding（1次元ベクトル→2次元のmatrixになる）
    if len(y_train) > 0:
        y_train_onehot = np_utils.to_categorical(y_train)
    if len(y_test) > 0:
        y_test_onehot = np_utils.to_categorical(y_test)

    return np.array(x_train), np.array(y_train_onehot), np.array(x_test), np.array(y_test_onehot), y_train, y_test, weights_dict, label_dict



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




def main():
    # 画像を読み込む（今回の用途に必要ないも変数にはdummyを付けた）
    x, y_dummy, x_dummy, y_test_dummy, y_train_o, y_test_o_dummy, weights_dict_dummy, label_dict_4prediction = read_images(['1_test', '2_test'], preprocess_func=preprocessing, validation_rate=0.0)

    # 機械学習器を復元
    model = model_from_json(open('model', 'r').read())
    model.load_weights('param.hdf5')

    # 出力をラベルに変換する辞書を復元
    label_dict = None
    with open('label_dict.pickle', 'rb') as f:
        label_dict = pickle.load(f)              # オブジェクト復元

    # 予測とその結果の保存
    prediciton_data = model.predict_classes(x)   # 予測の実行
    prediciton_data = [str(label_dict[z]) for z in prediciton_data]   # 出力をラベルに変換（ラベルが文字列だった場合に便利だし、str.join(list)が使えて便利）
    input_data = [str(label_dict_4prediction[z]) for z in y_train_o]  # 入力をフォルダ名に変換
    df = pd.DataFrame()
    df["input"] = input_data
    df["prediciton"] = prediciton_data
    df.to_csv("prediction_result.csv", index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    main()