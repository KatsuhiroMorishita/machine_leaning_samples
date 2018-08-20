# purpose: 画像の前処理用モジュール
# main()では、画像の読み込みと保存を行います。
# author: Katsuhiro MORISHITA　森下功啓
# created: 2018-08-20
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


def to_categorical(array_1d):
    """ 整数で表現されたカテゴリを、ニューラルネットワークの出力層のユニットに合わせてベクトルに変換する
    kerasが無くても動作した方が良い気がして、自前で実装した。
    array_1d: ndarray or list, 1次元配列で整数が格納されていることを期待している
    """
    _max = np.max(array_1d)
    ans = []
    for val in array_1d:
        vector = [0] * (_max + 1)
        vector[val] = 1.          # mixupを考えると、浮動小数点が良い
        ans.append(vector)
    
    return np.array(ans)


def one_hotencoding(data=[]):
    """ one-hotencodingを行う
    （2018-08-12: クラスの数の割にクラス毎のサンプル数が少ないことが原因でdata内の各要素におけるクラスIDの欠落が生じないように、ロジックを書き換えた）
    data: list<ndarray>, 1次元のndarrayを格納したリスト
    """
    fusion = []   # 一旦、全部結合させる
    for mem in data:
        fusion += list(mem)
    fusion_onehot = to_categorical(fusion)  # 全部を一緒にしてからone-hotencoding

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



def load_save_images(read_func, param, validation_rate=0.1):
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
    np.save('y_train.npy', y_train)
    np.save('y_train_o.npy', y_train_o)
    np.save('x_test.npy', x_test)
    np.save('y_test.npy', y_test)
    np.save('y_test_o.npy', y_test_o)
    with open('weights_dict.pickle', 'wb') as f:  # 再利用のために、ファイルに保存しておく
        pickle.dump(weights_dict, f)
    with open('label_dict.pickle', 'wb') as f:  # 再利用のために、ファイルに保存しておく
        pickle.dump(label_dict, f)
    with open('param.pickle', 'wb') as f:  # 再利用のために、ファイルに保存しておく
        pickle.dump(param, f)

    return x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim



def main():
    data_format = "channels_last"

    # pattern 1, flower
    dir_names_dict = {"yellow":["sample_image_flower/1_train"], 
                      "white":["sample_image_flower/2_train"]} 
    param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
    load_save_images(read_images1, param, validation_rate=0.2)

    
    # pattern 1, animal
    #dir_names_dict = {"cat":["sample_image_animal/cat"], 
    #                  "dog":["sample_image_animal/dog"]} 
    #param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
    #load_save_images(read_images1, param, validation_rate=0.2)

    # pattern 2, animal
    #dir_names_list = ["sample_image_animal/cat", "sample_image_animal/dog"]
    #name_dict = read_name_dict("sample_image_animal/file_list.csv")
    #param = {"dir_names_list":dir_names_list, "name_dict":name_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
    #load_save_images(read_images2, param, validation_rate=0.2)
    

if __name__ == "__main__":
    main()


