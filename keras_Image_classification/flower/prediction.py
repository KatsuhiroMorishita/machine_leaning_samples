# purpose: kerasによる花の画像を利用したCNNのテスト　　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 
# created: 2018-02-17
from sklearn import preprocessing # 次元毎の正規化に使う
from keras.models import model_from_json
from PIL import Image
import numpy as np
import os



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
img1 = read_image('1_test')
img2 = read_image('2_test')
x = np.array(img1 + img2)  # リストを結合
x = preprocessing(x)       # 画像の前処理

# 機械学習器を復元
model = model_from_json(open('model', 'r').read())
model.load_weights('param.hdf5')


# テスト用のデータを保存
with open("test_result.csv", "w") as fw:
    test = model.predict_classes(x)
    test = [str(val) for val in test] # 出力を文字列に変換
    print(test)
    fw.write("{0}\n".format("\n".join(test)))