# purpose: kerasによるCNNの画像識別テスト　　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 
# created: 2018-08-15
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import os
import statistics

from mlcore import *


# 動作上の条件（必要に応じてコメントアウト）
np.random.seed(seed=1)


def main():
    # 画像を読み込む（必要ない変数にはdummyを付けた）
    label_dict, param = restore(['label_dict.pickle', 'param.pickle'])
    param["dir_names_dict"] = {"yellow":["sample_image_flower/1_test"],   # そもそも正解クラスが不明な場合は、keyを適当な文字列に置き換えてください
                               "white":["sample_image_flower/2_test"]} 
    x, y, weights_dict_dummy, label_dict_dummy, output_dim_dummy, file_names = read_images1(param)   # 予想のためだけに画像を読み込むと、意図によってはoutput_dimやlabel_dictは適当でなくなるのでdummyが良い（使わない）

    # 機械学習器を復元
    model = load_model('model.hdf5')

    # 画像データのちょっと変わったやつを作成するオブジェクト
    datagen = ImageDataGenerator(
        rotation_range = 10,                    # 回転角度[degree]
        zoom_range=0.1,                         # 拡大縮小率、[1-zoom_range, 1+zoom_range]
        fill_mode='nearest',                    # 引き伸ばしたときの外側の埋め方
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        rescale=1,                              # 
        width_shift_range=0.1,                  # 横方向のシフト率
        height_shift_range=0.1)                 # 縦方向のシフト率

    # 予測とその結果の保存
    th = 0.4  # 尤度の閾値
    batch_size = 10
    predicted_classes = []

    for img in x:
        # ImageDataGeneratorを使って、反転などのバージョン違いの画像を複数作成する
        img_array = []
        for _ in range(batch_size):
            img_array.append(img)
        img_array = np.array(img_array)
        #img_gen = datagen.flow(img_array, save_to_dir=dir_name, save_format="jpg", batch_size=5)  # 確認用に、生成した画像を保存
        img_gen = datagen.flow(img_array, batch_size=batch_size) 
        x_ = img_gen.next()  # 画像生成

        # 生成した画像に対して予測を実施
        result_raw = model.predict(x_, batch_size=batch_size, verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
        result_list = [len(arr) if np.max(arr) < th else arr.argmax() for arr in result_raw]  # 最大尤度を持つインデックスのlistを作る。ただし、最大尤度<thの場合は、"ND"扱いとする
        _predicted_classes = np.array([label_dict[class_id] for class_id in result_list])   # 予測されたclass_idをクラス名に変換
        _class = statistics.mode(_predicted_classes)  # 最頻値を求める
        predicted_classes.append(_class)

        # 確認のための表示
        print([np.max(arr) for arr in result_raw])
        print(_predicted_classes)

    print("test result: ", predicted_classes)
    correct_classse = [label_dict[z] for z in y]  # 正解class_idをラベルに変換
    save_validation_table(predicted_classes, correct_classse, label_dict)

    df = pd.DataFrame()
    df["file name"] = file_names
    df["correct classse"] = correct_classse
    df["predicited classes"] = predicted_classes
    df.to_csv("prediction_result.csv", index=False, encoding="utf-8-sig")



if __name__ == "__main__":
    main()