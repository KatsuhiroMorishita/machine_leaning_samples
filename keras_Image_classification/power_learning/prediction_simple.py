# purpose: kerasによるCNNの画像識別テスト　　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 
# created: 2018-08-15
from keras.models import load_model
import pandas as pd
import numpy as np
import pickle
import os

from mlcore import *



def main():
    # 画像を読み込む（必要ない変数にはdummyを付けた）
    label_dict, param = restore(['label_dict.pickle', 'param.pickle'])
    param["dir_names_dict"] = {"yellow":["sample_image_flower/1_test"],   # そもそも正解クラスが不明な場合は、keyを適当な文字列に置き換えてください
                               "white":["sample_image_flower/2_test"]} 
    x, y, weights_dict_dummy, label_dict_dummy, output_dim_dummy, file_names = read_images1(param)   # 予想のためだけに画像を読み込むと、意図によってはoutput_dimやlabel_dictは適当でなくなるのでdummyが良い（使わない）

    # 機械学習器を復元
    model = load_model('model.hdf5')

    # 予測とその結果の保存
    th = 0.4  # 尤度の閾値
    result_raw = model.predict(x, batch_size=len(x), verbose=0) # クラス毎の尤度を取得。 尤度の配列がレコードの数だけ取得される
    result_list = [len(arr) if np.max(arr) < th else arr.argmax() for arr in result_raw]  # 最大尤度を持つインデックスのlistを作る。ただし、最大尤度<thの場合は、"ND"扱いとする
    predicted_classes = np.array([label_dict[class_id] for class_id in result_list])   # 予測されたclass_local_idをラベルに変換
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