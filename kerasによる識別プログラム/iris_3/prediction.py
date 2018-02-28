# purpose: kerasによる弁別器テストスクリプト　予測編
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、全ての列が特徴ベクトルである（正解ラベルが付属しない）こと。
# created: 2017-07-30
import pandas
import pickle
import numpy as np
from sklearn import preprocessing # 次元毎の正規化に使う
from keras.models import model_from_json
from scaling import scaler


def main():
    # データの読み込み
    data = pandas.read_csv("iris_test.csv")
    x = data.iloc[:, :] # 全データが説明変数ならたぶんx = dfでもいいと思うが、たまには列指定したいかもしれないのでilocのままとする。
    sc = scaler()
    sc.load()
    x = sc.scale(x) # 次元毎に正規化する
    x = x.values # ndarrrayに変換

    # 機械学習器を復元
    model = model_from_json(open('model', 'r').read())
    model.load_weights('param.hdf5')

    # 出力をラベルに変換する辞書を復元
    label_dict = None
    with open('label_dict.pickle', 'rb') as f:
        label_dict = pickle.load(f)        # オブジェクト復元

    # 予測結果を保存
    with open("prediction_result.csv", "w") as fw:
        prediciton_data = model.predict_classes(x)
        prediciton_data = [str(label_dict[z]) for z in prediciton_data] # 出力をラベルに変換（ラベルが文字列だった場合に便利だし、str.join(list)が使えて便利）
        print(prediciton_data)
        fw.write("{0}\n".format("\n".join(prediciton_data)))


if __name__ == "__main__":
    main()
