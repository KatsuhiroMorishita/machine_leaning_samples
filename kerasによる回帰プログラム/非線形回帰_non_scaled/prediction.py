#----------------------------------------
# purpose: kerasによる非線形回帰問題の学習結果を利用した予測
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があること。なお、正解値は含まないこと（全列説明変数）。
# created: 2017-07-20
#----------------------------------------
import pandas
from keras.models import model_from_json


def ndarray2str(val):
    """ ndarray型の変数を文字列に変換する
    val: ndarray 2次元配列を仮定
    """
    out = []
    for x in val:
        temp = [str(y) for y in x]
        out.append(",".join(temp))
    return "\n".join(out)

# データの読み込み
df = pandas.read_csv("prediction_data.csv")
s = len(df.columns)
x = (df.iloc[:, :]).values # ndarrayに変換。全データが説明変数ならたぶんdf.valuesでもいいと思うが、たまには列指定したいかもしれないのでilocのままとする。

# 機械学習器を復元
model = model_from_json(open('model', 'r').read())
model.load_weights('param.hdf5')

# テスト用のデータを保存
with open("prediction_result.csv", "w") as fw:
    prediciton_data = model.predict(x)
    print(prediciton_data)
    fw.write(ndarray2str(prediciton_data))

