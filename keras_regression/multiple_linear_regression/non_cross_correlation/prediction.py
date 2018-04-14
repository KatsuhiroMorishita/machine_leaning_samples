#----------------------------------------
# purpose: kerasによる重回帰テストスクリプト　予測編
# このスクリプトは説明変数の各スケールに大きな違いがない場合に有効です。
# author: Katsuhiro MORISHITA　森下功啓
# memo: 読み込むデータは、1行目に列名があり、最終列に解が入っていること。
# created: 2017-07-08
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
df = pandas.read_csv("regression_test.csv")
s = len(df.columns)
x = (df.iloc[:, :-1]).values # ndarrayに変換

# 機械学習器を復元
model = model_from_json(open('model', 'r').read())
model.load_weights('param.hdf5')

# テスト用のデータを保存
with open("test_result.csv", "w") as fw:
    test = model.predict(x)
    print(test)
    fw.write(ndarray2str(test))
