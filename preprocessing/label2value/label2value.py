# purpose: 機械学習用のデータセットに含まれるラベルを整数に置換する
# author: Katsuhiro Morishita
# created: 2017-07-11
# license: MIT
import sys
import pandas as pd
import math
import numpy as np


def label2val(df):
    """ ラベルを整数に変換する
    df: DataFrame at pandas, 表状のオブジェクト
    """
    # 1列ずつ、ラベルかどうか確認しつつラベルを整数に置換する
    for i in range(len(df.columns)):
        print(i)

        # 列のデータを取得
        col = df[df.columns[i]]

        # ラベルを取得する
        labels = set()   # ラベルの種類を調べたいので、集合を使う
        for j in range(len(col)): # 全ての行を処理
            val = col.iloc[j]
            if isinstance(val, str):
                labels.add(val)
            elif math.isnan(val): # nanはfloatなので、float判定より先に無くてはならない。
                labels.add("nan")    
            elif isinstance(val, bool) or isinstance(val, np.bool_):
                labels.add(str(val))
            elif isinstance(val, int) or isinstance(val, float) or isinstance(val, np.int32) or isinstance(val, np.int64) or isinstance(val, np.float32) or isinstance(val, np.float64):
                labels = set() # 数値が列内に1つでもあれば変換しない。
                break

        # ラベルの列があれば、整数に変換
        print(labels)
        if len(labels) != 0: # ラベルが1つ以上あれば処理
            new_col = []
            labels = list(labels)
            for j in range(len(col)): # 全ての行を処理
                val = col.iloc[j]
                if not isinstance(val, str) and math.isnan(val):
                    new_col.append(labels.index("nan"))    
                elif isinstance(val, bool) or isinstance(val, np.bool_):
                    new_col.append(labels.index(str(val)))    
                else:
                    new_col.append(labels.index(val))
            #print(new_col)
            df[df.columns[i]] = new_col # 列をまるごと入れ替える

    return df


def main():
    # 引数から処理対象のファイル名を取得
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    if len(argvs) < 2:
        print("引数で処理対象を指定して下さい")
        exit()
    target = argvs[1]       # 分割対象のファイル名を取得

    # 指定されたファイルを処理
    df = pd.read_csv(target) # ファイルを読み込む
    df = label2val(df)
    df.to_csv("valued.csv", index=False)


if __name__ == "__main__":
    main()
    