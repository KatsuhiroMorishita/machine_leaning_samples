# purpose: 機械学習向けに、各次元をN(0,1)にスケーリングする
# author: Katsuhiro Morishita
# created: 2017-07-21
# lisence: MIT
import sys
import pandas as pd
import numpy as np



class scaler:
    """ 各次元をN(0,1)にスケーリングするクラス
    """
    def __init__(self):
        """ コンストラクタ
        """
        self._std = None
        self._mean = None

    def scale(self, df, reset=False):
        """ 各次元毎に正規化を行う
        df: pandas.DataFrame
        """
        if self._std is None or self._mean is None or reset or len(df.columns) != len(self._std.index):
            self._std = np.std(df)
            self._mean = np.mean(df)
        return (df - self._mean) / self._std

    def save(self):
        """ 保持している平均や標準偏差を保存する
        """
        if self._std is not None and self._mean is not None:
            df = pd.DataFrame(index=self._std.index.values)
            df["std"] = self._std
            df["mean"] = self._mean
            df.to_csv("sc_param")

    def load(self):
        """ 保存されていた平均や標準偏差を読み込む
        """
        df = pd.read_csv("sc_param", index_col=0)
        self._std = df["std"]
        self._mean = df["mean"]


def main():
    # 単独実行された場合は動作テスト（サンプルデータがファイルで必要）
    df = pd.read_csv("sapmle_data.csv") # ファイルを読み込む
    #print(df.iloc[:, :-1]) # データの切り出し方の練習（無視して下さい）
    sc = scaler()
    df2 = sc.scale(df)
    print(df2)
    sc.save()
    sc.load()
    df2 = sc.scale(df)
    print(df2)



if __name__ == '__main__':
    main()