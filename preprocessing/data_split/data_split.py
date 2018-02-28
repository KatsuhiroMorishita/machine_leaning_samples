# purpose: 機械学習用に、データをランダムに並べ替えた上で分割する
# author: Katsuhiro Morishita
# created: 2017-07-10
# license: MIT
import sys
import numpy as np
import pandas
 
argvs = sys.argv  # コマンドライン引数を格納したリストの取得
if len(argvs) < 3:
    print("以下の様に、引数で処理対象と分割比を指定して下さい")
    print(">python data_split.py hoge.csv 0.8")
    exit()

target = argvs[1]       # 分割対象のファイル名を取得
ratio = float(argvs[2]) # 分割の割合

# データの読み込み
df = pandas.read_csv(target)
df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True) # ランダムに並べ替える（ソートされたデータには効果が高い）

# 分割
p = int(ratio * len(df)) # 分割点を計算
df1 = df.iloc[:p, :]
df2 = df.iloc[p:, :]

# 保存する
df1.to_csv("split1.csv", index=False)
df2.to_csv("split2.csv", index=False)