# 同一フォルダ内の画像のファイル名一覧を作成する
# author: Katsuhiro Morishita
# created: 2018-08-12
# license: MIT
import glob


files = glob.glob("*.jpg") + glob.glob("*.png") + glob.glob("*.bmp")

with open("file_list.csv", "w") as fw:
    for fname in files:
        fw.write(fname)
        fw.write("\n")