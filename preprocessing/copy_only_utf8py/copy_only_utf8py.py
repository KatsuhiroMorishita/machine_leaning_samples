# purpose: Pythonで扱えない文字の入っているUTF-8のファイルから、処理できる文字のみを残す
# memo: 副作用として、文字列が崩れる可能性がある。
# author: Katsuhiro Morishita
# created: 2017-07-18
# license: MIT
import sys
import os


def main():
    # 引数から処理対象のファイル名を取得
    argvs = sys.argv  # コマンドライン引数を格納したリストの取得
    if len(argvs) < 2:
        print("引数で処理対象を指定して下さい")
        exit()
    target = argvs[1]       # 分割対象のファイル名を取得

    # 指定されたファイルを処理
    name, ext = os.path.splitext(target)
    txt = open(target, "r", encoding="utf-8-sig", errors="ignore").read()
    open("{0}_copy{1}".format(name, ext), "w", encoding="utf-8-sig", errors="ignore").write(txt)


if __name__ == "__main__":
    main()