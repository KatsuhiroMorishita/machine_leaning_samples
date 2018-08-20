# URLリストの記載されたファイルを読み込んで、順番にダウンロードする。主に画像を落とすためのツール
# author: Katsuhiro Morishita
# created: 2018-08-12
# license: MIT
import sys
import urllib.request
import re



#fname = "imagenet.synset_cat_url.txt"   # 読み込むURLリストのファイル名
fname = "imagenet.synset_dog_url.txt"
download_num = 100   # ダウンロード数


def download(url, path="./"):
    try:
        fp = urllib.request.urlopen(url)
    except Exception as e:
        print(str(e))
        return
    local = open(path + re.sub(r'[:\\\/]', '_', url), 'wb')
    local.write(fp.read())
    local.close()
    fp.close()



with open(fname, "r") as fr:
    lines = fr.readlines()

for i, url in enumerate(lines):
    url = url.rstrip()
    if i > download_num:
        break
    if url == "":
        continue

    print(url)
    download(url)