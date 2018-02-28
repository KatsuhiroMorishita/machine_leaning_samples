# http://blog.amedama.jp/entry/2017/04/02/130530
# 2017-07-21
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn import preprocessing 


def main():
    dataset = datasets.load_iris()

    features = dataset.data
    features = preprocessing.scale(features) # 相関行列を基にした解析をしたいなら、この行を有効にする
    targets = dataset.target
    print(targets)
    print(targets == targets[0])

    # 主成分分析する
    pca = PCA(n_components=4)
    pca.fit(features)

    # 分析結果を元にデータセットを主成分に変換する
    transformed = pca.fit_transform(features)
    print(type(transformed))

    # 主成分をプロットする
    for label in np.unique(targets):
        plt.scatter(transformed[targets == label, 0],
                    transformed[targets == label, 1])
    plt.title('principal component')
    plt.xlabel('pc1')
    plt.ylabel('pc2')

    # 主成分の寄与率を出力する
    print('各次元の寄与率: {0}'.format(pca.explained_variance_ratio_))
    print('累積寄与率: {0}'.format(sum(pca.explained_variance_ratio_)))

    # グラフを表示する
    plt.show()


if __name__ == '__main__':
    main()



    