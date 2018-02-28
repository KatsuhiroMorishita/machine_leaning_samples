# AUCの計算をsklearnのライブラリを使って計算する
# memo: 2値分類データの予測結果の成績評価に利用して下さい
# ref: http://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html
# author: Katsuhiro Morishita
# created: 2017-12-18
# lisence: MIT
import sklearn.metrics as mtr
import numpy as np
import matplotlib.pyplot as plt
                 
def main():
    # 正解データを乱数で作る
    size = 200
    y_true = np.random.randint(2, size=size)

    # 予測データを作る（正解にノイズを足して作る）
    mu = 0.0
    sigma = 0.3                                                        # sigmaを小さくするとAUCが大きくなる
    noise = np.random.normal(mu, sigma, size) 
    y_score = y_true + noise
    y_score[y_score > 1] = 1  # はみ出た分は調整
    y_score[y_score < 0] = 0
    print("正解と予測の散布図")
    plt.scatter(y_true, y_score, alpha=0.3)
    plt.show()
    
    fpr, tpr, thresholds = mtr.roc_curve(y_true, y_score, pos_label=1)   # ftr: false_positive,  tpr: true_positive
    plt.plot(fpr, tpr)
    plt.show()
    
    auc = mtr.auc(fpr, tpr)
    print(auc)
    
if __name__ == "__main__":
    main()