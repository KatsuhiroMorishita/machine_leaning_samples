# AUCの計算
# memo: 2値分類データの予測結果の成績評価に利用して下さい
# author: Katsuhiro Morishita
# created: 2017-12-18
# lisence: MIT
import numpy as np
import matplotlib.pyplot as plt


def get_AUC(correct_arr, predicted_arr):
    """ AUCを求めて返す
    correct_arr: list<int>, 正解ラベルが1か0で入っているものとする。また、1をpositive/陽とする。
    """
    amount_of_positive = np.sum(predicted_arr)                                  #  正解ラベル1の数
    amount_of_negative = len(predicted_arr) - amount_of_positive #  正解ラベル０の数
    
    # 閾値を変えつつ、真陽性と偽陽性の集計を取る
    tpr_array = []
    fpr_array = []
    for th in np.arange(0.1, 1.0, 0.05): # th: 閾値
        tp = 0                                                      # true positive, 真陽性
        fp = 0                                                      # false positive, 偽陽性
        for i in range(len(predicted_arr)):
            c = correct_arr[i]
            val = predicted_arr[i]
            r = 0
            if val > th: 
                r = 1
            if c == 0 and r == 1:
                fp += 1
            elif c == 1 and r == 1:
                tp += 1
        tp_rate = tp / amount_of_positive
        fp_rate = fp / amount_of_negative
        tpr_array.append(tp_rate)
        fpr_array.append(fp_rate)

        print("{0:.2f}\t{1:.2f}\t{2:.2f}".format(th, fp_rate, tp_rate))   # 計算の経過を表示, 不要ならコメントアウト

    # ROC曲線を描く, 不要ならコメントアウト
    plt.plot(fpr_array, tpr_array)
    plt.show()

    # AUC（ROC曲線の下の面積）を求める
    tpr_array.append(0)
    fpr_array.append(0)
    _x, _y, auc = 1, 1, 0
    for x, y in zip(fpr_array, tpr_array):
        w = _x - x
        auc += (y + _y) * w / 2                 # 台形積分
        _x = x
        _y = y

    return auc


def main():
    # 正解データを乱数で作る
    size = 1000
    current_data = np.random.randint(2, size=size)

    # 予測データを作る（正解にノイズを足して作る）
    mu = 0.0
    sigma = 0.3                                                        # sigmaを小さくするとAUCが大きくなる
    noise = np.random.normal(mu, sigma, size) 
    predicted_result = current_data + noise
    predicted_result[predicted_result > 1] = 1  # はみ出た分は調整
    predicted_result[predicted_result < 0] = 0
    print("正解と予測の散布図")
    plt.scatter(current_data, predicted_result, alpha=0.3)
    plt.show()
    
    print("auc", get_AUC(current_data, predicted_result))
    
if __name__ == "__main__":
    main()