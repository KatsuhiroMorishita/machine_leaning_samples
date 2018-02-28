# purpose: kerasによるiris識別プログラム　学習編
# memo: 読み込むデータは、1行目に列名があり、最終列に層名（文字列でクラス名、または整数で0-nの連番）が入っていること。
# author: Katsuhiro MORISHITA　森下功啓
# created: 2017-07-30
import numpy as np
import pandas as pd
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from sklearn.feature_extraction import DictVectorizer # 判別問題における文字列による正解ラベルをベクトル化する
import matplotlib.pyplot as plt
from scaling import scaler


def read_data(fname, ratio=0.8, shuffle=True, x_scaling=True):
    """ データの読み込み
    ratio: float, 分割比
    shuffle: bool, Trueだと、行方向でデータをランダムに並び替える
    x_scaling: bool, Trueだと、特徴ベクトルを次元毎（特徴量毎）に正規化N(0,1)する。モデルの活性化関数がLeakyReLU以外だと必須
    """
    df = pd.read_csv(fname)
    if shuffle:
        df = df.reindex(np.random.permutation(df.index)).reset_index(drop=True) # ランダムに並べ替える（効果高い）
    x = df.iloc[:, :-1]
    if x_scaling: # 必要なら正規化する
        sc = scaler()
        x = sc.scale(x)
        sc.save()
    x = x.values # ndarrayに変換
    y = (df.iloc[:, -1:]).values # 最後の列が正解データ
    print("x", x)
    print("y", y)
    p = int(ratio * len(df))
    x_train = x[:p] # 学習に使うデータ
    y_train = y[:p]
    x_test = x[p:] # 検証に使うデータ
    y_test = y[p:]
    return x_train, y_train, x_test, y_test, len(df.columns)


def to_vector(label_array):
    """ 正解ラベル（文字列 or 0-nの整数）をベクトル（例：[1,0,0]）に変換し、predict_classes()の出力をラベルに変換する辞書も作成する
    学習用と検証用に分割済みのデータにも対応できるように、やや複雑になってしまった・・・。
    label_array: list<ndarray>, ラベルの格納された1列だけで構成されたndarrayのリスト。リストの要素は1つでもOK
    """
    # 分割されたデータを結合
    data = label_array[0]
    for i in range(1, len(label_array)):
        _data = label_array[i]
        data = np.r_[data, _data]

    # ベクトルに変換
    vect = None
    if data.dtype == "object":  # ラベルが文字列かチェック
        vec = DictVectorizer()
        vect = vec.fit_transform([{"class":mem[0]} for mem in data]).toarray() # 判別問題における文字列による正解ラベルをベクトル化する
    else:
        vect = np_utils.to_categorical(data)
    print("vect", vect)

    # 0-nの整数からラベルに変換する辞書を作成（例： 0:setosa）。 predict_classes()が0-nを返す事を利用する
    labels = np.ravel(data) # 出力をラベルに変換するための布石
    label_dict = {list(vect[i]).index(vect[i].max()):labels[i] for i in range(len(labels))} # 出力をラベルに変換する辞書

    # ベクトルの配列を分割
    vectors = []
    p = 0
    for i in range(len(label_array)):
        _df = label_array[i]
        vectors.append(vect[p:p+len(_df)])
        p += len(_df)

    return (vectors, label_dict)


def get_class_weights(y, lavel_dict):
    """ 学習（訓練）データに含まれる正解ラベルの割合からクラスの重みを計算して返す
    y: 2D ndarray, ラベルの格納された1列だけで構成されたndarray
    lavel_dict: dict, predict_classes()が返す数値（クラス番号）をラベルに変換する辞書
    """
    labels = list(np.ravel(y))
    majority = max([labels.count(lavel_dict[key]) for key in lavel_dict])
    class_weight = {key:majority/labels.count(lavel_dict[key]) for key in lavel_dict}
    print("--class waights--", class_weight)
    return class_weight



def build_model(input_dim, output_dim):
    """ 機械学習のモデルを作成する
    """
    model = Sequential()
    model.add(Dense(3, input_shape=(input_dim, ))) # 入力層は全結合層で入力がinput_dim次元のベクトルで、出力先のユニット数が3。
    model.add(Activation('relu')) # 中間層の活性化関数がReLU
    model.add(Dense(output_dim)) # 出力層のユニット数
    model.add(Activation('sigmoid')) # 出力層の活性化関数がsigmoid
    model.add(Activation('softmax')) # 出力の合計を1にする層を追加
    opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0003) # 学習係数を大きめに取っている
    model.compile(optimizer=opt,
          loss='categorical_crossentropy', # binary_crossentropy
          metrics=['accuracy'])
    return model


def plot_history(history):
    """ 損失の履歴を図示する
    from http://www.procrasist.com/entry/2017/01/07/154441
    """
    plt.plot(history.history['loss'],"o-",label="loss",)
    plt.plot(history.history['val_loss'],"^-",label="val_loss")
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(loc='lower right')
    plt.grid(which="both")
    plt.yscale("log") # ケースバイケースでコメントアウト
    plt.show()


def save_validation_table(prediction, correct_data, label_dict):
    """ 学習に使わなかった検証データに対する予測と正解ラベルを使って、スレットスコアの表的なものを作って保存する
    prediction: ndarray, 1次元配列を想定。予測されたラベルが格納されている事を想定
    correct_data: ndrray, 2次元配列を想定。正解ラベルが格納されている事を想定
    label_dict: pandas.DataFrame, 整数のkeyでラベルを取り出す辞書を想定
    """
    prediction = np.array([label_dict[x] for x in prediction]) # 予測されたクラスをラベルに変換
    correct_data = np.ravel(correct_data) # 1次元配列に変換

    # 行と列の名称のリストを作成
    keys = list(label_dict.keys())
    names = [label_dict[x] for x in range(min(keys), max(keys)+1)]

    # 結果を集計する
    df1 = pd.DataFrame(index=names, columns=names)
    df1 = df1.fillna(0)
    for i in range(len(prediction)):
        v1 = prediction[i]
        v2 = correct_data[i]
        #print("--v--", v1, v2)
        df1.loc[[v1],[v2]] += 1 # 行名と列名でセルを指定できる
    print(df1)
    df1.to_csv("validation_table1.csv")

    # 正解ラベルを使って正規化する
    df2 = df1.copy()
    amount = [len(np.where(correct_data==x)[0]) for x in names] # 正解ラベルをカウント
    #amount = [len(np.where(prediction==x)[0]) for x in names] # 予測値をカウント
    print(amount)
    for i in range(len(df1)):
        df2.iloc[:,i] = df2.iloc[:,i] / amount[i] # 列単位で割る
        #df2.iloc[i,:] = df2.iloc[i,:] / amount[i] # 行単位で割る
    print(df2)
    df2.to_csv("validation_table2.csv")


def main():
    # データ読み込み
    x_train, y_train_o, x_test, y_test_o, s = read_data("iris_learning_str_label.csv")

    # 正解ラベルをベクトルに変換、ついでにpredict_classes()の出力をラベルに変換する辞書を作成
    y, label_dict = to_vector([y_train_o, y_test_o])
    y_train, y_test = y
    print(y_train_o, y_test_o)
    print(y_train, y_test)
    with open('label_dict.pickle', 'wb') as f: # 再利用のために、ファイルに保存しておく
        pickle.dump(label_dict, f)

    # 重みを計算
    class_weight = get_class_weights(y_train_o, label_dict)

    # 学習器の準備
    model = build_model(s-1, len(y_train[0]))

    # 学習
    epochs = 150 # 1つのデータ当たりの学習回数
    batch_size = 5
    history = model.fit(x_train, y_train,
        epochs=epochs, 
        batch_size=batch_size, 
        verbose=1, 
        validation_split=0.1,
        validation_data=(x_test, y_test), # validation_dataをセットするとvalidation_splitは無視される
        shuffle=True, # 1epoch毎にデータをシャッフル
        class_weight=class_weight
        ) # 返り値には、学習中のlossやaccなどが格納される（metricsに指定する必要がある）

    # 学習のチェック
    result = model.predict_classes(x_test, batch_size=batch_size, verbose=0) # クラス推定, 0-nの整数が得られる
    print("result: ", result)
    for i in range(len(result)):
        mem = result[i]
        #print("label convert test,", mem, label_dict[mem], y_test_o[i])
    save_validation_table(result, y_test_o, label_dict)

    # 学習結果を保存
    print(model.summary()) # レイヤー情報を表示(上で表示させると流れるので)
    open("model", "w").write(model.to_json()) # モデル情報の保存
    model.save_weights('param.hdf5') # 獲得した結合係数を保存
    plot_history(history) # lossの変化をグラフで表示


if __name__ == "__main__":
    main()

