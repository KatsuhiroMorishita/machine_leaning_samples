# purpose: 画像の前処理を行う
# author: Katsuhiro MORISHITA　森下功啓
# created: 2018-08-15
import numpy as np
from scipy.ndimage.interpolation import rotate
from scipy.misc import imresize
from PIL import Image
from PIL import ImageOps
#from keras.preprocessing.image import ImageDataGenerator

import image_preprocessing as ip



def preprocessing(imgs):
    """ 画像の前処理
    必要なら呼び出して下さい。
    
    imgs: ndarray, 画像が複数入っている多次元配列
    """
    return imgs / 255


def random_crop(image, crop_size=(224, 224)):
    """
    ref: https://www.kumilog.net/entry/numpy-data-augmentation
    """
    h, w, _ = image.shape

    # 0~(400-224)の間で画像のtop, leftを決める
    top = np.random.randint(0, h - crop_size[0])
    left = np.random.randint(0, w - crop_size[1])

    # top, leftから画像のサイズである224を足して、bottomとrightを決める
    bottom = top + crop_size[0]
    right = left + crop_size[1]

    # 決めたtop, bottom, left, rightを使って画像を抜き出す
    image = image[top:bottom, left:right, :]
    return image


def scale_augmentation(image, scale_range=(256, 400), crop_size=224):
    """
    ref: https://www.kumilog.net/entry/numpy-data-augmentation
    """
    scale_size = np.random.randint(*scale_range)
    image = imresize(image, (scale_size, scale_size))
    image = random_crop(image, (crop_size, crop_size))
    return image


def cutout(image_origin, mask_size):
    """
    ref: https://www.kumilog.net/entry/numpy-data-augmentation
    """
    # 最後に使うfill()は元の画像を書き換えるので、コピーしておく
    image = np.copy(image_origin)
    mask_value = image.mean()

    h, w, _ = image.shape
    # マスクをかける場所のtop, leftをランダムに決める
    # はみ出すことを許すので、0以上ではなく負の値もとる(最大mask_size // 2はみ出す)
    top = np.random.randint(0 - mask_size // 2, h - mask_size)
    left = np.random.randint(0 - mask_size // 2, w - mask_size)
    bottom = top + mask_size
    right = left + mask_size

    # はみ出した場合の処理
    if top < 0:
        top = 0
    if left < 0:
        left = 0

    # マスク部分の画素値を平均値で埋める
    image[top:bottom, left:right, :].fill(mask_value)
    return image



def random_erasing(image_origin, p=0.5, s=(0.02, 0.4), r=(0.3, 3)):
    """
    ref: https://www.kumilog.net/entry/numpy-data-augmentation
    """
    # マスクするかしないか
    if np.random.rand() > p:
        return image
    image = np.copy(image_origin)

    # マスクする画素値をランダムで決める
    mask_value = np.random.randint(0, 256)

    h, w, _ = image.shape
    # マスクのサイズを元画像のs(0.02~0.4)倍の範囲からランダムに決める
    mask_area = np.random.randint(h * w * s[0], h * w * s[1])

    # マスクのアスペクト比をr(0.3~3)の範囲からランダムに決める
    mask_aspect_ratio = np.random.rand() * r[1] + r[0]

    # マスクのサイズとアスペクト比からマスクの高さと幅を決める
    # 算出した高さと幅(のどちらか)が元画像より大きくなることがあるので修正する
    mask_height = int(np.sqrt(mask_area / mask_aspect_ratio))
    if mask_height > h - 1:
        mask_height = h - 1
    mask_width = int(mask_aspect_ratio * mask_height)
    if mask_width > w - 1:
        mask_width = w - 1

    top = np.random.randint(0, h - mask_height)
    left = np.random.randint(0, w - mask_width)
    bottom = top + mask_height
    right = left + mask_width
    image[top:bottom, left:right, :].fill(mask_value)
    return image



# kerasのImageDataGeneratorと同じような使い方ができる様にした
class MyImageDataGenerator:
    def __init__(self, rotation_range=0, zoom_range=0, horizontal_flip=False, vertical_flip=False, width_shift_range=0.0, height_shift_range=0.0, crop=True, random_erasing=False, mixup=0.0):
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.horizontal_flip = horizontal_flip 
        self.vertical_flip = vertical_flip
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.crop = crop
        self.random_erasing = random_erasing
        self.mixup = mixup

    def flow(self, x, y, save_to_dir=None, save_format=None, batch_size=10, shape=(100, 100)):
        """
        x: 0-1.0に正規化された画像
        """
        def get_img(index):
            img = x[index]
            img_ = img.copy() * 255.   # scipyの関数が整数にしか対応していないので、スケール返還と整数化。この行ではスケールを0-1から0-255に変換している
            img_ = img_.astype(np.uint8)
            return img_

        i = 0
        while True:
            x_ = []
            y_ = []
            for k in range(batch_size):
                j = (i + k) % len(x)
                img1 = get_img(j)
                lavel_vect1 = y[j]
                
                if self.rotation_range != 0:
                    theta = np.random.randint(-self.rotation_range, self.rotation_range)
                    h, w, channel = img1.shape
                    img1 = rotate(img1, theta)
                    img1 = imresize(img1, (h, w))
                if self.horizontal_flip and np.random.rand() < 0.5:
                    img1 = np.fliplr(img1)
                if self.vertical_flip and np.random.rand() < 0.5:
                    img1 = np.flipud(img1)
                if np.random.rand() < self.mixup:
                    n = np.random.randint(0, len(x))
                    r = np.random.rand()
                    img2 = get_img(n)
                    lavel_vect2 = y[n]

                    img1 = img1 * r + img2 * (1 - r)
                    lavel_vect1 = lavel_vect1 * r + lavel_vect2 * (1 - r)

                if save_to_dir is not None:
                    pilImg = Image.fromarray(np.uint8(img1))
                    pilImg.save("{0}/{1}_{2}_hoge.{3}".format(save_to_dir, i, k, save_format))

                img1 = img1.astype(np.float16) / 255.   # スケールを0-255から0-1に変換
                x_.append(img1)
                y_.append(lavel_vect1)
            yield x_, y_
            i = (i + batch_size) % len(x)




def main():
    data_format = "channels_last"
    dir_names_dict = {"yellow":["sample_image_flower/1_train"], 
                      "white":["sample_image_flower/2_train"]} 
    param = {"dir_names_dict":dir_names_dict, "data_format":data_format, "size":(100, 100), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
    x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = ip.load_save_images(ip.read_images1, param, validation_rate=0.2)

    # pattern 2, animal
    #dir_names_list = ["sample_image_animal/cat", "sample_image_animal/dog"]
    #name_dict = read_name_dict("sample_image_animal/file_list.csv")
    #param = {"dir_names_list":dir_names_list, "name_dict":name_dict, "data_format":data_format, "size":(32, 32), "mode":"RGB", "resize_filter":Image.NEAREST, "preprocess_func":preprocessing}
    #x_train, y_train_o, x_test, y_test_o, weights_dict, label_dict, y_train, y_test, output_dim = ip.load_save_images(read_images2, param, validation_rate=0.2)


    # 教師データを無限に用意するオブジェクトを作成
    datagen = MyImageDataGenerator(
        #samplewise_center = True,              # 平均をサンプル毎に0
        #samplewise_std_normalization = True,   # 標準偏差をサンプル毎に1に正規化
        #zca_whitening = True,                  # 計算に最も時間がかかる。普段はコメントアウトで良いかも
        rotation_range = 30,                    # 回転角度[degree]
        zoom_range=0.1,                         # 拡大縮小率、[1-zoom_range, 1+zoom_range]
        #fill_mode='nearest',                    # 引き伸ばしたときの外側の埋め方
        horizontal_flip=True,                   # 水平方向への反転
        vertical_flip=True,                     # 垂直方向での反転
        #rescale=1,                              # 
        width_shift_range=0.2,                  # 横方向のシフト率
        height_shift_range=0.2,                 # 縦方向のシフト率
        mixup = 0.5)                            # 画像の混合確率

    batch_size = 10
    amount_per_image = 1
    x_train_new = []
    y_train_new = []
    img_gen = datagen.flow(x_train, y_train, save_to_dir="generated_image", save_format="jpg", batch_size=10)  # 確認用に、生成した画像を保存
    #img_gen = datagen.flow(x_train, y_train, batch_size=10)
    for i in range(int(len(x_train) / batch_size * amount_per_image)):
        x_, y_ = next(img_gen)  # 画像生成
        x_train_new += list(x_)
        y_train_new += list(y_)
        print(i)
        #print(i, y_)

    x_train_new = np.array(x_train_new)
    y_train_new = np.array(y_train_new)
    print(x_train_new.shape)
    print(y_train_new.shape)
    np.save('x_train.npy', x_train_new)
    np.save('y_train.npy', y_train_new)


if __name__ == "__main__":
    main()


