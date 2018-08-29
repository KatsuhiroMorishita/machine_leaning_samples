# サンプル画像を作成する
# ref: https://gist.github.com/lazykyama/dabe526246d60fa937d1
# author: Katsuhiro Morishita
# created: 2018-08-21
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

def save_char_img(char, fontname='Osaka', size=(64, 64)):
    img = Image.new('L', size, 'white')
    draw = ImageDraw.Draw(img)
    fontsize = int(size[0] * 0.8)
    font = ImageFont.truetype(font=fontname, size=fontsize)

    # adjust charactor position.
    char_displaysize = font.getsize(char)
    offset = tuple((si - sc) // 2 for si, sc in zip(size, char_displaysize))  # 文字を表示する左上の座標を求める

    # adjust offset, half value is right size for height axis.
    draw.text((offset[0], offset[1] // 2), char, font=font, fill='#000')
    img.save("img_{0}.png".format(char))


for c in "0123456789abcdefghijklmnopqrstuvwxyz":
    save_char_img(c, fontname="arial.ttf")