'''
Author: SimonCK666 SimonYang223@163.com
Date: 2022-08-31 16:41:02
LastEditors: SimonCK666 SimonYang223@163.com
LastEditTime: 2022-08-31 17:07:03
FilePath: /SimonCK666.github.io/script/imageResize.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from PIL import Image

def ResizeImage(filein, fileout, width, height):
    img = Image.open(filein)
    out = img.resize((width, height),Image.ANTIALIAS) # resize image with high-quality
    # out.save(fileout, type)
    out.save(fileout)

if __name__ == "__main__":
    filein = r'../assets//resized//prof_pic_yh.jpeg'
    fileout = r'../assets//resized//prof_pic-480x659.jpeg'
    width = 480
    height = 659
    ResizeImage(filein, fileout, width, height)
