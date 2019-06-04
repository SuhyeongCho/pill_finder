from char_test import char_test
from shape_test import shape_test
import cv2
from image_processing import image_processing
from make_square import make_square
from findtext import findtext

image = cv2.imread('test1.jpeg')
imgcolorc,imgcolorf,imgshape,colorname = image_processing(image)
_,thres = findtext(imgcolorc,imgcolorf)

for th in thres:
    th_sq = make_square(th)
    print(char_test(th_sq)[0])

print(shape_test(imgshape)[0])
print(colorname)
