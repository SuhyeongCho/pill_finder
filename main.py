from char_test import char_test
from shape_test import shape_test
import cv2
from image_processing import image_processing
from make_square import make_square
from findtext import findtext

def char_decoding(c):
    if c < 10:
        return str(chr(c+48))
    elif c == 36:
        return ""
    else:
        return str(chr(c - 10 + 65))

def shape_decoding(s):
    if s == 0:
        return "원형"
    elif s == 1:
        return "타원형"
    elif s == 2:
        return "삼각형"
    elif s == 3:
        return "사각형"
    elif s == 4:
        return "마름모형"
    elif s == 5:
        return "오각형"
    elif s == 6:
        return "육각형"
    elif s== 7:
        return "팔각형"

image = cv2.imread('test7.png')
imgcolorc,imgcolorf,imgshape,colorname = image_processing(image)
_,thres = findtext(imgcolorc,imgcolorf)

print("char : ",end="")
for th in thres:
    cv2.imshow("th",th)
    cv2.waitKey(0)
    th_sq = make_square(th)
    c = char_test(th_sq)[0]
    print(char_decoding(c),end="")
print()

s = shape_test(imgshape)[0]
cv2.imshow("a",imgshape)
cv2.waitKey(0)
print("shape :",shape_decoding(s))

print("color :",colorname)
