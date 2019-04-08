import cv2
import numpy as np

image = cv2.imread('./7.jpeg')
row,col,_ = image.shape
# image = cv2.resize(image,(int(col/2),int(row/2)))
grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

hsvimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)

_,_,grayimage = cv2.split(hsvimage)

cv2.imshow('v',grayimage)
grayimage = cv2.GaussianBlur(grayimage, (5,5), 0)

for i in range(row):
    for j in range(col):
        if grayimage[i][j] + 30 >= 255:
            grayimage[i][j] = 255
        else:
            grayimage[i][j] = grayimage[i][j]+30


cv2.imshow('v1',grayimage)




sobelX = cv2.Sobel(grayimage, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(grayimage, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
add = sobelY + sobelX
mul = sobelY * sobelX

cv2.imshow('add',add)
cv2.imshow('mul',mul)








# image = cv2.imread('5.jpeg')
# # row,col,_ = image.shape
# # image = cv2.resize(image,(int(col/2),int(row/2)))

# lab = cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
# cv2.imshow('lab',image)

# lab = cv2.GaussianBlur(lab, (3,3), 0)
# lab = cv2.GaussianBlur(lab, (3,3), 0)



# edge = cv2.Canny(lab,100,200)

# cv2.imshow('egge',edge)

# th = cv2.adaptiveThreshold(edge,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2)

# cv2.imshow('th',th)












cv2.waitKey(0)