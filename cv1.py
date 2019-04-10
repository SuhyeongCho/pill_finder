import cv2
import numpy as np

image = cv2.imread('./croll/croll3.jpg',cv2.IMREAD_COLOR)

row,col,_ = image.shape
print(image.shape)
row = int(row/2)
col = int(col/2)
print(row,col)
image = image[47:374,:col]

cv2.imshow('image',image)
grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

# cv2.imshow('gray',grayimage)

sobelX = cv2.Sobel(image, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(image, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
add = sobelY + sobelX

cv2.imshow('add',add)
ret, th1 = cv2.threshold(add,127,255,cv2.THRESH_BINARY)

th3 = cv2.adaptiveThreshold(add,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,15,2)
cv2.imshow('th1',th1)

cv2.imshow('th3',th3)


# edges = cv2.Canny(image,50,200)

# cv2.imshow('canny',edges)



cv2.waitKey(0)