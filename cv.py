import cv2
import numpy as np

image = cv2.imread('./test3.jpg')
row,col,_ = image.shape
image = cv2.resize(image,(int(col/2),int(row/2)))
grayimage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
grayimage = cv2.GaussianBlur(grayimage, (3,3), 0)
cv2.namedWindow('image')

sobelX = cv2.Sobel(grayimage, cv2.CV_64F, 1, 0)
sobelY = cv2.Sobel(grayimage, cv2.CV_64F, 0, 1)

sobelX = np.uint8(np.absolute(sobelX))
sobelY = np.uint8(np.absolute(sobelY))
add = sobelY + sobelX
ret, thresh = cv2.threshold(add,30,255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(add,100,255, cv2.THRESH_BINARY)


cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel add", add)
cv2.imshow("Sobel thresh", thresh)
cv2.imshow("Sobel thresh2", thresh2)
cv2.imshow("Sobel min", thresh - thresh2)







# def nothing(x):
#     pass

# # create trackbars for color change
# cv2.createTrackbar('threshold1', 'image', 0, 255, nothing)
# cv2.createTrackbar('threshold2', 'image', 0, 255, nothing)

# while True:
#     threshold1 = cv2.getTrackbarPos('threshold1', 'image')
#     threshold2 = cv2.getTrackbarPos('threshold2', 'image')

#     edges = cv2.Canny(mul, threshold1, threshold2)
#     cv2.imshow('gray',grayimage)
#     cv2.imshow('image', edges)
#     k = cv2.waitKey(1) & 0xFF
#     if k == 27:
#         break

# cv2.destroyAllWindows()






cv2.waitKey(0)