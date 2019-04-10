import numpy as np
import cv2

def convex(img):

    grayimage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(grayimage,cv2.CV_64F,1,0)
    sobelY = cv2.Sobel(grayimage,cv2.CV_64F,0,1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    grayimage = sobelX + sobelY
    cv2.imshow('add',grayimage)


    _,th = cv2.threshold(grayimage,127,255,0)
    contours,_ = cv2.findContours(th,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow('th',th)
    cnt = contours[0]

    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),3)

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    cv2.drawContours(img,[box],0,(0,255,0),3)

    cv2.imshow('rectangle',img)
    cv2.waitKey(0)
