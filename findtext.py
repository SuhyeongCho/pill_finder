import numpy as np
import cv2

def findtext(imgcolorc,imgcolors):
    img = imgcolorc
    img2 = imgcolors
    row,col,_ = img.shape
    row2,col2,_ = img2.shape
    distrow = (int)(row*0.1)
    distcol = (int)(col*0.1)
    distrow2 = (int)(row2*0.1)
    distcol2 = (int)(col2*0.1)
    img = img[0+distrow:row-distrow,0+distcol:col-distcol]
    img2 = img2[0+distrow2:row2-distrow2,0+distcol2:col2-distcol2]

    grayimage = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    grayimage2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    sobelX = cv2.Sobel(grayimage,cv2.CV_64F,1,0)
    sobelY = cv2.Sobel(grayimage,cv2.CV_64F,0,1)

    sobelX2 = cv2.Sobel(grayimage2,cv2.CV_64F,1,0)
    sobelY2 = cv2.Sobel(grayimage2,cv2.CV_64F,0,1)

    sobelX = np.uint8(np.absolute(sobelX))
    sobelY = np.uint8(np.absolute(sobelY))

    sobelX2 = np.uint8(np.absolute(sobelX2))
    sobelY2 = np.uint8(np.absolute(sobelY2))

    grayimage = sobelX + sobelY
    grayimage2 = sobelX2 + sobelY2

    row,col = grayimage.shape

    for i in range(0,row):
        for j in range(0,col):
            if grayimage[i][j] >= grayimage[i][j]-grayimage2[i][j]:
                grayimage[i][j] = grayimage[i][j]-grayimage2[i][j]
    sum=0
    for i in range(0,row):
        for j in range(0,col):
            sum = sum + grayimage[i][j]

    sum = (int)(sum/(row*col))
    sum = sum+(int)(sum*0.2)

    grayimage = cv2.GaussianBlur(grayimage,(9,9),0)
    _,th = cv2.threshold(grayimage,sum,255,0)
    arr =[]
    arr2 =[]
    countp=0
    _,contours,_ = cv2.findContours(th,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    fcount=0
    flag=0
    count=0

    max = 0
    for i in range(1,len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt)
        if w*h>= max :
            max = w*h
    arr=[]
    count=0
    for i in range(0,len(contours)):
        cnt = contours[i]
        x,y,w,h = cv2.boundingRect(cnt) 
        if w*h<(int)(max*0.3):
            continue;
        tmp = th[y:y+h,x:x+w]
        arr.append(tmp.copy())
    return arr