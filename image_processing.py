import cv2
import numpy as np

def backproject(source, target, levels = 2, scale = 1):
        hsv = cv2.cvtColor(source,  cv2.COLOR_BGR2HSV)
        hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
        # calculating object histogram
        roihist = cv2.calcHist([hsv],[0, 1], None, [levels, levels], [0, 180, 0, 256] )

        # normalize histogram and apply backprojection
        cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
        dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256], scale)
        return dst
# def fill(imgx,imglast,threshMap,img_size):
#     row,col = img_size
#     kernel = np.ones((3,3),np.uint8)
#     threshMap1 = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)


#     for i1 in range(row):
#         if threshMap1[i1][(int)(col/2)] > 20 :
#             break
#     for j1 in range(col):
#         if threshMap1[(int)(row/2)][j1] > 20 :
#             break
#     size=5
#     while (size < 20):
#         break
#         kernel = np.ones((size,size),np.uint8)
#         threshMap2 = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)

#         size+=2
#         for i2 in range(row):
#             if threshMap2[i2][(int)(col/2)] > 20 :
#                 break
#         for j2 in range(col):
#             if threshMap2[(int)(row/2)][j2] > 20 :
#                 break
#         if i1 != i2 or j1 != j2:
#             break
#         else:
#             threshMap1 = threshMap2
#             i1 = i2
#             j1 = j2
    
#     threshMap = threshMap1
#     threshMap = cv2.blur(threshMap, (3,3))
#     for i in range(row):
#         for j in range(col):
#             if (threshMap[i][j]==0) :
#                 imgx[i][j]=0
#                 imglast[i][j]=0
#     return imglast,threshMap

def fill(imgx,imglast,threshMap,img_size):
    kernel = np.ones((19,19),np.uint8)
    row,col = img_size
    threshMap = cv2.morphologyEx(threshMap, cv2.MORPH_CLOSE, kernel)
    threshMap = cv2.GaussianBlur(threshMap, (7,7), 0)            
    for i in range(row):
        for j in range(col):
            if (threshMap[i][j]==0) :
                imgx[i][j]=0
                imglast[i][j]=0
    return imglast,threshMap

# 약 설정
def image_processing(img):

    # 약 설정
    imgx = img[:370,:]
    imglast = imgx.copy()
    cv2.pyrMeanShiftFiltering(imgx, 2, 10, imgx, 4)



    backproj = np.uint8(backproject(imgx, imgx, levels = 2))
    backproj = 255 - backproj

    cv2.normalize(backproj,backproj,0,255,cv2.NORM_MINMAX)

    saliencies = [backproj, backproj, backproj]
    saliency = cv2.merge(saliencies)


    cv2.pyrMeanShiftFiltering(saliency, 20, 200, saliency, 2)
    saliency = cv2.cvtColor(saliency, cv2.COLOR_BGR2GRAY)


    saliencyMap = saliency

    threshMap = cv2.threshold(saliencyMap.astype("uint8"), 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


    #hsv로 변환후 v가 0.5이하면 그림자로 취급 검은색으로 만듬
    hsv = cv2.cvtColor(imgx, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    (row,col) = v.shape


    #최대 v 최소v 찾고 0에서 1로 정규화
    max = v[0][0]
    min = v[0][0]

    ave=0
    cnt=0

    for i in range(row):
        for j in range(col):
            ave+=v[i][j]
            cnt+=1
            if (max<v[i][j]) :
                max = v[i][j]
            elif (min > v[i][j]) :
                min = v[i][j]

    # 그림자 제거            
    for i in range(row):
        for j in range(col):
            x = (v[i][j]-min)/((max-min)+1e-10)
            if(x<=0.3) :
                threshMap[i][j] = 0

    #가운데 기준으로 흰색 경계부분 검출


    imglast,threshMap = fill(imgx,imglast,threshMap,(row,col))


    contours, hierarchy = cv2.findContours(threshMap * 1,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    contours = sorted(contours, key = cv2.contourArea)


    x,y,w,h = cv2.boundingRect(contours[-1])
    if(h>w):
        x = x-(int)((h-w)/2)
        w=h
    else:
        y = y-(int)((w-h)/2)
        h=w
    aaa = threshMap[y:y+h,x:x+w]
    bbb = imglast[y:y+h,x:x+w]

    return bbb






