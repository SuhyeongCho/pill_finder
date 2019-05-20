# coding: utf-8

import cv2
import os
from image_processing import *
from findtext import *

# directory = './crawling/shape_01'
# extension = '.jpg'
for i in range(1,10000):
    # file_index = directory + '/_' + str(i)
    # file_path = file_index + extension
    file_path = '/Users/suhyeongcho/Desktop/crawling/shape_01/'+str(i)+'.jpg'
    try:
        img = cv2.imread(file_path,cv2.IMREAD_COLOR)
        row,col,_ = img.shape
        # 반반 나누기
        img_l = img[:,:390]
        img_r = img[:,390:]
    except:
        break
    try:
        cv2.imshow('imgl',img_l)
        imgcolorc,imgcolorf,_,_ = image_processing(img_l)
        # img = cv2.resize(img,(96,96))
        print(file_path,"left is processed")
        arr = findtext(imgcolorc,imgcolorf)
        for happy in arr:
            cv2.imshow('a',happy)
            cv2.waitKey(0)
        # cv2.imwrite(file_index + '_l' + extension,img)
    except:
            print("Error Occcured! at",file_path,"left")
    try:
        cv2.imshow('imgr',img_r)
        imgcolorc,imgcolorf,_,_ = image_processing(img_r)
        # img = cv2.resize(img,(96,96))
        print(file_path,"right is processed")
        arr = findtext(imgcolorc,imgcolorf)
        for happy in arr:
            cv2.imshow('b',happy)
            cv2.waitKey(0)

        # cv2.imwrite(file_index + '_r' + extension,img)

    except:
        print("Error Occcured! at",file_path,"right")

    # os.remove(file_path)
    
    
