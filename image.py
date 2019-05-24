# coding: utf-8

import cv2
import os
from image_processing import *
from findtext import *

s_cnt = 31750

for i in range(1,9000):
    file_path = '/Users/suhyeongcho/Desktop/crawling/shape_01/'+str(i)+'.jpg'
    shape_path = '/Users/suhyeongcho/Desktop/SHAPE/'
    thres_path = '/Users/suhyeongcho/Desktop/THRES/'
    try:
        img = cv2.imread(file_path,cv2.IMREAD_COLOR)
        row,col,_ = img.shape
        # 반반 나누기
        img_l = img[:,:390]
        img_r = img[:,390:]
    except:
        print('Cannot found',str(i)+'.jpg')
        continue
    try:
        imgcolorc,imgcolorf,_,_ = image_processing(img_l)
        # img = cv2.resize(img,(96,96))
        print(file_path,"left is processed")
        origarr, textarr = findtext(imgcolorc,imgcolorf)
        for original,text in zip(origarr,textarr):
            cv2.imwrite(shape_path+str(s_cnt)+'.jpg',original)
            cv2.imwrite(thres_path+str(s_cnt)+'.jpg',text)
            s_cnt = s_cnt + 1
    except:
            print("Error Occcured! at",file_path,"left")
    try:
        imgcolorc,imgcolorf,_,_ = image_processing(img_l)
        # img = cv2.resize(img,(96,96))
        print(file_path,"right is processed")
        origarr, textarr = findtext(imgcolorc,imgcolorf)
        for original,text in zip(origarr,textarr):
            cv2.imwrite(shape_path+str(s_cnt)+'.jpg',original)
            cv2.imwrite(thres_path+str(s_cnt)+'.jpg',text)
            s_cnt = s_cnt + 1
    except:
            print("Error Occcured! at",file_path,"right")

    # os.remove(file_path)