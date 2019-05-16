import cv2
import os
from image_processing import *

directory = './crawling/shape_01'
extension = '.jpg'
for i in range(1,10000):
    file_index = directory + '/_' + str(i)
    file_path = file_index + extension
    try:
        img = cv2.imread(file_path,cv2.IMREAD_COLOR)
        row,col,_ = img.shape
        # 반반 나누기
        img_l = img[:,:390]
        img_r = img[:,390:]
    except:
        break
    try:
        imgcolorc,imgcolors,_,_ = image_processing(img_l)
        # img = cv2.resize(img,(96,96))
        print(file_path,"left is processed")
        cv2.imwrite(file_index + '_l' + extension,img)
    except:
        print("Error Occcured! at",file_path,"left")
    try:
        imgcolorc,imgcolors,_,_ = image_processing(img_r)
        # img = cv2.resize(img,(96,96))
        print(file_path,"right is processed")
        cv2.imwrite(file_index + '_r' + extension,img)

    except:
        print("Error Occcured! at",file_path,"right")

    # os.remove(file_path)
    
    
