import cv2
import os

shape_dir = 'SHAPE/'
thres_dir = 'THRES/'
labeling_dir = 'LABEL/'

#1:15000 finished!!

for idx in range(14800,15001):
    file_name = str(idx)+'.jpg'
    shape_img = cv2.imread(shape_dir+file_name)
    thres_img = cv2.imread(thres_dir+file_name)

    cv2.imshow('shape',shape_img)
    cv2.imshow('thres',thres_img)

    input_key = cv2.waitKey(0)
    if input_key == 27:
        print(str(idx) + ' : default')
        shape_img_path = labeling_dir +shape_dir+ 'default' + '/'
        thres_img_path = labeling_dir +thres_dir+ 'default' + '/'
    else:
        print(str(idx) + ' : ' + str(chr(input_key)))
        shape_img_path = labeling_dir +shape_dir+ chr(input_key) + '/'
        thres_img_path = labeling_dir +thres_dir+ chr(input_key)+ '/'

    if not os.path.exists(shape_img_path):
        os.makedirs(shape_img_path)
    if not os.path.exists(thres_img_path):
        os.makedirs(thres_img_path)

    cv2.imwrite(shape_img_path+file_name,shape_img)
    cv2.imwrite(thres_img_path+file_name,thres_img)


