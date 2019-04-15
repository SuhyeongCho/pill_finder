import cv2
from image_processing import *
from convex import *

directory = './crawling'
for i in range(1,201):
    filename = str(i)+'.jpg'
    filepath = directory +'/'+filename
    try:
        img = cv2.imread(filepath,cv2.IMREAD_COLOR)
        row,col,_ = img.shape
        img_l = img[:,:390]
        img_r = img[:,390:]

        img = image_processing(img_l)
        print(filepath,"left is processed")
        cv2.imshow(filename+"left",img)
        k = cv2.waitKey(0)
        if k == 27:
            break

    except:
        print("Error Occcured! at",filepath,"left")
        continue
    try:
        img = image_processing(img_r)
        print(filepath,"right is processed")
        cv2.imshow(filename+"right",img)
        k = cv2.waitKey(0)
        if k == 27:
            break

    except:
        print("Error Occcured! at",filepath,"right")
        continue

    
    
