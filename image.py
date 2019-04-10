import cv2
from image_processing import *

directory = './croll'
for i in range(36):
    filename = 'croll'+str(i+1)+'.jpg'
    filepath = directory +'/'+filename
    try:
        img = cv2.imread(filepath,cv2.IMREAD_COLOR)
        img = image_processing(img)
        print(filepath," is processed")
        cv2.imshow(filename,img)
        k = cv2.waitKey(0)
        if k == 27:
            break
        cv2.destroyAllWindow()
    except:
        print("Error Occcured! at ",filepath)
        continue
    
