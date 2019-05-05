import cv2


image = cv2.imread('./pic/triangle/a.jpg',cv2.IMREAD_COLOR)
for i in range(1,100):
    cv2.imwrite('./pic/triangle/'+str(i)+'.jpg',image)
