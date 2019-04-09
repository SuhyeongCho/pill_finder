import cv2
import numpy as np

image = cv2.imread('./5.jpeg')

row,col,_ = image.shape

row = int(row/2)
col = int(col/2)
image = cv2.resize(image,(col,row))

cv2.imshow('image',image)

hsvimage = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsvimage)


z = np.zeros((row,col))
for i in range(row):
    for j in range(col):
         z[i][j] = v[i][j] / 255

b3 = np.zeros((row,col))
b5 = np.zeros((row,col))
b3_1 = np.zeros((row,col))

for i in range(row):
    for j in range(col):
        if z[i][j] < 0.3:
            b3[i][j] = 1.0
        else:
            b3[i][j] = 0.0

# cv2.imshow('b3',b3)
for i in range(row):
    for j in range(col):
        if z[i][j] >= 0.3 and z[i][j] < 0.5:
            b5[i][j] = 1.0
        else:
            b5[i][j] = 0.0
# cv2.imshow('b5',b5)

for i in range(row):
    for j in range(col):
        b3_1[i][j] = 1 - b3[i][j]
# cv2.imshow('b3_1',b3_1)

m3 = 0.0
for i in range(row):
    for j in range(col):
        m3 = m3 + b3[i][j]*z[i][j]


m3_1 = 0.0
for i in range(row):
    for j in range(col):
        m3_1 = m3_1 + b3[i][j]


m3 = m3/m3_1
print('m3',m3)

bw = np.zeros((row,col))
for i in range(row):
    for j in range(col):
        bw[i][j] = b3[i][j] + 0.7*b5[i][j]
# cv2.imshow('bw',bw)

g_v = cv2.GaussianBlur(z,(3,3),5)

# cv2.imshow('g_v',g_v)

g_bw = cv2.GaussianBlur(bw,(31,31),5)
# cv2.imshow('g_bw',g_bw)

for i in range(row):
    for j in range(col):
        z[i][j] = z[i][j] + g_v[i][j]*g_bw[i][j]

cv2.imshow('z',z)
print(z)

hsv = cv2.merge([h,s,z])

# print(hsv.shape)
# image = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
# cv2.imshow('image1',image)

cv2.waitKey(0)