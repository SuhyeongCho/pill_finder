import cv2

def make_square(img):
    row,col = img.shape
    if row > col:
        size = row
    else:
        size = col

    rowBorder = (int)((size - row)/2)
    colBorder = (int)((size - col)/2)

    dest = cv2.copyMakeBorder(img, rowBorder, rowBorder, colBorder, colBorder, cv2.BORDER_CONSTANT, value = [0,0,0])

    return dest

# dir = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','default']



# for idx in dir:
#     for i in range(1,64912):
#         try :
#             filepath = './char/'+idx+'/'+str(i)+'.jpg'
#             img = cv2.imread(filepath)
#             # dest = resize(img)
#             dest = cv2.resize(img,(48,48))
#             cv2.imwrite(filepath,dest)
#             print("write : "+filepath)
#         except:
#             continue


