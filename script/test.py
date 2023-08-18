import cv2

dirpath = "./script/data/bird.png"
img_cv = cv2.imread(dirpath)
print(img_cv.shape)
