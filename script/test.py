import cv2

dirpath = "./data/bird.png"
img_cv = cv2.imread(dirpath)
print(img_cv.shape)
