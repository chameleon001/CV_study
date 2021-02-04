#reference : https://github.com/hughesj919/HarrisCorner/blob/master/Corners.py

import cv2
import numpy as np

file_name = ''
img = cv2.imread(file_name)
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
img[dst>0.01*dst.max()]=[0,0,255]
cv2.imshow('dst',img)
cv2.waitKey(0)
cv2.destroyAllWindows()