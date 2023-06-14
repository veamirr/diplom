import class2_preprocessing
import cv2
import numpy as np

path = r'images\\be_129.png'
img = cv2.imread(path)
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
compare = gray

cntrs, y_max, y_min = class2_preprocessing.contour_segmentation(gray)
compare = np.concatenate((compare, cntrs), axis=1)
res = class2_preprocessing.hough_straight_lines(gray, cntrs, y_max, y_min)
compare = np.concatenate((compare, res), axis=1)

#res = class2_preprocessing.letter_hor_trim(res)
#res = class2_preprocessing.letter_vert_trim(res)

cv2.imwrite('images\\be_129_segm.png', compare)
cv2.imwrite('images\\be_129_segm_res.png', res)
