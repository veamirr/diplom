import cv2
import numpy as np
import skimage
from skimage.filters import threshold_otsu
from skimage.measure import euler_number, label


def preprocessing(img):
    kernel = np.ones((5,5),np.uint8)
    open5 = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return open5


def neighbours(x, y, image):
    "Return 8-neighbours of image point P1(x,y), in a clockwise order"
    img = image
    x_1, y_1, x1, y1 = x - 1, y - 1, x + 1, y + 1
    return [img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1],  # P2,P3,P4,P5
            img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1]]  # P6,P7,P8,P9


def transitions(neighbours):
    "No. of 0,1 patterns (transitions from 0 to 1) in the ordered sequence"
    n = neighbours + neighbours[0:1]  # P2, P3, ... , P8, P9, P2
    return sum((n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]))  # (P2,P3), (P3,P4), ... , (P8,P9), (P9,P2)


def zhangSuen(image):
    "the Zhang-Suen Thinning Algorithm"
    Image_Thinned = image.copy()  # deepcopy to protect the original image
    changing1 = changing2 = 1  # the points to be removed (set as 0)
    while changing1 or changing2:  # iterates until no further changes occur in the image
        # Step 1
        changing1 = []
        rows, columns = Image_Thinned.shape  # x for rows, y for columns
        for x in range(1, rows - 1):  # No. of  rows
            for y in range(1, columns - 1):  # No. of columns
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0: Point P1 in the object regions
                        2 <= sum(n) <= 6 and  # Condition 1: 2<= N(P1) <= 6
                        transitions(n) == 1 and  # Condition 2: S(P1)=1
                        P2 * P4 * P6 == 0 and  # Condition 3
                        P4 * P6 * P8 == 0):  # Condition 4
                    changing1.append((x, y))
        for x, y in changing1:
            Image_Thinned[x][y] = 0
        # Step 2
        changing2 = []
        for x in range(1, rows - 1):
            for y in range(1, columns - 1):
                P2, P3, P4, P5, P6, P7, P8, P9 = n = neighbours(x, y, Image_Thinned)
                if (Image_Thinned[x][y] == 1 and  # Condition 0
                        2 <= sum(n) <= 6 and  # Condition 1
                        transitions(n) == 1 and  # Condition 2
                        P2 * P4 * P8 == 0 and  # Condition 3
                        P2 * P6 * P8 == 0):  # Condition 4
                    changing2.append((x, y))
        for x, y in changing2:
            Image_Thinned[x][y] = 0
    return Image_Thinned


def texture_features(img):
    # Calculate the co-occurrence matrix for the image
    co_matrix = skimage.feature.graycomatrix(img, [5], [0], levels=256, symmetric=True, normed=True)

    # Calculate texture features from the co-occurrence matrix
    contrast = skimage.feature.graycoprops(co_matrix, 'contrast')
    correlation = skimage.feature.graycoprops(co_matrix, 'correlation')
    energy = skimage.feature.graycoprops(co_matrix, 'energy')
    homogeneity = skimage.feature.graycoprops(co_matrix, 'homogeneity')

    return float(contrast), float(correlation), float(energy), float(homogeneity)


def features(img):
    contrast, correlation, energy, homogeneity = texture_features(img)

    thresh = threshold_otsu(img)
    img = img < thresh

    euler = euler_number(img, connectivity=1)
    objects = label(img, connectivity=1).max()
    holes = objects - euler

    width = img.shape[0]
    height = img.shape[1]

    img_skeleton = zhangSuen(img)
    # plt.imshow(img_skeleton)
    skeleton_size = sum(sum(img_skeleton))
    # skelet = skeleton_size(img)

    return [width, height, objects, holes, contrast, correlation, energy, homogeneity]