import cv2
import numpy as np


# вычисление общей площади контуров
def all_contour_areas(contours):
    all_areas = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        all_areas += area
    return all_areas


# вычиcление верхней и нижней координаты наибольшего контура
def largest_contour_y_coordinates(contours):
    largest_item = max(contours, key=cv2.contourArea)
    y_max = np.amax(largest_item, axis=0)[0][1]
    y_min = np.amin(largest_item, axis=0)[0][1]
    return y_max, y_min


# выделение контуров с наибольшей площадью
def contour_segmentation(img):
    # удаляем все непрокрашенные части (не сильно)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    # compare = img

    # диалатация для удаления тонких соединений
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    # compare = np.concatenate((compare, img), axis=1)

    # вычисление контуров
    edges = cv2.Canny(img, 50, 200)
    # compare = np.concatenate((compare, edges), axis=1)

    # соединение разорванных контуров
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    # compare = np.concatenate((compare, edges), axis=1)

    # контуры
    contours, hierarchy = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # поиск контуров, лежащих в границах наибольшего контура по оси ОУ
    # и занимающих больше чем 5% от площади всех контуров
    all_areas = all_contour_areas(contours)
    y_max, y_min = largest_contour_y_coordinates(contours)
    contour_list = []
    for cnt in contours:
        if (cv2.contourArea(cnt) > (0.05 * all_areas)) and (np.amax(cnt, axis=0)[0][1] <= y_max * 1.1):
            contour_list.append(cnt)
    res = np.zeros([img.shape[0], img.shape[1], 1], dtype=np.uint8)
    res.fill(255)

    for cnt in contour_list:
        res = cv2.drawContours(res, [cnt], -1, color=0, thickness=cv2.FILLED)

    res = cv2.bitwise_not(res)

    return res, y_max, y_min


# поиск горизонтальных линий на изображении
def hough_straight_lines(img, img_cntr_segm, y_max, y_min):
    # сглаживание Гаусса
    kernel_size = 5
    blur_gray = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # вычисление контуров
    edges = cv2.Canny(blur_gray, 50, 150)

    # преобразовния Хафа
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / (180)  # angular resolution in radians of the Hough grid
    threshold = 10  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 15  # minimum number of pixels making up a line
    max_line_gap = 100  # maximum gap in pixels between connectable line segments

    line_image = np.copy(img) * 0  # creating a blank to draw lines on
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    res = img_cntr_segm
    for line in lines:
        for x1, y1, x2, y2 in line:
            # условие на тангенс, что угол меньше 60 градусов и координаты контуров лежат в пределах наибольшего
            if (10 >= (y2 - y1) >= -10) and (max(y1, y2) <= (y_min + 0.8 * (y_max - y_min))):
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)
                res = cv2.rectangle(img_cntr_segm, (x1, y1), (x2, y2), (255, 255, 255), 3)

    return res


# обрезка буквицы с помощью вертикального профиля
def letter_vert_trim(img):
    old = np.asarray(img)
    rows, cols = old.shape
    new = np.zeros((rows, cols))

    # поиск верхней границы
    curr_row = 0
    for i in range(rows):
        flag = 0
        for j in range(cols):
            new[curr_row][j] = old[i][j]
            # if old[i][j] < 255:
            if old[i][j] > 0:
                # в строке найден кусочек буквы
                flag = 1
        if flag == 0:
            # удаление пустой строки
            new = np.delete(new, curr_row, 0)
            break
        else:
            curr_row += 1

    # поиск нижней границы
    curr_row = rows - 1
    for i in range(rows-1,-1,-1):
        flag = 0
        for j in range(cols):
            new[curr_row][j] = old[i][j]
            # if old[i][j] < 255:
            if old[i][j] > 0:
                # в строке найден кусочек буквы
                flag = 1
        if flag == 0:
            # удаление пустой строки
            new = np.delete(new, curr_row, 0)
            break
        else:
            curr_row -= 1

    new = new.astype(np.uint8)
    return new


# обрезка буквицы с помощью горизонтального профиля
def letter_hor_trim(img):
    old = np.asarray(img)
    rows, cols = old.shape
    new = np.zeros((rows, cols))

    curr_col = 0
    for j in range(cols):
        flag = 0
        for i in range(rows):
            new[i][curr_col] = old[i][j]
            #if old[i][j] < 255:
            if old[i][j] > 0:
                # в строке найден кусочек буквы
                flag = 1
        if flag == 0:
            # удаление пустой строки
            new = np.delete(new, curr_col, 1)
        else:
            curr_col += 1

    new = new.astype(np.uint8)
    return new
