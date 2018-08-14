import cv2
import numpy as np


def draw_four_vectors(img, line, color=(0, 255, 0)):
    """
    :param line: (x1,y1,x2,y2,x3,y3,x4,y4)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    img = cv2.line(img, (line[0], line[1]), (line[2], line[3]), color)
    img = cv2.line(img, (line[2], line[3]), (line[4], line[5]), color)
    img = cv2.line(img, (line[4], line[5]), (line[6], line[7]), color)
    img = cv2.line(img, (line[6], line[7]), (line[0], line[1]), color)
    return img


def draw_bounding_box(img, line, color=(255, 0, 0)):
    """
    :param line: (xmin, ymin, xmax, ymax)
    """
    img = cv2.line(img, (line[0], line[1]), (line[2], line[1]), color)
    img = cv2.line(img, (line[2], line[1]), (line[2], line[3]), color)
    img = cv2.line(img, (line[2], line[3]), (line[0], line[3]), color)
    img = cv2.line(img, (line[0], line[3]), (line[0], line[1]), color)
    return img


def get_min_area_rect(line):
    """
    :param line: (x1,y1,x2,y2,x3,y3,x4,y4)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return  (x1,y1,x2,y2,x3,y3,x4,y4, angle)
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    rect = cv2.minAreaRect(np.asarray([[line[0], line[1]],
                                       [line[2], line[3]],
                                       [line[4], line[5]],
                                       [line[6], line[7]]]))
    angle = rect[2]

    # 获得最小 rotate rect 的四个角点
    box = cv2.boxPoints(rect)
    box = [
        box[0][0], box[0][1],
        box[1][0], box[1][1],
        box[2][0], box[2][1],
        box[3][0], box[3][1],
        angle
    ]

    return box
