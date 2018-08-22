from scipy.spatial import distance as dist
import math
import cv2
import numpy as np


def draw_four_vectors(img, line, color=(0, 255, 0)):
    """
    :param line: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    img = cv2.line(img, (line[0][0], line[0][1]), (line[1][0], line[1][1]), color)
    img = cv2.line(img, (line[1][0], line[1][1]), (line[2][0], line[2][1]), color)
    img = cv2.line(img, (line[2][0], line[2][1]), (line[3][0], line[3][1]), color)
    img = cv2.line(img, (line[3][0], line[3][1]), (line[0][0], line[0][1]), color)
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
    :param line: numpy array [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :return [[x1,y1],[x2,y2],[x3,y3],[x4,y4], angle]
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    """
    rect = cv2.minAreaRect(line)

    # https://stackoverflow.com/questions/15956124/minarearect-angles-unsure-about-the-angle-returned
    # opencv 返回的 angle，取值范围为 [-90, 0)
    # -90 代表没有旋转，有两条边水平，两条边竖直,角度增加代表顺时针旋转

    # https://blog.csdn.net/a553654745/article/details/45743063
    # 与x轴平行的方向为角度为0，逆时针旋转角度为负，顺时针旋转角度为正，
    # angle 是水平轴（x轴）逆时针旋转，与碰到的第一个边的夹角，所以 angle 一定是负的
    angle = np.deg2rad(rect[2] + 45.)

    # 获得最小 rotate rect 的四个角点
    box = cv2.boxPoints(rect)
    box = clockwise_points(box)

    box = [
        np.asarray([[box[0][0], box[0][1]],
                    [box[1][0], box[1][1]],
                    [box[2][0], box[2][1]],
                    [box[3][0], box[3][1]]]).astype(np.int64),
        angle
    ]

    return box


# https://www.pyimagesearch.com/2016/03/21/ordering-coordinates-clockwise-with-python-and-opencv/
def clockwise_points(pnts):
    """
    sort clockwise
    :param pnts: numpy array [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :return: numpy array [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    """
    # sort the points based on their x-coordinates
    xSorted = pnts[np.argsort(pnts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    # now, sort the left-most coordinates according to their
    # y-coordinates so we can grab the top-left and bottom-left
    # points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    # now that we have the top-left coordinate, use it as an
    # anchor to calculate the Euclidean distance between the
    # top-left and right-most points; by the Pythagorean
    # theorem, the point with the largest distance will be
    # our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="float32")
