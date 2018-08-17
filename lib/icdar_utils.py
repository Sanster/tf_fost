import numpy as np
import cv2

from lib import cv2_utils

MLT_IGNORE_TEXT = '###'
MLT_IGNORE_LANGUAGES = ['None', 'Symbols']


def get_ltrb(line):
    """
    :param line: (x1,y1,x2,y2,x3,y3,x4,y4)
    :return: (xmin, ymin, xmax, ymax)
    """
    xmin = min(line[0], line[6])
    ymin = min(line[1], line[3])
    xmax = max(line[2], line[4])
    ymax = max(line[5], line[7])

    return np.asarray([xmin, ymin, xmax, ymax])


def get_img_scale(img, scale, max_scale):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(scale) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_scale:
        im_scale = float(max_scale) / float(im_size_max)

    return im_scale


def parse_mlt_line(pnts, im_scale=1):
    """
    :param pnts:
        "x1,y1,x2,y2,x3,y3,x4,y4,language,text"
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return:
        [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], language, text
    """
    splited_line = pnts.split(',')
    if len(splited_line) > 10:
        splited_line[-1] = ','.join(splited_line[10:])

    for i in range(8):
        splited_line[i] = int(int(splited_line[i]) * im_scale)

    pnts = np.asarray([[splited_line[0], splited_line[1]],
                       [splited_line[2], splited_line[3]],
                       [splited_line[4], splited_line[5]],
                       [splited_line[6], splited_line[7]]]).astype(np.float64)

    return pnts, splited_line[-2], splited_line[-1]


def load_mlt_gt(gt_path):
    """
    :param gt_path:
    :return: [
            [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],language,text,ignore],
            ...
        ]
    """
    with open(gt_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    parsed_lines = [parse_mlt_line(line) for line in lines]

    out = []
    for line in parsed_lines:
        ignore = False
        if line[1] in MLT_IGNORE_LANGUAGES or line[2] == MLT_IGNORE_TEXT:
            ignore = True
        out.append([line[0], line[1], line[2], ignore])

    return out


if __name__ == "__main__":
    img_path = '/home/cwq/data/MLT2017/val/img_757.jpg'
    gt_path = '/home/cwq/data/MLT2017/val_gt/gt_img_757.txt'
    fixed_height = 32

    aa = load_mlt_gt(gt_path)
    print(aa)

    with open(gt_path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    parsed_lines = [parse_mlt_line(line) for line in lines]
    rboxs = [cv2_utils.get_min_area_rect(line[0]) for line in parsed_lines]

    img = cv2.imread(img_path)

    for line in rboxs:
        img = cv2_utils.draw_four_vectors(img, line[:8])

    w = img.shape[1]
    h = img.shape[0]

    for rbox in rboxs:
        print(rbox)

        cx = (rbox[0] + rbox[4]) / 2
        cy = (rbox[1] + rbox[5]) / 2
        angle = rbox[8]

        roi_w = int(abs(rbox[0] - rbox[6]))
        roi_h = int(abs(rbox[1] - rbox[3]))

        scale = roi_h / fixed_height
        roi_scale_w = int(roi_w / scale)

        M = cv2.getRotationMatrix2D((cx, cy), angle, 1 / scale)

        t = cv2.createAffineTransformer(True)

        scale_h = int(h / scale)
        scale_w = int(w / scale)

        affine_img = cv2.warpAffine(img, M, (scale_w, scale_h))
        cv2.imshow('test', affine_img)
        cv2.waitKey()

    cv2.imshow('test', img)
    cv2.waitKey()
