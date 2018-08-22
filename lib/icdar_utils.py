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


def get_ltrb_by4vec(line):
    """
    :param line: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
    :return: (xmin, ymin, xmax, ymax)
    """
    _line = [
        line[0][0], line[0][1],
        line[1][0], line[1][1],
        line[2][0], line[2][1],
        line[3][0], line[3][1]
    ]

    return get_ltrb(_line)


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
                       [splited_line[6], splited_line[7]]]).astype(np.int32)

    return pnts, splited_line[-2], splited_line[-1]


def load_mlt_gt(gt_path, include_ignore=True):
    """
    :param gt_path:
    :param include_ignore:
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

        if include_ignore:
            out.append([line[0], line[1], line[2], ignore])
        else:
            if not ignore:
                out.append([line[0], line[1], line[2], ignore])

    return out


def parse_ic15_line(pnts, im_scale=1):
    """
    :param pnts:
        "x1,y1,x2,y2,x3,y3,x4,y4,text"
        矩形四点坐标的顺序： left-top, right-top, right-bottom, left-bottom
    :return:
        [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text
    """
    splited_line = pnts.split(',')
    if len(splited_line) > 10:
        splited_line[-1] = ','.join(splited_line[10:])

    for i in range(8):
        splited_line[i] = int(int(splited_line[i]) * im_scale)

    pnts = np.asarray([[splited_line[0], splited_line[1]],
                       [splited_line[2], splited_line[3]],
                       [splited_line[4], splited_line[5]],
                       [splited_line[6], splited_line[7]]]).astype(np.int32)

    return pnts, splited_line[-1]


def load_ic15_gt(gt_path, include_ignore=True):
    """
    :param gt_path:
    :param include_ignore:
    :return: [
            [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],text,ignore],
            ...
        ]
    """
    # utf-8-sig encode with BOM
    with open(gt_path, 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    parsed_lines = [parse_ic15_line(line) for line in lines]

    out = []
    for line in parsed_lines:
        ignore = False
        if line[1] == MLT_IGNORE_TEXT:
            ignore = True

        if include_ignore:
            out.append([line[0], line[1], ignore])
        else:
            if not ignore:
                out.append([line[0], line[1], ignore])

    return out


if __name__ == "__main__":
    img_path = '/home/cwq/data/MLT2017/val/img_757.jpg'
    gt_path = '/home/cwq/data/MLT2017/val_gt/gt_img_757.txt'

    fixed_height = 32

    parsed_lines = load_mlt_gt(gt_path)

    rboxs = [cv2_utils.get_min_area_rect(line[0]) for line in parsed_lines]

    img = cv2.imread(img_path)

    for line in rboxs:
        img = cv2_utils.draw_four_vectors(img, line[0])

    w = img.shape[1]
    h = img.shape[0]

    ori_img = img.copy()
    for rbox in rboxs:
        # print(rbox)
        #
        # cy = int((rbox[0][0][1] + rbox[0][2][1]) / 2)
        # cx = int((rbox[0][0][0] + rbox[0][2][0]) / 2)
        #
        # ori_img = cv2.circle(ori_img, (cx, cy), radius=5, color=(0, 0, 255))
        #
        # angle = rbox[1]
        #
        # roi_w = int(np.linalg.norm(rbox[0][0] - rbox[0][1]))
        # roi_h = int(np.linalg.norm(rbox[0][1] - rbox[0][2]))
        #
        # scale = fixed_height / roi_h
        # roi_scale_w = int(roi_w * scale)
        #
        # M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
        #
        # scale_h = int(h * scale)
        # scale_w = int(w * scale)
        #
        # affine_img = cv2.warpAffine(img, M, (w, h))
        #
        # # 在 warpAffine 后的图片上截取高度为 32 的区域
        # pnts_affined = cv2.transform(np.asarray([rbox[0]]), M)[0]
        # affine_img = cv2_utils.draw_four_vectors(affine_img, pnts_affined, color=(255, 0, 0))

        print(rbox)

        angle = rbox[1]
        print(angle)
        rect = rbox[0]
        roi_cy = int((rbox[0][0][1] + rbox[0][2][1]) / 2)
        roi_cx = int((rbox[0][0][0] + rbox[0][2][0]) / 2)
        cx = int(w / 2)
        cy = int(h / 2)

        roi_w = int(np.linalg.norm(rbox[0][0] - rbox[0][1]))
        roi_h = int(np.linalg.norm(rbox[0][1] - rbox[0][2]))

        scale = fixed_height / roi_h
        roi_scale_w = int(roi_w * scale)

        src = np.float32([
            rect[0], rect[1], rect[2]
        ])

        dst = np.float32([
            [int(cx - roi_scale_w / 2), int(cy - fixed_height / 2)],
            [int(cx + roi_scale_w / 2), int(cy - fixed_height / 2)],
            [int(cx + roi_scale_w / 2), int(cy + fixed_height / 2)],
        ])

        M = cv2.getAffineTransform(src, dst)

        affine_img = cv2.warpAffine(img, M, (w, h))

        # 在 warpAffine 后的图片上截取高度为 32 的区域
        pnts_affined = cv2.transform(np.asarray([rbox[0]]), M)[0]
        affine_img = cv2_utils.draw_four_vectors(affine_img, pnts_affined, color=(255, 0, 0))

        """
        将 roi 区域移动到图片中心后再旋转
        """
        cv2.imshow('test', affine_img)
        cv2.waitKey()

    cv2.imshow('test', img)
    cv2.waitKey()
