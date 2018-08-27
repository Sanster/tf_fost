import glob
import time

import cv2

import tensorflow as tf
import numpy as np

from lib import lanms
from lib.config import load_config
from lib.cv2_utils import draw_four_vectors, draw_bounding_box
from nets.resnet_v2 import ResNetV2
from parse_args import parse_args

DEBUG = False


def main(args):
    cfg = load_config(args.cfg_name)

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    with tf.Session(config=tfconfig) as sess:
        model = ResNetV2(cfg, 100, is_training=False)
        model.create_architecture()

        saver = tf.train.Saver()
        print('Checkpoint dir: %s' % args.ckpt_dir)
        ckpt = tf.train.get_checkpoint_state(args.ckpt_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Loaded checkpoint {:s}'.format(ckpt.model_checkpoint_path))

        fetches = [
            model.F_score,
            model.F_geometry
        ]

        im_files = glob.glob(args.infer_dir + "/*.*")
        for im_file in im_files:
            img = cv2.imread(im_file)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64)
            img -= np.array([[[cfg.R_MEAN, cfg.G_MEAN, cfg.B_MEAN]]])

            timer = {'net': 0, 'restore': 0, 'nms': 0}
            start = time.time()
            score, geometry = sess.run(fetches, {model.input_images: [img], model.input_is_training: False})
            timer['net'] = time.time() - start

            img = img.astype(np.float64) + np.array([[[cfg.R_MEAN, cfg.G_MEAN, cfg.B_MEAN]]])
            img = img.astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            boxes, timer = detect(img, score, geometry, timer)
            print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                im_file, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))

            if boxes is not None:
                boxes = boxes[:, :8].reshape((-1, 4, 2))
                for box in boxes:
                    cv2.polylines(img, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                                  thickness=2)

            cv2.imshow('result', img)
            k = cv2.waitKey()
            if k == 27:  # ESC
                exit(-1)


def detect(img, score_map, geo_map, timer, score_map_thresh=0.8, box_thresh=0.1, nms_thres=0.2):
    '''
    restore text boxes from score map and geo map
    :param img: only for save test image
    :param score_map: [batch_size, height, width, 1]
    :param geo_map: [batch_size, height, width, 5]
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # 获得大于 score_map_thresh 的 score 的索引位置
    # [None, 2]， dim 1 上，第一个数代表 y 轴，第二个代表 x 轴
    xy_text = np.argwhere(score_map > score_map_thresh)

    # argsort 按照 xy_text y 轴坐标从小到大排序，并返回排完序的索引
    # 根据排完序的索引获得排完序的 xy 坐标
    xy_text = xy_text[np.argsort(xy_text[:, 0])]

    # 获得有效区域的 geo 信息
    geometry = geo_map[xy_text[:, 0], xy_text[:, 1], :]

    if DEBUG:
        # draw all rbox before restore
        debug_xy_text = xy_text * 4
        rboxes = geometry[:, :4]
        for i, yx in enumerate(debug_xy_text):
            trbl = rboxes[i]
            y = yx[0]
            x = yx[1]
            box = (int(x - trbl[3]), int(y - trbl[0]), int(x + trbl[1]), int(y + trbl[2]))
            draw_bounding_box(img, box, color=(0, 0, 188))
        cv2.imshow('rboxes', img)
        cv2.waitKey()

    start = time.time()
    text_box_restored = restore_rectangle_rbox(xy_text, geometry)

    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    print('{} text boxes after nms'.format(boxes.shape[0]))
    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer


def restore_rectangle_rbox(yx_pos, geometry):
    """
    :param yx_pos: selected xy position, score > score threshold
    :param geometry: selected gep [N, 5]
    :return: rotated box of text line[N, 4, 2]
    """
    if yx_pos.shape[0] <= 0:
        return np.zeros((0, 4, 2))

    # 把 x 轴放到 dim 1 的第一个位置
    position = yx_pos[:, ::-1]

    # cnn 的 total stride 为 4，这里恢复到原图尺寸
    position = position * 4

    trbls = geometry[:, :4]
    angles = geometry[:, 4]

    rboxes = []
    for i, trbl in enumerate(trbls):
        angle = angles[i]
        x = position[i][0]
        y = position[i][1]
        # 根据像素点到 trbl 的距离恢复出 bounding boxj
        xmin = x - trbl[3]
        xmax = x + trbl[1]
        ymin = y - trbl[0]
        ymax = y + trbl[2]

        cx = (xmax + xmin) / 2
        cy = (ymin + ymax) / 2

        angle = np.rad2deg(angle)

        M = cv2.getRotationMatrix2D((cx, cy), -angle, 1.0)
        pnts = np.array([
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax)
        ])
        pnts = np.array([pnts])
        new_pnts = cv2.transform(pnts, M)[0]
        rboxes.append(new_pnts)

    rboxes = np.asarray(rboxes)
    return rboxes


if __name__ == "__main__":
    args = parse_args(infer=True)
    main(args)
