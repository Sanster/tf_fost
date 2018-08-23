import glob
import math
import os

import tensorflow as tf
from tensorflow.python.framework import dtypes
import cv2
import numpy as np

from lib import cv2_utils
from lib.cv2_utils import get_min_area_rect, clockwise_points
from lib.icdar_utils import load_ic15_gt, load_mlt_gt, get_ltrb, get_ltrb_by4vec

# noinspection PyMethodMayBeStatic
from lib.label_converter import LabelConverter
from lib.utils import clip

DEBUG = False


class Dataset:
    """
    Use tensorflow Dataset api to load images in parallel
    """

    def __init__(self,
                 cfg,
                 img_dir,
                 gt_dir,
                 converter,
                 batch_size,
                 num_parallel_calls=4,
                 shuffle=True):
        self.cfg = cfg
        self.img_dir = img_dir
        self.converter = converter
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.shuffle = shuffle

        self.base_names = self._get_base_name(self.img_dir)

        self.size = len(self.base_names)
        self.num_batches = math.ceil(self.size / self.batch_size)

        dataset = self._create_dataset()

        iterator = dataset.make_initializable_iterator()
        self.next_batch = iterator.get_next()
        self.init_op = iterator.initializer

        self.pixel_mean = np.array([[[self.cfg.R_MEAN, self.cfg.G_MEAN, self.cfg.B_MEAN]]])

    def _get_base_name(self, img_dir):
        img_paths = glob.glob(img_dir + '/*.*')
        base_names = [os.path.basename(p) for p in img_paths]
        return base_names

    def get_next_batch(self, sess):
        # imgs, score_maps, geo_maps, training_mask, text_roi_count, affine_matrixs, affine_rects, labels, img_paths = sess.run(
        #     self.next_batch)
        imgs, score_maps, geo_maps, training_mask = sess.run(
            self.next_batch)

        # if DEBUG:
        #     print("text_roi_count")
        #     print(text_roi_count)
        #     print("affine_matrixs shape")
        #     print(affine_matrixs.shape)
        #     print("affine_rects shape")
        #     print(affine_rects.shape)

        # batch_encoded_labels = []
        # for img_labels in labels:
        #     decoded_labels = [l.decode() for l in img_labels]
        #     # remove padded labels
        #     decoded_labels = list(filter(lambda x: x, decoded_labels))
        #     encoded_labels = self.converter.encode_list(decoded_labels)
        #     batch_encoded_labels.extend(encoded_labels)
        #
        # sparse_labels = self._sparse_tuple_from_label(batch_encoded_labels)
        #
        # batch_img_paths = []
        # for p in img_paths:
        #     batch_img_paths.append(p[0])

        # return imgs, score_maps, geo_maps, training_mask, text_roi_count, affine_matrixs, affine_rects, sparse_labels, batch_img_paths
        return imgs, score_maps, geo_maps, training_mask

    def _create_dataset(self):
        tf_base_names = tf.convert_to_tensor(self.base_names, dtype=dtypes.string)
        d = tf.data.Dataset.from_tensor_slices(tf_base_names)
        # d = tf.data.Dataset.from_generator(self._input_py_parser,
        #                                    output_types=(
        #                                        tf.uint8, tf.uint8, tf.float32, tf.float64, tf.int32,
        #                                        tf.string))

        if self.shuffle:
            d = d.shuffle(buffer_size=self.size)

        d = d.map(lambda base_name: tf.py_func(self._input_py_parser, [base_name],
                                               [tf.float32, tf.float32, tf.float32, tf.uint8]))

        # d = d.map(lambda base_name: tf.py_func(self._input_py_parser, [base_name],
        #                                        [tf.float32, tf.float32, tf.float32, tf.uint8, tf.int64, tf.float64,
        #                                         tf.int32, tf.string, tf.string]))

        d = d.batch(self.batch_size)
        # d = d.padded_batch(self.batch_size,
        #                    padded_shapes=([self.cfg.train.croped_img_size, self.cfg.train.croped_img_size, 3],
        #                                   [160, 160, 1],
        #                                   [160, 160, 5],
        #                                   [160, 160, 1],
        #                                   [1],
        #                                   [None, 2, 3],
        #                                   [None, 4],
        #                                   [None],
        #                                   [None]))
        d = d.prefetch(buffer_size=2)
        return d

    def _input_py_parser(self, base_name):
        """
        按照论文当中进行 data augmentation 的方法进行处理
        - TODO: 将图片的长边 resize 到 640 ~ 2560 之间
        - TODO: 图片随机旋转 -10 ~ 10 度
        - TODO: 宽度保持不变，图片的高度随机缩放 0.8 ~ 1.2
        - TODO: random crop 出 640 x 640 的图片，这一步应该要保证 crop 时不能把文字截断
        """
        base_name = base_name.decode()
        gt_name = 'gt_%s.txt' % base_name.split('.')[0]
        img_path = os.path.join(self.img_dir, base_name)
        gt_path = os.path.join(self.gt_dir, gt_name)

        if DEBUG:
            print(img_path)

        # gts = load_mlt_gt(gt_path)
        gts = load_ic15_gt(gt_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img -= self.pixel_mean

        # long_side_length = np.random.randint(640, 2560)

        # 放大的倍数，e.g 放大 1.2 倍，放大 0.5 倍(即缩小2倍)
        # scale = long_side_length / max(img.shape[0], img.shape[1])
        # img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        xscale = self.cfg.train.croped_img_size / img.shape[1]
        yscale = self.cfg.train.croped_img_size / img.shape[0]

        for gt in gts:
            # scale
            gt[0][:, 0] = (gt[0][:, 0] * xscale).astype(np.int32)
            gt[0][:, 1] = (gt[0][:, 1] * yscale).astype(np.int32)

            # ignore small height text
            rbox = get_min_area_rect(gt[0])
            roi_h = int(np.linalg.norm(rbox[0][1] - rbox[0][2]))
            if roi_h < self.cfg.min_text_height:
                gt[-1] = True

        # TODO: Use random crop
        img = cv2.resize(img, (self.cfg.train.croped_img_size, self.cfg.train.croped_img_size),
                         interpolation=cv2.INTER_AREA)

        # if min(img.shape[0], img.shape[1]) < self.cfg.train.croped_img_size:
        #     img_croped = img
        # else:
        #     img_croped, gts = self._crop_img(img, gts)

        if DEBUG:
            for gt in gts:
                gt[0] = gt[0].astype(np.int32)
                img = cv2_utils.draw_four_vectors(img, gt[0])
            recoverd_img = img + self.pixel_mean
            recoverd_img = recoverd_img.astype(np.uint8)
            bgr = cv2.cvtColor(recoverd_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite('test.jpg', bgr)

        score_map, geo_map, training_mask = self.generate_rbox(img.shape, gts)

        valid_text_count = 0
        for gt in gts:
            ignore = gt[-1]
            if not ignore:
                # Ground true label data for CRNN
                encoded_label = self.converter.encode(gt[-2])
                if len(encoded_label) == 0:
                    continue
                valid_text_count += 1

        labels = [""]
        affine_matrixs = np.zeros((valid_text_count, 2, 3), np.float64)
        affine_rects = np.zeros((valid_text_count, 4), np.int32)

        count = 0
        for gt in gts:
            ignore = gt[-1]
            if not ignore:
                # Ground true label data for CRNN
                encoded_label = self.converter.encode(gt[-2])
                if len(encoded_label) == 0:
                    continue

                labels.append(gt[-2])
                # 计算访射变换
                M, rect = self._get_affine_M(gt[0], self.cfg.train.share_conv_stride,
                                             self.cfg.train.roi_rotate_fix_height)
                affine_matrixs[count][:] = M
                affine_rects[count][:] = rect
                count += 1

        if DEBUG:
            print(affine_matrixs.shape)
            print(affine_rects.shape)

        # return img, score_map, geo_map, training_mask, [valid_text_count], affine_matrixs, affine_rects, labels, [
        #     img_path]
        return img, score_map, geo_map, training_mask

    def _get_affine_M2(self):
        # TODO use method in paper to cal M
        pass

    def _get_affine_M(self, poly, stride=4, fix_height=8):
        """
        先将 roi 的中心移至图片中心，再进行旋转
        :param poly: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        :param stride: share feature 最后输出的大小为原图的 1/4, 用来计算 gt rbox 映射到 feature map 层的坐标
        :param fix_height: 论文中将 share feature 上对应的 gt 区域都访射变换到高度为8
        :return:
            M: 仿射变换矩阵, 3x3 , float64
            pnt_affined: (4,2) rotate box 在最后一层 feature map 上经过仿射变换后的坐标, float64
        """
        rbox = get_min_area_rect(poly)

        # 最后一层 feature map 上的坐标按照 stride 缩小
        rbox[0] = rbox[0] / stride

        rect = rbox[0]

        # TODO: rewrite this
        cx = 80
        cy = 80
        angle = rbox[1]

        # 最后一层 feature map 上 roi 的长宽
        roi_w = int(np.linalg.norm(rbox[0][0] - rbox[0][1]))
        roi_h = int(np.linalg.norm(rbox[0][1] - rbox[0][2]))

        # 放大的倍数，e.g 放大 1.2 倍，放大 0.5 倍(即缩小2倍)
        scale = fix_height / roi_h
        roi_scale_w = int(roi_w * scale)

        src = np.float32([
            rect[0], rect[1], rect[2]
        ])

        dst = np.float32([
            [int(cx - roi_scale_w / 2), int(cy - fix_height / 2)],
            [int(cx + roi_scale_w / 2), int(cy - fix_height / 2)],
            [int(cx + roi_scale_w / 2), int(cy + fix_height / 2)],
        ])

        # 返回 2 x 3 的矩阵, float64
        M = cv2.getAffineTransform(src, dst)

        # (4,2) float64
        pnts_affined = np.int32([
            [int(cx - roi_scale_w / 2), int(cy - fix_height / 2)],
            [int(cx + roi_scale_w / 2), int(cy - fix_height / 2)],
            [int(cx + roi_scale_w / 2), int(cy + fix_height / 2)],
            [int(cx - roi_scale_w / 2), int(cy + fix_height / 2)],
        ])
        pnts_affined = clockwise_points(pnts_affined)
        ltrb = get_ltrb_by4vec(pnts_affined)
        rect = ltrb.astype(np.int32)

        # TODO: 太粗暴了？
        # make rect height is 8
        if rect[3] - rect[1] != fix_height:
            rect[3] = rect[1] + fix_height

        # print("before clip")
        # print(rect)
        rect = clip(rect, (160, 160))
        rect = rect.astype(np.int32)
        # print("after clip")
        # print(rect)
        return M, rect

    def _crop_img(self, img, gts):
        """
        使用窗口在图片上滑动，窗口不能把文字截断，窗口必须包含文字
        :param img:
        :param gts: [((x1,y1,x2,y2,x3,y3,x4,y4),language,text,ignore)]
        :return:
        """
        # 先根据 polys 计算出 bounding box
        ltrb_gts = [(get_ltrb(g[0]).astype(np.int32), g[3]) for g in gts]

        # 因为滑窗的尺寸是定的，所以这里只计算滑窗左上角点的取值范围
        # 用来记录图像上的每一个像素是否可以所谓 left-top 点
        corner_map = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

        # bbox 区域不能作为 left-top
        for bbox, ignore in ltrb_gts:
            if not ignore:
                corner_map[bbox[1]:bbox[3], bbox[0]: bbox[2]] = 0

        if DEBUG:
            cv2.imwrite('test.jpg', corner_map * 255)

        return img, gts

    def generate_rbox(self, im_size, gts):
        """
        :param im_size:
        :param gts:
        :return:
            score_map: poly 所占区域的文字区域为 1，其他地方为 0. [height, width]
            geo_map: poly 中 每一个像素点到 minAreaRect 的四边的距离. [height, width, 5]
                     如果像素点不在 poly 中则都为 0
            training_mask: 计算 detect loss 时忽略文字较小或者模糊的区域
        """
        w = im_size[1]
        h = im_size[0]

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 5), dtype=np.float32)
        training_mask = np.ones((h, w), dtype=np.uint8)

        if DEBUG:
            unshrink_score_map = np.zeros((h, w), dtype=np.uint8)

        for idx, gt in enumerate(gts):
            poly = gt[0]
            ignore = gt[-1]
            if ignore:
                cv2.fillPoly(training_mask, [poly], 0)
                if DEBUG:
                    cv2.imwrite('training_mask.jpg', training_mask * 255)
                continue

            # score map
            shrinked_poly = self.shrink_poly(poly.copy()).astype(np.int32)

            cv2.fillPoly(score_map, [shrinked_poly], 1)

            if DEBUG:
                cv2.fillPoly(unshrink_score_map, [poly], 1)
                cv2.imwrite('score_map.jpg', score_map * 255)
                cv2.imwrite('unshrink_score_map.jpg', unshrink_score_map * 255)

            cv2.fillPoly(poly_mask, [shrinked_poly], idx + 1)
            xy_in_poly = np.argwhere(poly_mask == (idx + 1))

            rbox = get_min_area_rect(poly)

            self._calculate_geo_map(geo_map, xy_in_poly, rbox[0], rbox[1])

            # 可视化距离 map，越亮代表距离越远
            if DEBUG:
                cv2.imwrite('geo_map_lt_rt.jpg', geo_map[::, ::, 0])
                cv2.imwrite('geo_map_rt_rb.jpg', geo_map[::, ::, 1])
                cv2.imwrite('geo_map_rb_lb.jpg', geo_map[::, ::, 2])
                cv2.imwrite('geo_map_lb_lt.jpg', geo_map[::, ::, 3])

        # TODO: why this is right?
        score_map = score_map[::4, ::4, np.newaxis].astype(np.float32)
        geo_map = geo_map[::4, ::4, :].astype(np.float32)
        training_mask = training_mask[::4, ::4, np.newaxis].astype(np.uint8)

        # TODO: rewrite this
        score_map = score_map[:160, :160, :]
        geo_map = geo_map[:160, :160, :]
        training_mask = training_mask[:160, :160, :]

        return score_map, geo_map, training_mask

    # https://github.com/argman/EAST/issues/160
    def _calculate_geo_map(self, geo_map, xy_in_poly, rectange, rotate_angle):
        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        height = (self.point_dist_to_line(p0_rect, p1_rect, p2_rect) + self.point_dist_to_line(p0_rect, p1_rect,
                                                                                               p3_rect)) / 2
        width = (self.point_dist_to_line(p3_rect, p0_rect, p1_rect) + self.point_dist_to_line(p3_rect, p0_rect,
                                                                                              p2_rect)) / 2

        ys = xy_in_poly[:, 0]
        xs = xy_in_poly[:, 1]
        num_points = xy_in_poly.shape[0]
        top_distance_tmp = self.point_dist_to_line(np.tile(p0_rect, (num_points, 1)),
                                                   np.tile(p1_rect, (num_points, 1)),
                                                   xy_in_poly[:, ::-1])
        geo_map[ys, xs, 0] = top_distance_tmp
        right_distance_tmp = self.point_dist_to_line(np.tile(p1_rect, (num_points, 1)),
                                                     np.tile(p2_rect, (num_points, 1)),
                                                     xy_in_poly[:, ::-1])
        geo_map[ys, xs, 1] = right_distance_tmp
        geo_map[ys, xs, 2] = height - top_distance_tmp
        geo_map[ys, xs, 3] = width - right_distance_tmp
        geo_map[ys, xs, 4] = rotate_angle
        return geo_map

    def point_dist_to_line(self, p1, p2, p3):
        if len(p3.shape) < 2:
            return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)
        else:
            points = p3.shape[0]
            cross_product = np.cross(p2 - p1, p1 - p3)
            cross_product = np.resize(cross_product, [points, 1])
            return np.linalg.norm(cross_product, axis=1) / np.linalg.norm(p2 - p1, axis=1)

    def _sparse_tuple_from_label(self, sequences, default_val=-1, dtype=np.int32):
        """Create a sparse representention of x.
        Args:
            sequences: a list of lists of type dtype where each element is a sequence
                      encode label, e.g: [2,44,11,55]
            default_val: value should be ignored in sequences
        Returns:
            A tuple with (indices, values, shape)
        """
        indices = []
        values = []

        for n, seq in enumerate(sequences):
            seq_filtered = list(filter(lambda x: x != default_val, seq))
            indices.extend(zip([n] * len(seq_filtered), range(len(seq_filtered))))
            values.extend(seq_filtered)

        indices = np.asarray(indices, dtype=np.int32)
        values = np.asarray(values, dtype=dtype)

        if len(indices) == 0:
            shape = np.asarray([len(sequences), 0], dtype=np.int32)
        else:
            shape = np.asarray([len(sequences), np.asarray(indices).max(0)[1] + 1], dtype=np.int32)

        return indices, values, shape

    def shrink_poly(self, poly, R=0.3):
        """
        fit a poly inside the origin poly, maybe bugs here...
        used for generate the score map
        :param poly: the text poly
        :param R: shrink radio
        :return: the shrinked poly
        """
        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))

        # find the longer pair
        if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
                np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
            # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
            ## p0, p3
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
        else:
            ## p0, p3
            # print poly
            theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
            poly[0][0] += R * r[0] * np.sin(theta)
            poly[0][1] += R * r[0] * np.cos(theta)
            poly[3][0] -= R * r[3] * np.sin(theta)
            poly[3][1] -= R * r[3] * np.cos(theta)
            ## p1, p2
            theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
            poly[1][0] += R * r[1] * np.sin(theta)
            poly[1][1] += R * r[1] * np.cos(theta)
            poly[2][0] -= R * r[2] * np.sin(theta)
            poly[2][1] -= R * r[2] * np.cos(theta)
            ## p0, p1
            theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
            poly[0][0] += R * r[0] * np.cos(theta)
            poly[0][1] += R * r[0] * np.sin(theta)
            poly[1][0] -= R * r[1] * np.cos(theta)
            poly[1][1] -= R * r[1] * np.sin(theta)
            ## p2, p3
            theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
            poly[3][0] += R * r[3] * np.cos(theta)
            poly[3][1] += R * r[3] * np.sin(theta)
            poly[2][0] -= R * r[2] * np.cos(theta)
            poly[2][1] -= R * r[2] * np.sin(theta)
        return poly


if __name__ == "__main__":
    converter = LabelConverter(chars_file='./data/chars/eng.txt')

    from lib.config import load_config
    from nets.resnet_v2 import ResNetV2

    cfg = load_config()
    ds = Dataset(
        cfg,
        # img_dir='/home/cwq/data/MLT2017/val',
        # gt_dir='/home/cwq/data/MLT2017/val_gt',
        img_dir='/home/cwq/data/ocr/IC15/ch4_training_images',
        gt_dir='/home/cwq/data/ocr/IC15/ch4_training_localization_transcription_gt',
        converter=converter,
        batch_size=6,
        num_parallel_calls=1,
        shuffle=False)

    with tf.Session() as sess:
        ds.init_op.run()
        ds.get_next_batch(sess)

    # model = ResNetV2(cfg, converter.num_classes)
    # model.create_architecture()
    # with tf.Session() as sess:
    #     ds.init_op.run()
    #     imgs, score_maps, geo_maps, affine_matrixs, affine_rects, labels = ds.get_next_batch(sess)
    #
    #     feed = {
    #         model.input_images: imgs,
    #         model.input_score_maps: score_maps,
    #         model.input_geo_maps: geo_maps,
    #         model.input_affine_matrixs: affine_matrixs,
    #         model.input_affine_rects: affine_rects,
    #         model.input_labels: labels,
    #         model.is_training: True
    #     }
    #     sess.run([model.train_op], feed_dict=feed)
