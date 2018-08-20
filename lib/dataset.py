import glob
import math
import os

import tensorflow as tf
from tensorflow.python.framework import dtypes
import cv2
import numpy as np

from lib import cv2_utils
from lib.cv2_utils import get_min_area_rect
from lib.icdar_utils import load_mlt_gt, get_ltrb
from lib.config import cfg

# noinspection PyMethodMayBeStatic
from lib.label_converter import LabelConverter


class Dataset:
    """
    Use tensorflow Dataset api to load images in parallel
    """

    def __init__(self,
                 img_dir,
                 gt_dir,
                 converter,
                 batch_size,
                 num_parallel_calls=4,
                 shuffle=True):
        self.img_dir = img_dir
        self.converter = converter
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.num_parallel_calls = num_parallel_calls
        self.shuffle = shuffle

        self.base_names = self._get_base_name(self.img_dir)

        self.size = len(self.base_names)
        self.num_batches = math.ceil(self.size / self.batch_size)

        dataset = self._create_dataset(self.base_names)

        iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)

        self.next_batch = iterator.get_next()
        self.init_op = iterator.make_initializer(dataset)

    def _get_base_name(self, img_dir):
        img_paths = glob.glob(img_dir + '/*.*')
        base_names = [os.path.basename(p) for p in img_paths]
        return base_names

    def get_next_batch(self, sess):
        imgs, score_maps, geo_maps, alline_matrixs, alline_pnts, labels = sess.run(self.next_batch)

        sparse_labels = []
        for img_labels in labels:
            decoded_labels = [l.decode() for l in img_labels]
            print(decoded_labels)
            encoded_labels = self.converter.encode_list(decoded_labels)
            sparse_labels.append(self._sparse_tuple_from_label(encoded_labels))

        return imgs, score_maps, geo_maps, alline_matrixs, alline_pnts, sparse_labels

    def _create_dataset(self, base_names):
        tf_base_names = tf.convert_to_tensor(base_names, dtype=dtypes.string)

        d = tf.data.Dataset.from_tensor_slices(tf_base_names)

        if self.shuffle:
            d = d.shuffle(buffer_size=self.size)

        d = d.map(lambda base_name: tf.py_func(self._input_py_parser, [base_name],
                                               [tf.uint8, tf.uint8, tf.float32, tf.float64, tf.float64, tf.string]))

        d = d.batch(self.batch_size)
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
        print(img_path)

        mlt_gts = load_mlt_gt(gt_path)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # long_side_length = np.random.randint(640, 2560)

        # 放大的倍数，e.g 放大 1.2 倍，放大 0.5 倍(即缩小2倍)
        # scale = long_side_length / max(img.shape[0], img.shape[1])
        # img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        xscale = cfg.TRAIN.CROPED_IMG_SIZE / img.shape[1]
        yscale = cfg.TRAIN.CROPED_IMG_SIZE / img.shape[0]

        for gt in mlt_gts:
            # print(gt[0])
            # print(gt[0].dtype)
            gt[0][:, 0] = (gt[0][:, 0] * xscale).astype(np.int32)
            gt[0][:, 1] = (gt[0][:, 1] * yscale).astype(np.int32)
            # gt[0][:, 1] *= yscale
            # gt[0] = gt[0].astype(np.int32)

        # TODO: Use random crop
        img = cv2.resize(img, (cfg.TRAIN.CROPED_IMG_SIZE, cfg.TRAIN.CROPED_IMG_SIZE), interpolation=cv2.INTER_AREA)

        # for gt in mlt_gts:
        #     gt[0] = gt[0].astype(np.int32)
        #     img = cv2_utils.draw_four_vectors(img, gt[0])
        # cv2.imwrite('test.jpg', img)

        # if min(img.shape[0], img.shape[1]) < cfg.TRAIN.CROPED_IMG_SIZE:
        #     img_croped = img
        # else:
        #     img_croped, mlt_gts = self._crop_img(img, mlt_gts)

        score_map, geo_map = self.generate_rbox(img.shape, mlt_gts)

        # TODO: 计算仿射变换参数

        labels = []
        alline_matrixs = []
        alline_pnts = []
        for gt in mlt_gts:
            ignore = gt[-1]
            if not ignore:
                # Ground true label data for CRNN
                labels.append(gt[-2])

                # 计算访射变换
                M, pnts = self._get_affine_M(gt[0], cfg.TRAIN.SHARE_CONV_STRIDE, cfg.TRAIN.ROI_ROTATE_FIX_HEIGHT)
                alline_matrixs.append(M)
                alline_pnts.append(pnts)

        return img, score_map, geo_map, alline_matrixs, alline_pnts, labels

    def _get_affine_M(self, poly, stride=4, fix_height=8):
        """
        :param poly: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        :param stride: share feature 最后输出的大小为原图的 1/4, 用来计算 gt rbox 映射到 feature map 层的坐标
        :param fix_height: 论文中将 share feature 上对应的 gt 区域都访射变换到高度为8
        :return:
            M: 仿射变换矩阵, 3x3 , float64
            pnt_affined: rotate box 在最后一层 feature map 上经过仿射变换后的坐标, float64
        """
        rbox = get_min_area_rect(poly)

        # 最后一层 feature map 上的坐标按照 stride 缩小
        rbox[0] = rbox[0] / stride

        cx = (rbox[0][0][0] + rbox[0][2][0]) / 2
        cy = (rbox[0][0][1] + rbox[0][2][1]) / 2
        angle = rbox[1]

        # TODO: Is this right??
        # 最后一层 feature map 上 roi 的长宽
        roi_w = int(np.linalg.norm(rbox[0][0] - rbox[0][1]))
        roi_h = int(np.linalg.norm(rbox[0][1] - rbox[0][2]))

        # 放大的倍数，e.g 放大 1.2 倍，放大 0.5 倍(即缩小2倍)
        scale = fix_height / roi_h
        # roi_scale_w = int(roi_w * scale)

        # 返回 2 x 3 的矩阵, float64
        M = cv2.getRotationMatrix2D((cx, cy), angle, scale)

        # (4,2) float64
        pnts_affined = cv2.transform(np.asarray([rbox[0]]), M)[0]
        # print(pnts_affined.shape)

        # Debug: check whether cv2.transform works right
        # src = rbox[0][:3]
        # dst = pnts_affined[0][:3]
        # _M = cv2.getAffineTransform(src.astype(np.float32), dst.astype(np.float32))
        # print("MMMMMMMMMM")
        # print(M)
        # print("__________")
        # print(_M)

        # print("Before stack")
        # print(M.shape)
        # print(M)
        # Tensorflow matrices_to_flat_transforms need 3x3
        # https://www.tensorflow.org/api_docs/python/tf/contrib/image/matrices_to_flat_transforms
        M = np.vstack([M, [0, 0, 1]])
        # print("After stack")
        # print(M.shape)
        # print(M)

        return M, pnts_affined

    def _crop_img(self, img, mlt_gts):
        """
        使用窗口在图片上滑动，窗口不能把文字截断，窗口必须包含文字
        :param img:
        :param mlt_gts: [((x1,y1,x2,y2,x3,y3,x4,y4),language,text,ignore)]
        :return:
        """
        # 先根据 polys 计算出 bounding box
        ltrb_gts = [(get_ltrb(g[0]).astype(np.int32), g[3]) for g in mlt_gts]

        # 因为滑窗的尺寸是定的，所以这里只计算滑窗左上角点的取值范围
        # 用来记录图像上的每一个像素是否可以所谓 left-top 点
        corner_map = np.ones((img.shape[0], img.shape[1]), dtype=np.uint8)

        # bbox 区域不能作为 left-top
        for bbox, ignore in ltrb_gts:
            if not ignore:
                corner_map[bbox[1]:bbox[3], bbox[0]: bbox[2]] = 0

        cv2.imwrite('test.jpg', corner_map * 255)

        return img, mlt_gts

    def generate_rbox(self, im_size, gts):
        """
        TODO: shrink poly
        :param im_size:
        :param gts:
        :return:
            score_map: poly 所占区域的文字区域为 1，其他地方为 0. [height, width]
            geo_map: poly 中 每一个像素点到 minAreaRect 的四边的距离. [height, width, 5]
                     如果像素点不在 poly 中则都为 0
        """
        w = im_size[1]
        h = im_size[0]

        poly_mask = np.zeros((h, w), dtype=np.uint8)
        score_map = np.zeros((h, w), dtype=np.uint8)
        geo_map = np.zeros((h, w, 5), dtype=np.float32)

        for idx, gt in enumerate(gts):
            poly = gt[0]
            ignore = gt[-1]
            if ignore:
                continue

            cv2.fillPoly(score_map, [poly], 1)

            cv2.imwrite('score_map.jpg', score_map * 255)

            cv2.fillPoly(poly_mask, [poly], idx + 1)
            xy_in_poly = np.argwhere(poly_mask == (idx + 1))

            rbox = get_min_area_rect(poly)

            self._calculate_gep_map(geo_map, xy_in_poly, rbox[0], rbox[1])

            # 可视化距离 map，越亮代表距离越远
            cv2.imwrite('geo_map_lt_rt.jpg', geo_map[::, ::, 0])
            cv2.imwrite('geo_map_rt_rb.jpg', geo_map[::, ::, 1])
            cv2.imwrite('geo_map_rb_lb.jpg', geo_map[::, ::, 2])
            cv2.imwrite('geo_map_lb_lt.jpg', geo_map[::, ::, 3])

        return score_map, geo_map

    # https://github.com/argman/EAST/issues/160
    def _calculate_gep_map(self, geo_map, xy_in_poly, rectange, rotate_angle):
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


if __name__ == "__main__":
    converter = LabelConverter(chars_file='./data/chars/eng.txt')

    ds = Dataset(
        img_dir='/home/cwq/data/MLT2017/val',
        gt_dir='/home/cwq/data/MLT2017/val_gt',
        converter=converter,
        batch_size=1,
        num_parallel_calls=1,
        shuffle=False)

    with tf.Session() as sess:
        ds.init_op.run()
        ds.get_next_batch(sess)
