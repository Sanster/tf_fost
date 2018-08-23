# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import tensorflow.contrib.slim as slim

import numpy as np

from abc import abstractmethod

# noinspection PyAttributeOutsideInit,PyProtectedMember,PyMethodMayBeStatic
from lib.roi_rotate_layer import roi_rotate_layer


class Network(object):
    CTC_INVALID_INDEX = -1

    def __init__(self, cfg, num_classes):
        self.cfg = cfg
        self.num_classes = num_classes

    @abstractmethod
    def _image_to_head(self, inputs, is_training):
        """
        Layers from image input to last Conv layer
        """
        raise NotImplementedError

    def create_architecture(self):
        # ResNetV2 要求的, 需要 + 1
        self.input_images = tf.placeholder(tf.float32, shape=[None, 641, 641, 3], name='input_images')
        self.input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
        self.input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
        self.input_training_mask = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_mask')

        # num_of_text_roi for each image
        self.input_text_roi_count = tf.placeholder(tf.int32, shape=[None, 1], name='input_text_roi_count')
        # [batch_size, padded num of text roi, 2, 3]
        self.input_affine_matrixs = tf.placeholder(tf.float64, shape=[None, None, 2, 3], name='input_affine_matrixs')
        # [batch_size, padded num of text roi, 4]
        self.input_affine_rects = tf.placeholder(tf.int32, shape=[None, None, 4], name='input_affine_pnts')

        self.input_text_labels = tf.sparse_placeholder(tf.int32, name='input_text_labels')

        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self._build_network()

        self._build_losses()

        self._build_train_op()

        self.merged_summary = tf.summary.merge_all()

    def _build_network(self):
        # stride 4, channels 320
        self.shared_conv = self._build_share_conv()

        print("Shared conv shape")
        print(self.shared_conv)

        self.F_score, self.F_geometry = self._build_detect_output(self.shared_conv)

        rois, rois_width = self._roi_rotate_layer(self.shared_conv, self.input_text_roi_count,
                                                  self.input_affine_matrixs, self.input_affine_rects,
                                                  self.cfg.train.roi_rotate_fix_height)

        self.seq_len = rois_width

        self.reco_logits = self._build_reco_output(rois, self.seq_len, self.num_classes)

    def _build_losses(self):
        self._build_detect_loss(self.input_score_maps, self.input_geo_maps, self.F_geometry)

        self.reco_ctc_loss = self._build_reco_loss(self.reco_logits, self.input_text_labels, self.seq_len)

        self.regularization_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self.total_loss = self.detect_loss + self.reco_ctc_loss + self.regularization_loss

        tf.summary.scalar('detect_loss', self.detect_loss)
        tf.summary.scalar('detect_cls_loss', self.detect_cls_loss)
        tf.summary.scalar('detect_reg_loss', self.detect_reg_loss)
        tf.summary.scalar('reco_ctc_loss', self.reco_ctc_loss)
        tf.summary.scalar('regularization_loss', self.regularization_loss)
        tf.summary.scalar('total_loss', self.total_loss)

    def _build_detect_output(self, shared_conv):
        self.F_score_logits = slim.conv2d(shared_conv, 1, 1, activation_fn=None, normalizer_fn=None)
        F_score = tf.nn.sigmoid(self.F_score_logits)

        print("F_score shape")
        print(F_score)

        # geo_map 的 ground truth 算的是某像素点到四边的距离占长宽的比例，所以这里要用 sigmoid 限制到 (0,1)
        # TODO: why * text_scale(640)
        geo_map = slim.conv2d(shared_conv, 4, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * 640

        print("geo_map shape")
        print(geo_map)

        # 要限制 angle 的范围为 [-45, 45]，所以要加 sigmoid
        angle_map = (slim.conv2d(shared_conv, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) - 0.5) * np.pi / 2

        # [ -90, 0)
        # angle_map = - slim.conv2d(shared_conv, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None) * np.pi / 2

        print("angle_map shape")
        print(angle_map)

        F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return F_score, F_geometry

    def _build_train_op(self):
        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.piecewise_constant(self.global_step, self.cfg.lr_boundaries, self.cfg.lr_values)

        tf.summary.scalar("learning_rate", self.lr)

        if self.cfg.train.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        elif self.cfg.train.optimizer == 'rms':
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.lr,
                                                       epsilon=1e-8)
        elif self.cfg.train.optimizer == 'adadelate':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.lr,
                                                        rho=0.9,
                                                        epsilon=1e-06)
        elif self.cfg.train.optimizer == 'sgd':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.lr,
                                                        momentum=0.9)

        # required by batch normalize
        # add update ops(for moving_mean and moving_variance) as a dependency to the train_op
        # https://www.tensorflow.org/api_docs/python/tf/layers/batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

    def _roi_rotate_layer(self, share_conv, text_roi_count, affine_matrixs, affine_pnts,
                          fix_height=8, scope='roi_rotate_layer'):
        """
        根据 ground true 实际位置计算仿射变换的参数，
        然后对 shared feature map 上相应位置区域进行仿射变换，
        获得原图中文字区域经过前向传播后的 feature
        最终的输出是固定高度的，保持长宽比不变
        :param share_conv:
        :param text_roi_count: 每张图片中实际包含了几个文字区域
        :param affine_matrixs
        :param affine_pnts
        :param roi_height: 从 shared feature map 上获得的经过仿射变换后的 roi 高度, roi 的宽度根据长宽比算出
        :return:
            按照一个 batch 里面 roi 最宽的宽度进行 zero padding
            rois: [roi_batch, fix_height, roi_max_width, channels]
                roi_batch: 从 image batch 里面抽出文字区域的个数， roi_batch >= image_batch
                roi_max_width: roi_rotate_layer 中计算出来
                channels: share_conv 最后一层的通道数
            rois_width: [roi_batch]
        """
        with tf.variable_scope(scope):
            rois, rois_width = tf.py_func(roi_rotate_layer,
                                          [share_conv, fix_height, text_roi_count, affine_matrixs,
                                           affine_pnts],
                                          [tf.float32, tf.int32])

            rois.set_shape([None, fix_height, None, share_conv.shape.as_list()[3]])
            rois_width.set_shape([None])
        return rois, rois_width

    def _build_detect_loss(self, y_true_cls, y_true_geo, y_pred_geo):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :return:
        '''
        cls_loss = self._dice_coefficient(y_true_cls, self.F_score)
        cls_loss *= 0.1

        # FOST 论文里的分类 loss，交叉熵
        # cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_score_maps * self.input_training_mask,
        #                                                    logits=self.F_score_logits * self.input_training_mask)
        # cls_loss = tf.reduce_mean(cls_loss)

        # d1 -> top, d2->right, d3->bottom, d4->left
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = tf.minimum(d2_gt, d2_pred) + tf.minimum(d4_gt, d4_pred)
        h_union = tf.minimum(d1_gt, d1_pred) + tf.minimum(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect

        L_AABB = -tf.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - tf.cos(theta_pred - theta_gt)

        L_g = tf.reduce_mean((L_AABB + 20 * L_theta) * self.input_score_maps * self.input_training_mask)

        self.detect_cls_loss = cls_loss
        self.detect_reg_loss = L_g
        self.detect_loss = L_g + cls_loss

        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * self.input_training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * self.input_training_mask))

    def _dice_coefficient(self, y_true_cls, y_pred_cls):
        """
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        """
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * self.input_training_mask)
        union = tf.reduce_sum(y_true_cls * self.input_training_mask) + tf.reduce_sum(
            y_pred_cls * self.input_training_mask) + eps
        loss = 1. - (2 * intersection / union)
        tf.summary.scalar('classification_dice_loss', loss)
        return loss

    def _build_share_conv(self):
        self.conv_net, self.end_points = self._image_to_head(self.input_images, self.is_training)

        batch_norm_params = {
            'decay': 0.997,
            'epsilon': 1e-5,
            'scale': True,
            'is_training': self.is_training
        }

        with slim.arg_scope([slim.conv2d],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.00005),
                            biases_regularizer=slim.l2_regularizer(0.00005),
                            biases_initializer=tf.constant_initializer(0.0)):
            # stride 4, channels 64
            stride_4_conv = self.end_points['resnet_v2_50/block1/unit_3/bottleneck_v2/conv1']

            # stride 8, channels 128
            stride_8_conv = self.end_points['resnet_v2_50/block2/unit_4/bottleneck_v2/conv1']

            # stride 16, channels 256
            stride_16_conv = self.end_points['resnet_v2_50/block3/unit_6/bottleneck_v2/conv1']

            # stride 32, channels 2048
            stride_32_conv = self.end_points['resnet_v2_50/block4']
            print(stride_32_conv.shape.as_list())

            with tf.variable_scope('deconv_1'):
                # stride 16, out channel 256 + 64 = 320
                deconv_block1 = self._deconv(stride_32_conv, stride_16_conv, 128, 128)
                print(deconv_block1)

            with tf.variable_scope('deconv_2'):
                # stride 8, channel 128 + 128 = 256
                deconv_block2 = self._deconv(deconv_block1, stride_8_conv, 64, 64)
                print(deconv_block2)

            with tf.variable_scope('deconv_3'):
                # stride 4, channel 256 + 64 = 320
                deconv_block3 = self._deconv(deconv_block2, stride_4_conv, 32, 32, last_layer=True)
                print(deconv_block3)

        return deconv_block3

    def _deconv(self, input, conv_to_concat, conv_channels, dconv_channels, last_layer=False):
        deconv = slim.conv2d(input, conv_channels, 1, 1, 'SAME', scope='conv_1x1')
        deconv = slim.conv2d(deconv, conv_channels, 3, 1, 'SAME', scope='conv_3x3')

        print(deconv)
        deconv = self._upsample_layer(deconv, dconv_channels, last_layer)
        print(deconv)
        deconv = self._crop_and_concat(deconv, conv_to_concat)

        deconv.set_shape([None, None, None, dconv_channels + conv_to_concat.shape.as_list()[3]])

        deconv = tf.nn.relu(deconv)
        return deconv

    def _upsample_layer(self, input, out_channels, last_layer, name='upsample', ksize=3, stride=2):
        """
        对 Feature map 进行上采样， 放大 input shape 的两倍
        """
        with tf.variable_scope(name):
            in_shape = tf.shape(input)

            # h = ((in_shape[1] - 1) * stride) + 1
            # w = ((in_shape[2] - 1) * stride) + 1
            # output_shape = tf.stack([in_shape[0], h, w, out_channels])
            # if last_layer:
            #     output_shape = tf.stack([in_shape[0], in_shape[1] * 2 - 1, in_shape[2] * 2 - 1, out_channels])
            # else:
            output_shape = tf.stack([in_shape[0], in_shape[1] * 2, in_shape[2] * 2, out_channels])

            static_in_shape = input.shape.as_list()
            filter_shape = [ksize, ksize, out_channels, static_in_shape[3]]
            weights = self._get_deconv_filter(filter_shape)

            upsample = tf.nn.conv2d_transpose(input, weights, output_shape,
                                              strides=[1, stride, stride, 1],
                                              padding='SAME')

            # upsample = tf.Print(upsample, [tf.shape(upsample)], message='Shape of %s' % name, summarize=4, first_n=1)

            # upsample.set_shape((None, static_in_shape[1] * 2, static_in_shape[2] * 2, out_channels))
            return upsample

    def _crop_and_concat(self, x1, x2):
        """
        x2 的长宽要大于 x1
        """
        with tf.name_scope("crop_and_concat"):
            x1_shape = tf.shape(x1)
            x2_shape = tf.shape(x2)
            # offsets for the top left corner of the crop
            offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
            size = [-1, x2_shape[1], x2_shape[2], x1_shape[3]]
            x1_crop = tf.slice(x1, offsets, size)
            return tf.concat([x1_crop, x2], 3)

    def _get_deconv_filter(self, filter_shape):
        width = filter_shape[0]
        height = filter_shape[1]
        f = math.ceil(width / 2.0)
        c = (2 * f - 1 - f % 2) / (2.0 * f)
        bilinear = np.zeros([filter_shape[0], filter_shape[1]])
        for x in range(width):
            for y in range(height):
                value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
                bilinear[x, y] = value
        weights = np.zeros(filter_shape)
        for i in range(filter_shape[2]):
            weights[:, :, i, i] = bilinear

        init = tf.constant_initializer(value=weights,
                                       dtype=tf.float32)
        return tf.get_variable(name="up_filter", initializer=init,
                               shape=weights.shape)

    def _build_reco_output(self, rois, seq_len, num_out, scope='crnn'):
        """
        :param rois: [roi_batch, fix_height, roi_max_width, channels]
        :param seq_len:
        :param num_out: 字符数
        :param scope:
        :return:
        """
        with tf.variable_scope(scope):
            net = slim.conv2d(rois, 64, 3, 1, scope='conv1')
            net = slim.conv2d(net, 64, 3, 1, scope='conv2')
            net = slim.max_pool2d(net, [2, 1], [2, 1], scope='pool1')

            net = slim.conv2d(net, 128, 3, 1, scope='conv3')
            net = slim.conv2d(net, 128, 3, 1, scope='conv4')
            net = slim.max_pool2d(net, [2, 1], [2, 1], scope='pool2')

            net = slim.conv2d(net, 256, 3, 1, scope='conv5')
            net = slim.conv2d(net, 256, 3, 1, scope='conv6')
            net = slim.max_pool2d(net, [2, 1], [2, 1], scope='pool3')
            print(net.shape)
            print(seq_len.shape)

            cnn_out = net
            cnn_output_shape = tf.shape(cnn_out)

            batch_size = cnn_output_shape[0]
            cnn_output_h = cnn_output_shape[1]
            cnn_output_w = cnn_output_shape[2]
            cnn_output_channel = cnn_output_shape[3]

            # Reshape to the shape lstm needed. [batch_size, max_time, ..]
            cnn_out_transposed = tf.transpose(cnn_out, [0, 2, 1, 3])
            cnn_out_reshaped = tf.reshape(cnn_out_transposed,
                                          [batch_size, cnn_output_w, cnn_output_h * cnn_output_channel])

            cnn_shape = cnn_out.get_shape().as_list()
            cnn_out_reshaped.set_shape([None, cnn_shape[2], cnn_shape[1] * cnn_shape[3]])

            with tf.variable_scope('bilstm'):
                bilstm = self._bidirectional_LSTM(cnn_out_reshaped, seq_len, num_out)

        # ctc require time major
        logits = tf.transpose(bilstm, (1, 0, 2))

        # inputs shape: [max_time x batch_size x num_classes]
        self.decoded, self.log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len, merge_repeated=True)

        # dense_decoded shape: [batch_size, encoded_code_size(not fix)]
        # use tf.cast here to support run model on Android
        self.dense_decoded = tf.sparse_tensor_to_dense(tf.cast(self.decoded[0], tf.int32),
                                                       default_value=self.CTC_INVALID_INDEX, name="output")

        # Edit distance for wrong result
        self.edit_distances = tf.edit_distance(tf.cast(self.decoded[0], tf.int32), self.input_text_labels)

        non_zero_indices = tf.where(tf.not_equal(self.edit_distances, 0))
        self.edit_distance = tf.reduce_mean(tf.gather(self.edit_distances, non_zero_indices))

        return logits

    def _bidirectional_LSTM(self, inputs, seq_len, num_out):
        outputs, _ = tf.nn.bidirectional_dynamic_rnn(self._LSTM_cell(),
                                                     self._LSTM_cell(),
                                                     inputs,
                                                     sequence_length=seq_len,
                                                     dtype=tf.float32)

        outputs = tf.concat(outputs, 2)
        outputs = tf.reshape(outputs, [-1, self.cfg.rnn_num_units * 2])

        outputs = slim.fully_connected(outputs, num_out, activation_fn=None)

        shape = tf.shape(inputs)
        outputs = tf.reshape(outputs, [shape[0], -1, num_out])

        return outputs

    def _LSTM_cell(self, num_proj=None):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=self.cfg.rnn_num_units, num_proj=num_proj)
        if self.cfg.rnn_keep_prob < 1:
            cell = tf.contrib.rnn.DropoutWrapper(cell=cell, output_keep_prob=self.cfg.rnn_keep_prob)
        return cell

    def _build_reco_loss(self, reco_logits, input_labels, seq_len):
        # labels:   An `int32` `SparseTensor`.
        #           `labels.indices[i, :] == [b, t]` means `labels.values[i]` stores
        #           the id for (batch b, time t).
        #           `labels.values[i]` must take on values in `[0, num_labels)`.
        # inputs shape: [max_time, batch_size, num_classes]`

        # reco_logits batch_size 的顺序必须和 input_labels 的顺序一样
        # 这一点是通过 dataset.py 中 _input_py_parser 保证的
        ctc_loss = tf.nn.ctc_loss(inputs=reco_logits,
                                  labels=input_labels,
                                  ignore_longer_outputs_than_inputs=True,
                                  sequence_length=seq_len)
        ctc_loss = tf.reduce_mean(ctc_loss)
        return ctc_loss
