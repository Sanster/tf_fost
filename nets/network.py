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


# noinspection PyAttributeOutsideInit,PyProtectedMember,PyMethodMayBeStatic,PyUnresolvedReferences
class Network(object):
    def __init__(self):
        pass

    @abstractmethod
    def _image_to_head(self, inputs, is_training):
        """
        Layers from image input to last Conv layer
        """
        raise NotImplementedError

    def create_architecture(self):
        self.input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        self.input_score_maps = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_score_maps')
        self.input_geo_maps = tf.placeholder(tf.float32, shape=[None, None, None, 5], name='input_geo_maps')
        self.input_training_masks = tf.placeholder(tf.float32, shape=[None, None, None, 1], name='input_training_masks')

        self.is_training = tf.placeholder(tf.bool, name="is_training")

        self._build_network()

        self._build_losses()

    def _build_network(self):
        # stride 4, channels 320
        self.shared_conv = self._build_share_conv()

        self.F_score, self.F_geometry = self._build_detect_output(self.shared_conv)

        self.detect_loss = self._build_detect_loss(self.input_score_maps, self.F_score,
                                                   self.input_geo_maps, self.F_geometry,
                                                   self.input_training_masks)

        self.total_loss = tf.add_n([self.detect_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    def _build_detect_output(self, shared_conv):
        F_score = slim.conv2d(shared_conv, 1, 1, activation_fn=tf.nn.sigmoid, normalizer_fn=None)

        # 4 channel of axis aligned bbox and 1 channel rotation angle
        # TODO: why x 512?
        geo_map = slim.conv2d(shared_conv, 4, 1, activation_fn=tf.nn.sigmoid,
                              normalizer_fn=None) * 512
        # angle is between [-45, 45]
        angle_map = (slim.conv2d(shared_conv, 1, 1, activation_fn=tf.nn.sigmoid,
                                 normalizer_fn=None) - 0.5) * np.pi / 2

        F_geometry = tf.concat([geo_map, angle_map], axis=-1)

        return F_score, F_geometry

    def _roi_rotate(self, share_conv, y_true_geo, roi_height=8):
        """
        根据 ground true 实际位置计算仿射变换的参数，
        然后对 shared feature map 上相应位置区域进行仿射变换，
        获得原图中文字区域经过前向传播后的 feature
        最终的输出是固定高度的，保持长宽比不变
        :param share_conv:
        :param y_true_geo:
        :param roi_height: 从 shared feature map 上获得的经过仿射变换后的 roi 高度,
        roi 的宽度根据长宽比算出
        :return:
        """

        pass

    def _build_detect_loss(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, training_mask):
        '''
        define the loss used for training, contraning two part,
        the first part we use dice loss instead of weighted logloss,
        the second part is the iou loss defined in the paper
        :param y_true_cls: ground truth of text
        :param y_pred_cls: prediction os text
        :param y_true_geo: ground truth of geometry
        :param y_pred_geo: prediction of geometry
        :param training_mask: mask used in training, to ignore some text annotated by ###
        :return:
        '''
        classification_loss = self._dice_coefficient(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

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
        tf.summary.scalar('geometry_AABB', tf.reduce_mean(L_AABB * y_true_cls * training_mask))
        tf.summary.scalar('geometry_theta', tf.reduce_mean(L_theta * y_true_cls * training_mask))
        L_g = L_AABB + 20 * L_theta

        return tf.reduce_mean(L_g * y_true_cls * training_mask) + classification_loss

    def _dice_coefficient(self, y_true_cls, y_pred_cls, training_mask):
        """
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        """
        eps = 1e-5
        intersection = tf.reduce_sum(y_true_cls * y_pred_cls * training_mask)
        union = tf.reduce_sum(y_true_cls * training_mask) + tf.reduce_sum(y_pred_cls * training_mask) + eps
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

            with tf.variable_scope('deconv_1'):
                # stride 16, channel 64
                deconv_block1 = self._deconv(stride_32_conv, 128, 128)
                # stride 16, channel 256 + 64 = 320
                concat_deconv_block1 = tf.concat((deconv_block1, stride_16_conv), 3)

                print(deconv_block1)
                print(concat_deconv_block1)

            with tf.variable_scope('deconv_2'):
                # stride 8, channel 128
                deconv_block2 = self._deconv(concat_deconv_block1, 64, 64)
                # stride 8, channel 128 + 128 = 256
                concat_deconv_block2 = tf.concat((deconv_block2, stride_8_conv), 3)

                print(deconv_block2)
                print(concat_deconv_block2)

            with tf.variable_scope('deconv_3'):
                # stride 4, channel 128
                deconv_block3 = self._deconv(concat_deconv_block2, 32, 32)
                # stride 4, channel 256 + 64 = 320
                concat_deconv_block3 = tf.concat((deconv_block3, stride_4_conv), 3)
                print(deconv_block3)
                print(concat_deconv_block3)

        return concat_deconv_block3

    def _deconv(self, input, conv_channels, out_channels):
        deconv = slim.conv2d(input, conv_channels, 1, 1, 'SAME', scope='conv_1x1')
        print(deconv)
        deconv = slim.conv2d(deconv, conv_channels, 3, 1, 'SAME', scope='conv_3x3')
        print(deconv)
        deconv = self._upsample_layer(deconv, out_channels)
        return deconv

    def _upsample_layer(self, input, out_channels, name='upsample', ksize=3, stride=2):
        """
        对 Feature map 进行上采样， 放大 input shape 的两倍
        """
        with tf.variable_scope(name):
            in_shape = tf.shape(input)

            h = ((in_shape[1] - 1) * stride) + 1
            w = ((in_shape[2] - 1) * stride) + 1
            output_shape = tf.stack([in_shape[0], h, w, out_channels])

            static_in_shape = input.shape.as_list()
            filter_shape = [ksize, ksize, out_channels, static_in_shape[3]]
            weights = self._get_deconv_filter(filter_shape)

            upsample = tf.nn.conv2d_transpose(input, weights, output_shape,
                                              strides=[1, stride, stride, 1],
                                              padding='SAME')

            # upsample = tf.Print(upsample, [tf.shape(upsample)], message='Shape of %s' % name, summarize=4, first_n=1)

            # upsample.set_shape((None, static_in_shape[1] * 2, static_in_shape[2] * 2, out_channels))
            return upsample

    def _build_losses(self):
        pass

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

# def train_step(self, sess, blobs, train_op):
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes']}
#     rpn_loss_cls, rpn_loss_box, rpn_loss, total_loss, _ = sess.run(
#         [self._losses["rpn_cross_entropy"],
#          self._losses['rpn_loss_box'],
#          self._losses['rpn_loss'],
#          self._losses['total_loss'],
#          train_op],
#         feed_dict=feed_dict)
#
#     return rpn_loss_cls, rpn_loss_box, rpn_loss, total_loss, _
#
# def train_step_with_summary(self, sess, blobs, train_op):
#     feed_dict = {self._image: blobs['data'], self._im_info: blobs['im_info'],
#                  self._gt_boxes: blobs['gt_boxes']}
#     rpn_loss_cls, rpn_loss_box, rpn_loss, loss, summary, _ = sess.run([self._losses["rpn_cross_entropy"],
#                                                                        self._losses['rpn_loss_box'],
#                                                                        self._losses['rpn_loss'],
#                                                                        self._losses['total_loss'],
#                                                                        self._summary_op,
#                                                                        train_op],
#                                                                       feed_dict=feed_dict)
#     return rpn_loss_cls, rpn_loss_box, rpn_loss, loss, summary
