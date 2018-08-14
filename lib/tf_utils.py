import math
import os
from functools import reduce

import tensorflow as tf


def add_scalar_summary(writer, tag, val, step):
    summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=val)])
    writer.add_summary(summary, step)


def print_endpoints(net, img_path, CPU=True):
    img_file = tf.read_file(img_path)
    img_decoded = tf.image.decode_image(img_file, channels=3)

    if CPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        img = sess.run(img_decoded)
        sess.run(net.conv_net, feed_dict={net.input_images: [img], net.is_training: True})

        print('-' * 50)
        run_meta = tf.RunMetadata()
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)
        print("Net FLOP: %.02fM" % (flops.total_float_ops / 1000000))

    def size(v):
        return reduce(lambda x, y: x * y, v.get_shape().as_list())

    print("-" * 50)

    n = sum(size(v) for v in tf.trainable_variables())
    print("Tensorflow trainable params: %.02fM (%dK)" % (n / 1000000, n / 1000))
    print("Input shape: {}".format(net.input_images))
    print("Output shape: {}".format(net.conv_net))

    # print("Total stride: %d" % math.ceil(net.input_images.shape.as_list()[1] / net.conv_net.shape.as_list()[1]))
