import os
import time
import math

from nets.resnet_v2 import ResNetV2

RNG_SEED = 42
import numpy as np

np.random.seed(RNG_SEED)
import tensorflow as tf

tf.set_random_seed(RNG_SEED)

from lib.dataset import Dataset
from lib.label_converter import LabelConverter

from parse_args import parse_args
from lib.config import load_config


# noinspection PyAttributeOutsideInit
class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.cfg = load_config(args.cfg_name)

        self.converter = LabelConverter(chars_file=args.chars_file)

        self.tr_ds = Dataset(self.cfg, args.train_dir, args.train_gt_dir,
                             self.converter, self.cfg.batch_size)

        self.cfg.lr_boundaries = [self.tr_ds.num_batches * epoch for epoch in self.cfg.lr_decay_epochs]
        self.cfg.lr_values = [self.cfg.lr * (self.cfg.lr_decay_rate ** i) for i in
                              range(len(self.cfg.lr_boundaries) + 1)]

        if args.val_dir is None:
            self.val_ds = None
        else:
            self.val_ds = Dataset(self.cfg, args.val_dir, args.val_gt_dir,
                                  self.converter, self.cfg.batch_size, shuffle=False)

        if args.test_dir is None:
            self.test_ds = None
        else:
            # Test images often have different size, so set batch_size to 1
            self.test_ds = Dataset(self.cfg, args.test_dir, args.test_gt_dir,
                                   self.converter, shuffle=False, batch_size=1)

        self.model = ResNetV2(self.cfg, self.converter.num_classes)
        self.model.create_architecture()
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        self.epoch_start_index = 0
        self.batch_start_index = 0

    def train(self):
        self.sess.run(tf.global_variables_initializer())

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=8)
        self.train_writer = tf.summary.FileWriter(self.args.log_dir, self.sess.graph)

        # if self.args.restore:
        #     self._restore()

        print('Begin training...')
        for epoch in range(self.epoch_start_index, self.cfg.epochs):
            self.sess.run(self.tr_ds.init_op)

            for batch in range(self.batch_start_index, self.tr_ds.num_batches):
                batch_start_time = time.time()
                total_cost, detect_loss, detect_cls_loss, detect_reg_loss, reco_loss, global_step, lr = self._train()

                # if batch != 0 and (batch % self.args.log_step == 0):
                #     batch_cost, global_step, lr = self._train_with_summary()
                # else:
                #     batch_cost, global_step, lr = self._train()

                print("{:.02f}s, epoch: {}, batch: {}/{}, total_loss: {:.03}, "
                      "detect_loss: {:.03}, detect_cls_loss: {:.03}, detect_reg_loss: {:.03}, "
                      "reco_loss: {:.03},  lr: {:.05}"
                      .format(time.time() - batch_start_time, epoch, batch, self.tr_ds.num_batches,
                              total_cost, detect_loss, detect_cls_loss, detect_reg_loss, reco_loss, lr))

                # if global_step != 0 and (global_step % self.args.val_step == 0):
                #     val_acc = self._do_val(self.val_ds, epoch, global_step, "val")
                #     test_acc = self._do_val(self.test_ds, epoch, global_step, "test")
                #     self._save_checkpoint(self.args.ckpt_dir, global_step, val_acc, test_acc)

            self.batch_start_index = 0

    # def _restore(self):
    #     utils.restore_ckpt(self.sess, self.saver, self.args.ckpt_dir)
    #
    #     step_restored = self.sess.run(self.model.global_step)
    #
    #     self.epoch_start_index = math.floor(step_restored / self.tr_ds.num_batches)
    #     self.batch_start_index = step_restored % self.tr_ds.num_batches
    #
    #     print("Restored global step: %d" % step_restored)
    #     print("Restored epoch: %d" % self.epoch_start_index)
    #     print("Restored batch_start_index: %d" % self.batch_start_index)

    def _train(self):
        imgs, score_maps, geo_maps, text_roi_count, affine_matrixs, affine_rects, labels, img_paths = \
            self.tr_ds.get_next_batch(self.sess)

        # print(imgs.shape)
        # print(score_maps.shape)
        # print(geo_maps.shape)
        # print(affine_matrixs.shape)
        # print(affine_rects.shape)
        # print(labels[0].shape)

        fetches = [
            self.model.total_loss,
            self.model.detect_loss,
            self.model.detect_cls_loss,
            self.model.detect_reg_loss,
            self.model.reco_ctc_loss,
            self.model.global_step,
            self.model.lr,
            self.model.train_op
        ]

        feed = {
            self.model.input_images: imgs,
            self.model.input_score_maps: score_maps,
            self.model.input_geo_maps: geo_maps,
            self.model.input_text_roi_count: text_roi_count,
            self.model.input_affine_matrixs: affine_matrixs,
            self.model.input_affine_rects: affine_rects,
            self.model.input_text_labels: labels,
            self.model.is_training: True
        }

        # try:
        total_loss, detect_loss, detect_cls_loss, detect_reg_loss, reco_ctc_loss, global_step, lr, _ = self.sess.run(
            fetches, feed)
        # except:
        #     print(img_paths)
        #     exit(-1)

        return total_loss, detect_loss, detect_cls_loss, detect_reg_loss, reco_ctc_loss, global_step, lr

    # def _train_with_summary(self):
    #     img_batch, label_batch, labels, _ = self.tr_ds.get_next_batch(self.sess)
    #     feed = {self.model.inputs: img_batch,
    #             self.model.labels: label_batch,
    #             self.model.is_training: True}
    #
    #     fetches = [self.model.total_loss,
    #                self.model.ctc_loss,
    #                self.model.regularization_loss,
    #                self.model.global_step,
    #                self.model.lr,
    #                self.model.merged_summay,
    #                self.model.dense_decoded,
    #                self.model.edit_distance,
    #                self.model.train_op]
    #
    #     batch_cost, _, _, global_step, lr, summary, predicts, edit_distance, _ = self.sess.run(fetches, feed)
    #     self.train_writer.add_summary(summary, global_step)
    #
    #     predicts = [self.converter.decode(p, CRNN.CTC_INVALID_INDEX) for p in predicts]
    #     accuracy, _ = infer.calculate_accuracy(predicts, labels)
    #
    #     tf_utils.add_scalar_summary(self.train_writer, "train_accuracy", accuracy, global_step)
    #     tf_utils.add_scalar_summary(self.train_writer, "train_edit_distance", edit_distance, global_step)
    #
    #     return batch_cost, global_step, lr

    # def _do_val(self, dataset, epoch, step, name):
    #     if dataset is None:
    #         return None
    #
    #     accuracy, edit_distance = infer.validation(self.sess, self.model.feeds(), self.model.fetches(),
    #                                                dataset, self.converter, self.args.result_dir, name, step)
    #
    #     tf_utils.add_scalar_summary(self.train_writer, "%s_accuracy" % name, accuracy, step)
    #     tf_utils.add_scalar_summary(self.train_writer, "%s_edit_distance" % name, edit_distance, step)
    #
    #     print("epoch: %d/%d, %s accuracy = %.3f" % (epoch, self.cfg.epochs, name, accuracy))
    #     return accuracy

    def _save_checkpoint(self, ckpt_dir, step, val_acc=None, test_acc=None):
        ckpt_name = "crnn_%d" % step
        if val_acc is not None:
            ckpt_name += '_val_%.03f' % val_acc
        if test_acc is not None:
            ckpt_name += '_test_%.03f' % test_acc

        name = os.path.join(ckpt_dir, ckpt_name)
        print("save checkpoint %s" % name)

        meta_exists, meta_file_name = self._meta_file_exist(ckpt_dir)

        self.saver.save(self.sess, name)

        # remove old meta file to save disk space
        if meta_exists:
            try:
                os.remove(os.path.join(ckpt_dir, meta_file_name))
            except:
                print('Remove meta file failed: %s' % meta_file_name)

    def _meta_file_exist(self, ckpt_dir):
        fnames = os.listdir(ckpt_dir)
        meta_exists = False
        meta_file_name = ''
        for n in fnames:
            if 'meta' in n:
                meta_exists = True
                meta_file_name = n
                break

        return meta_exists, meta_file_name


def main():
    dev = '/gpu:0'
    args = parse_args()
    with tf.device(dev):
        trainer = Trainer(args)
        trainer.train()


if __name__ == '__main__':
    main()
