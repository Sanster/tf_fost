import cv2
import numpy as np


def roi_rotate_layer(share_conv, fix_height, text_roi_count, affine_matrixs, affine_rects):
    """
    :param share_conv:
    :param fix_height:
    :param text_roi_count: [batch_size, 1]
    :param affine_matrixs: [batch_size, max_pad_text_count, 2, 3]
    :param affine_rects: [batch_size, max_pad_text_count, 4]
    :return:
    """
    h, w = share_conv.shape[1:3]
    channels = share_conv.shape[3]
    assert affine_matrixs.shape[1] == affine_rects.shape[1]

    # find batch roi max width and total roi count
    rois_width = []
    roi_max_width = 0
    roi_batch_size = 0
    for i in range(affine_rects.shape[0]):
        text_count = int(text_roi_count[i][0])
        for k in range(text_count):
            rect = affine_rects[i][k]
            roi_batch_size += 1
            roi_w = rect[2] - rect[0]
            rois_width.append(roi_w)
            if roi_w > roi_max_width:
                roi_max_width = roi_w

    # print("Batch max roi width: %d" % roi_max_width)

    rois = np.zeros((roi_batch_size, fix_height, roi_max_width, channels), dtype=np.float32)

    # TODO: this may be slow
    # get affine transformed text roi feature map
    count = 0
    for i in range(share_conv.shape[0]):
        img = share_conv[i]
        text_count = int(text_roi_count[i][0])
        for k in range(text_count):
            M = affine_matrixs[i][k]
            rect = affine_rects[i][k]
            affine_img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)
            rois[count][:, 0:rect[2] - rect[0]] = affine_img[rect[1]: rect[3], rect[0]:rect[2]]
            count += 1

    assert count == roi_batch_size

    return rois, rois_width
