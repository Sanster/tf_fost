import cv2
import numpy as np


def roi_rotate_layer(share_conv, fix_height, affine_matrixs, affine_rects):
    h, w = share_conv.shape[1:3]
    channels = share_conv.shape[3]
    roi_batch_size = affine_matrixs.shape[1]

    assert affine_matrixs.shape[1] == affine_rects.shape[1]

    max_width_rect = max(affine_rects, key=lambda x: x[2] - x[0])
    max_width = max_width_rect[2] - max_width_rect[0]

    rois = np.zeros((roi_batch_size, fix_height, max_width, channels), dtype=np.float32)

    rois_width = []
    # TODO: this may be slow
    roi_count = 0
    for i in share_conv:
        img = share_conv[i]
        for k, M in enumerate(affine_matrixs[i]):
            rect = affine_rects[i][k]
            affine_img = cv2.warpAffine(img, M, (w, h))
            rois[roi_count][:] = affine_img[rect[1]: rect[3], rect[0]:rect[2]]
            rois_width.append(rect[2] - rect[0])

    return rois, rois_width, max_width
