import cv2
import argparse
import PIL.Image as Image
import numpy as np

from utils import (load_rgb_from_file, load_disp_from_file, resize_img, pad_img, draw_bbox)
from utils_yolo import (multiclass_nms, multi_level_grid_priors, post_processing)


def parse_args():
    parser = argparse.ArgumentParser(description='Test onnx model with opencv')
    parser.add_argument('onnx_path', help='path to .onnx file')
    parser.add_argument('rgb_path', help='path to rgb file')
    parser.add_argument('disp_path', help='path to disparity file')
    parser.add_argument('--img_scale', type=int, default=(720, 1280),
                        nargs=2, help='image scale (h, w)')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--score_thr', type=float, default=0.5, help='score threshold')
    return parser.parse_args()


def main():
    args = parse_args()

    rgb = load_rgb_from_file(args.rgb_path)
    disp = load_disp_from_file(args.disp_path)

    # pre-processing pipeline
    rgb = resize_img(rgb, (args.img_scale[1], args.img_scale[0]))
    disp = resize_img(disp, (args.img_scale[1], args.img_scale[0]))

    rgb = pad_img(rgb, pad_val=114.0)
    disp = pad_img(disp, pad_val=0)

    # prepare input blob
    rgb_blob = cv2.dnn.blobFromImage(image=rgb)
    disp_blob = cv2.dnn.blobFromImage(image=disp)

    # load onnx model
    net = cv2.dnn.readNetFromONNX(args.onnx_path)

    # set opencv dnn input
    net.setInput(rgb_blob, 'input.0')
    net.setInput(disp_blob, 'input.1')

    # forward pass
    output_list = net.forward(net.getUnconnectedOutLayersNames())

    bbox_preds = output_list[0:3]
    cls_scores = output_list[3:6]
    objectnesses = output_list[6:9]

    # == post-processing ==
    # compute the prior grid (don't need to do this every time)
    point_grids, stride_grids = multi_level_grid_priors(
        input_size=rgb.shape[:2], strides=[8, 16, 32])

    boxes_list, scores_list = post_processing(point_grids, stride_grids, cls_scores, bbox_preds,
                                              objectnesses, score_thr=args.score_thr)

    final_dets = multiclass_nms(boxes=boxes_list[0], scores=scores_list[0],
                                score_thr=args.score_thr, iou_thr=0.5)

    valid_dets = final_dets[:, :4]
    valid_scores = final_dets[:, 4]
    valid_labels = final_dets[:, 5]

    # draw result
    if args.show:
        id2label = {0: 'drone'}
        # to int
        valid_dets = valid_dets.astype(np.int32)
        rgb = load_rgb_from_file(args.rgb_path, to_rgb=True)
        new_img = draw_bbox(rgb, valid_dets, valid_labels, valid_scores,
                            id2label, color=(0, 0, 255), thickness=1)
        img = Image.fromarray(new_img)
        img.show()


if __name__ == '__main__':
    main()
