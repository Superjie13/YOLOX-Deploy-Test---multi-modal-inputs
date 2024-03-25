"""
Author: Sijie Hu
Date: 24/03/2024
Description: This script loads an .onnx model exported before the nms.
    {input: ['rgb', 'disp'], output: ['dets', 'labels']}.
Notes:
    1. load .onnx model with onnxruntime on CPU
    2. pass input tensor to onnxruntime with io_binding
        2.1 when using io_binding, input tensor should be contiguous
    3. run nms to get final detection results
"""

import PIL.Image as Image
import numpy as np
import torch
import onnxruntime as ort
import argparse

from utils import (load_rgb_from_file, load_disp_from_file, resize_img, pad_img, draw_bbox)
from utils_yolo import multiclass_nms


def parse_args():
    parser = argparse.ArgumentParser(description='Test onnx model with opencv')
    parser.add_argument('onnx_path', help='path to .onnx file')
    parser.add_argument('rgb_path', help='path to rgb file')
    parser.add_argument('disp_path', help='path to disparity file')
    parser.add_argument('--img_scale', type=int, default=(720, 1280),
                        nargs=2, help='image scale (h, w)')
    parser.add_argument('--ort_custom_op_path', type=str,
                      default='/media/sijeihu/Sijie/NII_Proj/model_deploy/mmdeploy/mmdeploy/lib/libmmdeploy_onnxruntime_ops.so',
                      help='path to custom op library')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--score_thr', type=float, default=0.5, help='score threshold')
    return parser.parse_args()

def main():
    args = parse_args()

    rgb = load_rgb_from_file(args.rgb_path, to_rgb=False)
    disp = load_disp_from_file(args.disp_path)

    # pre-processing pipeline
    rgb = resize_img(rgb, (args.img_scale[1], args.img_scale[0]))
    disp = resize_img(disp, (args.img_scale[1], args.img_scale[0]))

    rgb = pad_img(rgb, pad_val=114.0)
    disp = pad_img(disp, pad_val=0)

    # to tensor
    rgb = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).float().cpu()
    disp = torch.tensor(disp).permute(2, 0, 1).unsqueeze(0).float().cpu()

    # contiguous array  Note: !!! contiguous memory is very important for io_binding
    rgb = rgb.contiguous()
    disp = disp.contiguous()

    # load custom op library if exists
    session_options = ort.SessionOptions()

    # if os.path.exists(args.ort_custom_op_path):
    #     session_options.register_custom_ops_library(args.ort_custom_op_path)
    #     print(f'Successfully loaded onnxruntime custom ops from {args.ort_custom_op_path}')

    # build onnxruntime session
    providers = ['CPUExecutionProvider']
    sess = ort.InferenceSession(
        args.onnx_path, session_options, providers=providers)

    output_names = [_.name for _ in sess.get_outputs()]
    input_names = [_.name for _ in sess.get_inputs()]
    io_binding = sess.io_binding()

    input_dict = {input_names[0]: rgb, input_names[1]: disp}
    for name, input_tensor in input_dict.items():
        # Avoid unnecessary data transfer between host and device
        element_type = input_tensor.new_zeros(
            1, device='cpu').numpy().dtype
        io_binding.bind_input(
            name=name,
            device_type='cpu',
            device_id=-1,
            element_type=element_type,
            shape=input_tensor.shape,
            buffer_ptr=input_tensor.data_ptr())

    for name in output_names:
        io_binding.bind_output(name)

    # run inference
    sess.run_with_iobinding(io_binding)

    output_list = io_binding.copy_outputs_to_cpu()

    # for output_name, output_tensor in zip(output_names, output_list):
    #     print(f'{output_name}: {output_tensor}')

    final_dets = multiclass_nms(boxes=output_list[1][0], scores=output_list[0][0],
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