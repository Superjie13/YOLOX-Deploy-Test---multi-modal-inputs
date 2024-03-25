from typing import List, Dict
import os
import cv2
import numpy as np


def load_rgb_from_file(file_path, to_rgb=True):
    assert os.path.exists(file_path), f'Image file {file_path} does not exist'
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def load_disp_from_file(file_path):
    assert os.path.exists(file_path), f'Image file {file_path} does not exist'
    disp = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

    # to 3 channels
    disp = np.repeat(disp[:, :, np.newaxis], 3, axis=2)

    # to float32
    disp = disp.astype(np.float32)

    # pre-processing
    disp[disp == 65535] = 0
    disp = disp / 16.

    return disp

def resize_img(img, img_scale, interpolation=cv2.INTER_NEAREST):
    return cv2.resize(img, img_scale, interpolation=interpolation)

def pad_img(img, divisor=32, pad_val=114.0, mode=cv2.BORDER_CONSTANT):
    h, w, c = img.shape
    pad_h = int(np.ceil(h / divisor)) * divisor
    pad_w = int(np.ceil(w / divisor)) * divisor

    _h = max(pad_h - h, 0)
    _w = max(pad_w - w, 0)
    padding = (0, 0, _w, _h)  # left, top, right, bottom

    # if pad_val is a number, pad all channels with the same value
    if isinstance(pad_val, (int, float)):
        pad_val = (pad_val,) * c

    img = cv2.copyMakeBorder(img,
                             padding[1], padding[3], padding[0], padding[2],
                             mode,
                             value=pad_val)

    return img


def draw_bbox(img: np.ndarray, bboxes: List[np.ndarray], labels: List,
              scores: List, id2label: Dict, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on image.
    Args:
        img (np.ndarray): Image to draw on, shape (H, W, C).
        bboxes (List[np.ndarray]): List of bounding boxes, each is a numpy array with shape (4,).
        labels (List): List of labels.
        scores (List): List of scores.
        id2label (Dict): Mapping from class id to label name.
        color (tuple): Color of bounding boxes.
        thickness (int): Thickness of bounding boxes.
    Returns:
        np.ndarray: Image with bounding boxes.
    """
    for bbox, label, score in zip(bboxes, labels, scores):
        x1, y1, x2, y2 = bbox
        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        # text = id2label[label]
        # img = cv2.putText(img, str(text), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        text = f'{score:.2f}'
        img = cv2.putText(img, str(text), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img