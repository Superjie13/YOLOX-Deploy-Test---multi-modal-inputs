from typing import Tuple, Union, List
import numpy as np


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr=0.5):
    """
    Non-maximum suppression.
    Args:
        boxes: (np.ndarray): shape(N, 4)
        scores: (np.ndarray): shape(N,)
        iou_thr: (float): IoU threshold
    Returns:
        keep: (list): list of index to keep
    """
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)

        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= iou_thr)[0]
        order = order[inds + 1]

    return keep

def multiclass_nms_class_aware(boxes: np.ndarray, scores: np.ndarray,
                               iou_thr=0.5, score_thr=0.2):
    """
    Perform class-aware non-maximum suppression on bounding boxes.

    This function performs non-maximum suppression (NMS) for each class separately,
    and then concatenates the results. It is class-aware in the sense that the NMS
    operation is performed independently for each class.

    Args:
        boxes: (np.ndarray): shape(N, 4)
        scores: (np.ndarray): shape(N, num_classes)
        iou_thr: (float): IoU threshold
        score_thr: (float): score thr
    Returns:
        final_dets: (np.ndarray): shape(n, 6), [x1, y1, x2, y2, score, cls]
   """
    final_dets = []
    num_classes = scores.shape[1]
    for cls in range(num_classes):
        cls_scores = scores[:, cls]
        valid_socre_mask = cls_scores > score_thr
        if valid_socre_mask.sum() == 0:
            continue
        valid_scores = cls_scores[valid_socre_mask]
        valid_boxes = boxes[valid_socre_mask]
        keep = nms(valid_boxes, valid_scores, iou_thr)
        if len(keep) > 0:
            cls_inds = np.ones((len(keep), 1)) * cls
            dets = np.concatenate(
                [valid_boxes[keep], valid_scores[keep][:, None], cls_inds],
                axis=1  # [x1, y1, x2, y2, score, cls]
            )
            final_dets.append(dets)

    if len(final_dets) == 0:
        return None

    final_dets = np.concatenate(final_dets, axis=0)

    return final_dets

def multiclass_nms_class_agnostic(boxes: np.ndarray, scores: np.ndarray,
                                  iou_thr=0.5, score_thr=0.2):
    """
    Perform class-agnostic non-maximum suppression on bounding boxes.

    This function performs non-maximum suppression (NMS) without considering the classes of the bounding boxes.
    It first finds the class with the highest score for each box, and then performs NMS on these boxes.

    Args:
        boxes: (np.ndarray): shape(N, 4)
        scores: (np.ndarray): shape(N, num_classes)
        iou_thr: (float): IoU threshold
        score_thr: (float): score threshold
    Returns:
        final_dets: (np.ndarray): shape(n, 6), [x1, y1, x2, y2, score, cls]
    """
    final_dets = None
    cls_inds = scores.argmax(1)
    cls_scores = scores[np.arange(len(cls_inds)), cls_inds]  # or scores.max(1)

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, iou_thr)
    if len(keep) > 0:
        final_dets = np.concatenate(
            [valid_boxes[keep], valid_scores[keep][:, None], valid_cls_inds[keep][:, None]],
            axis=1
        )
    return final_dets

def multiclass_nms(boxes: np.ndarray, scores: np.ndarray,
                   iou_thr=0.5, score_thr=0.2, class_agnostic=True):
    """
    Perform non-maximum suppression on bounding boxes, either class-agnostic or class-aware.

    This function performs NMS on the bounding boxes. Depending on the 'class_agnostic' flag,
    it either performs class-agnostic NMS (not considering the classes of the bounding boxes)
    or class-aware NMS (performing NMS for each class separately).

    Args:
        boxes: (np.ndarray): shape(N, 4)
        scores: (np.ndarray): shape(N, num_classes)
        iou_thr: (float): IoU threshold
        score_thr: (float): score threshold
        class_agnostic: (bool): whether to perform class-agnostic NMS
    Returns:
        final_dets: (np.ndarray or None): shape(n, 6), [x1, y1, x2, y2, score, cls]
    """
    if class_agnostic:
        nms_func = multiclass_nms_class_agnostic
    else:
        nms_func = multiclass_nms_class_aware

    return nms_func(boxes, scores, iou_thr, score_thr)


def single_level_grid_priors(featmap_size: Tuple[int], stride: Union[Tuple[int], int]):
    """
    Generate grid priors for a single feature map level.
    Args:
        featmap_size: (Tuple[int]): feature map size (h, w)
        stride: (Union[Tuple[int], int]): stride for each dimension

    Returns:
        point_grid: (np.ndarray): prior points for each grid cell, shape(N, 2)
        stride_grid: (np.ndarray): stride for each grid cell, shape(N, 2)
    """
    feat_h, feat_w = featmap_size
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride

    g_x = np.arange(0, feat_w) * stride_w
    g_y = np.arange(0, feat_h) * stride_h

    g_x, g_y = np.meshgrid(g_x, g_y)
    g_x = g_x.reshape(-1)
    g_y = g_y.reshape(-1)
    point_grid = np.stack((g_x, g_y), axis=-1)

    stride_grid_w = np.full_like(g_x, stride_w)
    stride_grid_h = np.full_like(g_y, stride_h)
    stride_grid = np.stack((stride_grid_w, stride_grid_h), axis=-1)

    return point_grid, stride_grid

def multi_level_grid_priors(input_size: Tuple[int], strides: List[Union[Tuple[int], int]]):
    """
    Generates grid priors for multiple levels. For anchor-free detectors, e.g., YOLOX.
    Args:
        input_size: (Tuple[int]): input image size after preprocessing, (h, w)
        strides: (List[Union[Tuple[int], int]]): strides for each level

    Returns:
        point_grids: (List[np.ndarray]): list of prior points for each feature map
        stride_grids: (List[np.ndarray]): list of strides for each feature map
    """
    featmap_sizes = []
    for s in strides:
        if isinstance(s, int):
            featmap_sizes.append((input_size[0] // s, input_size[1] // s))
        if isinstance(s, tuple):
            featmap_sizes.append((input_size[0] // s[0], input_size[1] // s[1]))

    point_grids = []
    stride_grids = []
    for fs, s in zip(featmap_sizes, strides):
        point_grid, stride_grid = single_level_grid_priors(fs, s)
        point_grids.append(point_grid)
        stride_grids.append(stride_grid)
    return point_grids, stride_grids


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def box_decoder_yolox(priors_grid: np.ndarray, stride_grid: np.ndarray,
                      pred_bboxes: np.ndarray, ) -> np.ndarray:
    """
    Decode regression results (delta_x, delta_x, w, h) to
    bboxes (tl_x, tl_y, br_x, br_y).
    Args:
        priors_grid (np.ndarray): Basic points of boxes, shape(N, 2).
        stride_grid (np.ndarray): Strides of boxes, shape(N, 2).
        pred_bboxes (np.ndarray): Encoded boxes with shape (N, 4).
    Returns:
        np.ndarray: Decoded boxes.
    """
    xys = (pred_bboxes[..., :2] * stride_grid) + priors_grid
    whs = np.exp(pred_bboxes[..., 2:]) * stride_grid

    tl_x = (xys[..., 0] - whs[..., 0] / 2)
    tl_y = (xys[..., 1] - whs[..., 1] / 2)
    br_x = (xys[..., 0] + whs[..., 0] / 2)
    br_y = (xys[..., 1] + whs[..., 1] / 2)

    return np.stack([tl_x, tl_y, br_x, br_y], -1)


def filter_score_and_topk(scores: np.ndarray, score_thr: float, topk: int):
    """
    Filter out low-confidence detections and select topk detections.
    Args:
        scores: (np.ndarray): shape(N, num_classes)
        score_thr: (float): score threshold
        topk: (int): number of topk
    Returns:
        scores: (np.ndarray)
        labels: (np.ndarray)
        keep_idxs: (np.ndarray)
    """

    valid_mask = scores > score_thr
    scores = scores[valid_mask]
    valid_idxs, labels = np.nonzero(valid_mask)

    num_topk = min(topk, len(valid_idxs))
    # sort and get topk
    topk_inds = scores.argsort()[-num_topk:][::-1]
    scores = scores[topk_inds]
    labels = labels[topk_inds]
    keep_idxs = valid_idxs[topk_inds]

    return scores, labels, keep_idxs


def post_processing(point_grids: List[np.ndarray], stride_grids: List[np.ndarray],
                    cls_scores: List[np.ndarray], bbox_preds: List[np.ndarray],
                    objectnesses: List[np.ndarray], score_thr: float):
    """
    Processes detection raw outputs of YOLOX model for each image grid and obtain final bounding boxes and class scores.

    This function performs several steps:
    1. Flattens and concatenates various prediction components across all image grids.
    2. Decodes bounding boxes from predictions.
    3. Applies objectness thresholding and filters out low-confidence detections.
    4. Merges class scores with objectness scores to finalize the detection confidence.
    5. Selects top-scoring detections.

    Args:
        point_grids (List[np.ndarray]): List of prior points for each feature map.
        stride_grids (List[np.ndarray]): List of strides for each feature map.
        cls_scores (List[np.ndarray]): List of class scores for each feature map.
        bbox_preds (List[np.ndarray]): List of bounding box predictions for each feature map.
        objectnesses (List[np.ndarray]): List of objectness scores for each feature map.
        score_thr (float): Score threshold for filtering.

    Returns:
        boxes_list (List[np.ndarray]): List of final bounding boxes.
        scores_list (List[np.ndarray]): List of final scores.
    """
    flatten_priors_points = np.concatenate(point_grids, axis=0)
    flatten_priors_strides = np.concatenate(stride_grids, axis=0)

    flatten_cls_scores = [cls_scores.transpose(0, 2, 3, 1).reshape(
        1, -1, cls_scores.shape[1]) for cls_scores in cls_scores]
    flatten_bbox_preds = [bbox_preds.transpose(0, 2, 3, 1).reshape(
        1, -1, bbox_preds.shape[1]) for bbox_preds in bbox_preds]
    flatten_objectnesses = [objectnesses.transpose(0, 2, 3, 1).reshape(
        1, -1, objectnesses.shape[1]) for objectnesses in objectnesses]

    flatten_cls_scores = sigmoid(np.concatenate(flatten_cls_scores, axis=1))
    flatten_objectnesses = sigmoid(np.concatenate(flatten_objectnesses, axis=1)).squeeze(-1)
    flatten_bbox_preds = np.concatenate(flatten_bbox_preds, axis=1)
    flatten_decoded_bboxes = box_decoder_yolox(
        flatten_priors_points, flatten_priors_strides, flatten_bbox_preds)

    boxes_list = []
    scores_list = []
    for (bboxes, scores, objectness) in zip(flatten_decoded_bboxes, flatten_cls_scores,
                                            flatten_objectnesses):
        conf_inds = objectness > score_thr
        bboxes = bboxes[conf_inds]
        scores = scores[conf_inds]
        objectness = objectness[conf_inds]

        if objectness is not None:
            scores = scores * objectness[:, None]

        scores, labels, keep_idxs = filter_score_and_topk(scores, score_thr, 1000)

        if np.ndim(scores) == 1:
            # add a new axis to indicate the class, i.e., (N, Cls)
            scores = scores[:, None]

        boxes_list.append(bboxes[keep_idxs])
        scores_list.append(scores)

    return boxes_list, scores_list

