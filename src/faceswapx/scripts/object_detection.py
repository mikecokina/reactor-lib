import os
from pathlib import Path

import onnxruntime as ort
import numpy as np
import cv2

from faceswapx import settings
from faceswapx.shared import get_bytes_from_url

YOLO_PERSON_STR = 'person'
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.8

_sess_options = ort.SessionOptions()
_sess_options.intra_op_num_threads = os.cpu_count()
_sess_options.enable_mem_pattern = False
_sess_options.enable_cpu_mem_arena = False
_sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

_MODEL = None
YOLO_MODEL = os.environ.get('YOLO_MODEL', 'yolov8x.onnx')
DIR = os.path.dirname(os.path.realpath(__file__))

LABEL_NAMES_URL = os.environ.get('YOLO_LABELS', str(Path(DIR) / 'classes.names'))
LABEL_NAMES = get_bytes_from_url(LABEL_NAMES_URL).decode().splitlines()
MODEL_PATH = Path(settings.MODELS_PATH) / "yolov8x.onnx"


def load_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = ort.InferenceSession(MODEL_PATH, _sess_options, providers=["CUDAExecutionProvider"])
    return _MODEL


def preprocess_image(pil_img, new_shape, scaleup=False):
    img = np.array(pil_img)
    shape = img.shape[:2]

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))  # add border
    if img.shape[:2] != new_shape:  # final resize
        img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)

    # Scale input pixel value to 0 to 1
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = img[np.newaxis, :, :, :].astype(np.float32)

    return img


# non-max suppression
def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])
        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes


def compute_iou(box, boxes):
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[0] + box[2], boxes[:, 0] + boxes[:, 2])
    ymax = np.minimum(box[1] + box[3], boxes[:, 1] + boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = box[2] * box[3]
    boxes_area = boxes[:, 2] * boxes[:, 3]
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area
    return iou


def process_res(np_onnx_res, conf_threshold, iou_threshold, label_names):
    labels = []
    for i in range(len(label_names)):

        scores = np_onnx_res[:, 4 + i]
        keep = scores > conf_threshold
        np_onnx_res_aux = np_onnx_res[keep, :]
        scores = scores[keep]

        boxes = np_onnx_res_aux[:, :4]
        # Make x0, y0 left upper corner instead of box center
        boxes[:, 0:2] -= boxes[:, 2:4] / 2
        boxes = boxes.astype(np.int32)
        keep = nms(boxes, scores, iou_threshold)

        for _ in boxes[keep]:
            labels.append(
                label_names[i]
            )
    return labels


def get_object_detection_res(
        pil_image,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
):
    label_names = LABEL_NAMES
    model_sess = load_model()

    model_inputs = model_sess.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_height, input_width = model_inputs[0].shape[2:]
    model_output = model_sess.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]

    input_tensor = preprocess_image(pil_image, (input_height, input_width))

    outputs = model_sess.run(output_names, {input_names[0]: input_tensor})[0]
    predictions = np.squeeze(outputs).T
    detections_list = process_res(predictions, conf_threshold, iou_threshold, label_names)

    return detections_list
