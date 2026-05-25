# -*- coding: utf-8 -*-

import cv2
import numpy as np
from rknnlite.api import RKNNLite

# -------------------------------
# 模型配置
# -------------------------------
RKNN_MODEL = "conversion/models/yolo11n.rknn"
MODEL_SIZE = (640, 640)
OBJ_THRESH = 0.55
NMS_THRESH = 0.45

# COCO 80类别
CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# 初始化 RKNN
rknn_lite = RKNNLite()

def load_rknn_model():
    ret = rknn_lite.load_rknn(RKNN_MODEL)
    if ret != 0:
        print("❌ 加载 RKNN 模型失败！")
        exit(ret)
    ret = rknn_lite.init_runtime()
    if ret != 0:
        print("❌ 初始化 RKNN 运行时失败！")
        exit(ret)
    print("✅ RKNN 模型加载成功！")

# -------------------------------
# 工具函数
# -------------------------------
def letter_box(im, new_shape, pad_color=(0, 0, 0), info_need=True):
    shape = im.shape[:2]  # h, w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    if info_need:
        return im, r, (dw, dh)
    return im

def softmax(x, axis=None):
    x = x - x.max(axis=axis, keepdims=True)
    y = np.exp(x)
    return y / y.sum(axis=axis, keepdims=True)

def dfl(position):
    n, c, h, w = position.shape
    p_num = 4
    mc = c // p_num
    y = position.reshape(n, p_num, mc, h, w)
    y = softmax(y, 2)
    acc_metrix = np.arange(mc, dtype=float).reshape(1, 1, mc, 1, 1)
    y = (y * acc_metrix).sum(2)
    return y

def box_process(position):
    grid_h, grid_w = position.shape[2:4]
    col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([MODEL_SIZE[1] // grid_w, MODEL_SIZE[0] // grid_h]).reshape(1, 2, 1, 1)
    position = dfl(position)
    box_xy = grid + 0.5 - position[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
    xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
    return xyxy

def filter_boxes(boxes, box_confidences, box_class_probs):
    box_confidences = box_confidences.reshape(-1)
    class_max_score = np.max(box_class_probs, axis=-1)
    classes = np.argmax(box_class_probs, axis=-1)
    keep = np.where(class_max_score * box_confidences >= OBJ_THRESH)
    boxes = boxes[keep]
    classes = classes[keep]
    scores = (class_max_score * box_confidences)[keep]
    return boxes, classes, scores

def nms_boxes(boxes, scores):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= NMS_THRESH)[0]
        order = order[inds + 1]
    return np.array(keep)

def post_process(input_data):
    boxes, scores, classes_conf = [], [], []
    default_branch = 3
    pair_per_branch = len(input_data) // default_branch
    for i in range(default_branch):
        boxes.append(box_process(input_data[pair_per_branch * i]))
        classes_conf.append(input_data[pair_per_branch * i + 1])
        scores.append(np.ones_like(input_data[pair_per_branch * i + 1][:, :1, :, :], dtype=np.float32))
    def sp_flatten(_in):
        ch = _in.shape[1]
        _in = _in.transpose(0, 2, 3, 1)
        return _in.reshape(-1, ch)
    boxes = np.concatenate([sp_flatten(v) for v in boxes])
    classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
    scores = np.concatenate([sp_flatten(v) for v in scores])
    boxes, classes, scores = filter_boxes(boxes, scores, classes_conf)
    final_boxes, final_classes, final_scores = [], [], []
    for c in set(classes):
        inds = np.where(classes == c)
        b, s = boxes[inds], scores[inds]
        keep = nms_boxes(b, s)
        if len(keep) != 0:
            final_boxes.append(b[keep])
            final_classes.append(np.full(len(keep), c))
            final_scores.append(s[keep])
    if not final_boxes:
        return None, None, None
    return np.concatenate(final_boxes), np.concatenate(final_classes), np.concatenate(final_scores)

# -------------------------------
# 主流程
# -------------------------------
def run_inference(image_path, save_path="result.jpg"):
    """对单张图片进行推理"""
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"❌ 无法读取图像: {image_path}")
        return 0

    orig_h, orig_w = frame.shape[:2]
    img, ratio, (dw, dh) = letter_box(frame, MODEL_SIZE)
    input_data = np.expand_dims(img, axis=0)

    # 模型推理
    import time
    start_time = time.time()
    outputs = rknn_lite.inference([input_data])
    inference_time = time.time() - start_time
    
    boxes, classes, scores = post_process(outputs)

    detection_count = 0
    if boxes is not None and len(classes) > 0:
        detection_count = len(classes)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # === 坐标映射回原图 ===
            x1 = int((x1 - dw) / ratio)
            y1 = int((y1 - dh) / ratio)
            x2 = int((x2 - dw) / ratio)
            y2 = int((y2 - dh) / ratio)

            # 裁剪到原图边界
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))

            cls_id = int(classes[i])
            score = scores[i]
            # 安全获取类别名称
            cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class{cls_id}"
            label = f"{cls_name} {score:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(y1 - 5, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        print(f"  检测到 {detection_count} 个目标")
    else:
        print("  ⚠️ 未检测到目标")

    # 添加信息文本
    info_text = f"RKNN | {inference_time*1000:.1f}ms | {detection_count} objects"
    cv2.putText(frame, info_text, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imwrite(save_path, frame)
    print(f"  ✓ 结果已保存: {save_path}")
    
    return inference_time


# -------------------------------
if __name__ == '__main__':
    import os
    from pathlib import Path
    
    # 配置
    images_dir = "conversion/int8_images_test"
    results_dir = "results"
    
    print("=" * 60)
    print("RKNN模型循环推理测试")
    print("=" * 60)
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    print(f"\n结果保存目录: {results_dir}")
    
    # 加载模型
    print(f"\n初始化RKNN推理器...")
    load_rknn_model()
    
    # 获取所有测试图片
    test_images = sorted(Path(images_dir).glob("*.jpg"))
    if not test_images:
        print(f"❌ 在 {images_dir} 中未找到测试图片")
        rknn_lite.release()
        exit(1)
    
    print(f"\n找到 {len(test_images)} 张测试图片")
    print(f"置信度阈值: {OBJ_THRESH}")
    print(f"NMS阈值: {NMS_THRESH}")
    print("\n" + "=" * 60)
    
    # 推理统计
    total_time = 0
    total_detections = 0
    
    # 循环处理每张图片
    for i, img_path in enumerate(test_images, 1):
        print(f"\n[{i}/{len(test_images)}] 处理: {img_path.name}")
        print(f"  图像路径: {img_path}")
        
        # 构建保存路径
        save_path = os.path.join(results_dir, f"rknn_{img_path.name}")
        
        # 执行推理
        inference_time = run_inference(str(img_path), save_path)
        total_time += inference_time
    
    # 统计信息
    print("\n" + "=" * 60)
    print("推理统计")
    print("=" * 60)
    print(f"总图片数: {len(test_images)}")
    print(f"平均推理时间: {(total_time/len(test_images))*1000:.2f}ms")
    print(f"平均FPS: {len(test_images)/total_time:.2f}")
    print(f"\n✓ 所有结果已保存到: {results_dir}/")
    
    # 释放资源
    rknn_lite.release()
    print("\n✅ 推理完成！")
