"""
YOLOv11检测器 - 使用RKNNLite进行板端推理
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
import time


class YOLOv11Detector:
    """
    YOLO11目标检测器
    使用RKNNLite API进行RK3576板端推理
    """
    
    def __init__(self, config: dict):
        """
        初始化检测器
        
        Args:
            config: 检测器配置字典
        """
        self.config = config
        self.model_path = config.get('model_path', 'models/yolov11n_int8.rknn')
        self.input_size = config.get('input_size', 640)
        self.conf_threshold = config.get('conf_threshold', 0.55)
        self.nms_threshold = config.get('nms_threshold', 0.45)
        self.target_class = config.get('target_class', 0)
        
        # RKNNLite对象
        self.rknn_lite = None
        
        # 统计信息
        self.inference_count = 0
        self.total_inference_time = 0.0
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载RKNN模型"""
        try:
            from rknnlite.api import RKNNLite
            
            print(f"加载RKNN模型: {self.model_path}")
            
            # 检查模型文件是否存在
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"RKNN模型文件不存在: {self.model_path}")
            
            # 创建RKNNLite对象
            self.rknn_lite = RKNNLite(verbose=False)
            
            # 加载RKNN模型
            ret = self.rknn_lite.load_rknn(self.model_path)
            if ret != 0:
                raise RuntimeError(f"加载RKNN模型失败，错误码: {ret}")
            
            # 初始化运行时环境
            ret = self.rknn_lite.init_runtime()
            if ret != 0:
                raise RuntimeError(f"初始化RKNN运行时失败，错误码: {ret}")
            
            print(f"✓ RKNN模型加载成功")
            print(f"  输入尺寸: {self.input_size}x{self.input_size}")
            print(f"  置信度阈值: {self.conf_threshold}")
            print(f"  NMS阈值: {self.nms_threshold}")
            
        except ImportError:
            raise ImportError("无法导入rknnlite，请确保已安装 rknn-toolkit-lite2")
        except Exception as e:
            raise RuntimeError(f"加载RKNN模型失败: {e}")
    
    def letter_box(self, im, new_shape=(640, 640), pad_color=(114, 114, 114)):
        """
        Letterbox缩放 - 保持宽高比
        
        Args:
            im: 输入图像
            new_shape: 目标尺寸
            pad_color: 填充颜色
            
        Returns:
            处理后的图像, 缩放比例, (pad_w, pad_h)
        """
        shape = im.shape[:2]  # h, w
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # 计算缩放比例
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        
        # Resize
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        # 添加边框
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, 
                               cv2.BORDER_CONSTANT, value=pad_color)
        
        return im, r, (dw, dh)
    
    def softmax(self, x, axis=None):
        """Softmax激活函数"""
        x = x - x.max(axis=axis, keepdims=True)
        y = np.exp(x)
        return y / y.sum(axis=axis, keepdims=True)
    
    def dfl(self, position):
        """
        DFL (Distribution Focal Loss) 解码
        将分布转换为坐标值
        """
        n, c, h, w = position.shape
        p_num = 4
        mc = c // p_num
        y = position.reshape(n, p_num, mc, h, w)
        y = self.softmax(y, 2)
        acc_metrix = np.arange(mc, dtype=float).reshape(1, 1, mc, 1, 1)
        y = (y * acc_metrix).sum(2)
        return y
    
    def box_process(self, position):
        """
        处理bbox位置信息
        将DFL输出转换为xyxy格式
        """
        grid_h, grid_w = position.shape[2:4]
        col, row = np.meshgrid(np.arange(0, grid_w), np.arange(0, grid_h))
        col = col.reshape(1, 1, grid_h, grid_w)
        row = row.reshape(1, 1, grid_h, grid_w)
        grid = np.concatenate((col, row), axis=1)
        stride = np.array([self.input_size // grid_w, self.input_size // grid_h]).reshape(1, 2, 1, 1)
        
        position = self.dfl(position)
        box_xy = grid + 0.5 - position[:, 0:2, :, :]
        box_xy2 = grid + 0.5 + position[:, 2:4, :, :]
        xyxy = np.concatenate((box_xy * stride, box_xy2 * stride), axis=1)
        return xyxy
    
    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """过滤低置信度的框"""
        box_confidences = box_confidences.reshape(-1)
        class_max_score = np.max(box_class_probs, axis=-1)
        classes = np.argmax(box_class_probs, axis=-1)
        
        # 计算最终分数
        _class_pos = np.where(class_max_score * box_confidences >= self.conf_threshold)
        scores = (class_max_score * box_confidences)[_class_pos]
        
        # 过滤目标类别
        if self.target_class is not None:
            class_mask = classes[_class_pos] == self.target_class
            boxes = boxes[_class_pos][class_mask]
            classes = classes[_class_pos][class_mask]
            scores = scores[class_mask]
        else:
            boxes = boxes[_class_pos]
            classes = classes[_class_pos]
        
        return boxes, classes, scores
    
    def nms_boxes(self, boxes, scores):
        """非极大值抑制"""
        if len(boxes) == 0:
            return np.array([])
        
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
            inds = np.where(ovr <= self.nms_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)
    
    def postprocess(
        self, 
        outputs: List[np.ndarray], 
        scale: float, 
        pad: Tuple[float, float],
        orig_shape: Tuple[int, int]
    ) -> List[List[float]]:
        """
        YOLO11后处理
        
        Args:
            outputs: RKNN推理输出 (9个特征图)
            scale: 缩放比例
            pad: 填充 (pad_w, pad_h)
            orig_shape: 原始图像尺寸 (height, width)
            
        Returns:
            检测结果列表，每个元素为 [x1, y1, x2, y2, conf, class_id]
        """
        # YOLO11输出：9个特征图，3个尺度，每个尺度3个输出
        boxes, scores, classes_conf = [], [], []
        default_branch = 3
        pair_per_branch = len(outputs) // default_branch
        
        # 处理每个尺度
        for i in range(default_branch):
            boxes.append(self.box_process(outputs[pair_per_branch * i]))
            classes_conf.append(outputs[pair_per_branch * i + 1])
            scores.append(np.ones_like(outputs[pair_per_branch * i + 1][:, :1, :, :], 
                                      dtype=np.float32))
        
        def sp_flatten(_in):
            ch = _in.shape[1]
            _in = _in.transpose(0, 2, 3, 1)
            return _in.reshape(-1, ch)
        
        # 展平所有尺度的输出
        boxes = np.concatenate([sp_flatten(v) for v in boxes])
        classes_conf = np.concatenate([sp_flatten(v) for v in classes_conf])
        scores = np.concatenate([sp_flatten(v) for v in scores])
        
        # 过滤和NMS
        boxes, classes, scores = self.filter_boxes(boxes, scores, classes_conf)
        
        # 按类别NMS
        final_boxes, final_classes, final_scores = [], [], []
        for c in set(classes):
            inds = np.where(classes == c)
            b, s = boxes[inds], scores[inds]
            keep = self.nms_boxes(b, s)
            if len(keep) != 0:
                final_boxes.append(b[keep])
                final_classes.append(np.full(len(keep), c))
                final_scores.append(s[keep])
        
        if not final_boxes:
            return []
        
        boxes = np.concatenate(final_boxes)
        classes = np.concatenate(final_classes)
        scores = np.concatenate(final_scores)
        
        # 坐标映射回原图
        pad_w, pad_h = pad
        orig_h, orig_w = orig_shape
        
        detections = []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            # 映射回原图
            x1 = int((x1 - pad_w) / scale)
            y1 = int((y1 - pad_h) / scale)
            x2 = int((x2 - pad_w) / scale)
            y2 = int((y2 - pad_h) / scale)
            
            # 裁剪到原图边界
            x1 = max(0, min(x1, orig_w - 1))
            y1 = max(0, min(y1, orig_h - 1))
            x2 = max(0, min(x2, orig_w - 1))
            y2 = max(0, min(y2, orig_h - 1))
            
            detections.append([
                float(x1), float(y1), float(x2), float(y2),
                float(scores[i]), int(classes[i])
            ])
        
        return detections
    
    def detect(self, image: np.ndarray) -> List[List[float]]:
        """
        执行目标检测
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            检测结果列表，每个元素为 [x1, y1, x2, y2, conf, class_id]
        """
        if self.rknn_lite is None:
            raise RuntimeError("RKNN模型未加载")
        
        # 记录原始图像尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 预处理
        img, ratio, (dw, dh) = self.letter_box(image, (self.input_size, self.input_size))
        input_data = np.expand_dims(img, axis=0)
        
        # 推理
        start_time = time.time()
        outputs = self.rknn_lite.inference([input_data])
        inference_time = time.time() - start_time
        
        # 更新统计
        self.inference_count += 1
        self.total_inference_time += inference_time
        
        # 后处理
        detections = self.postprocess(outputs, ratio, (dw, dh), (orig_h, orig_w))
        
        return detections
    
    def get_avg_inference_time(self) -> float:
        """获取平均推理时间（秒）"""
        if self.inference_count == 0:
            return 0.0
        return self.total_inference_time / self.inference_count
    
    def get_fps(self) -> float:
        """获取平均FPS"""
        avg_time = self.get_avg_inference_time()
        if avg_time == 0:
            return 0.0
        return 1.0 / avg_time
    
    def release(self):
        """释放资源"""
        if self.rknn_lite is not None:
            self.rknn_lite.release()
            self.rknn_lite = None
            print("RKNN资源已释放")
    
    def __del__(self):
        """析构函数"""
        self.release()


# 测试代码
if __name__ == '__main__':
    import sys
    
    # 配置
    config = {
        'model_path': 'conversion/models/yolo11n.rknn',
        'input_size': 640,
        'conf_threshold': 0.55,
        'nms_threshold': 0.45,
        'target_class': 0  # person类
    }
    
    try:
        # 创建检测器
        print("创建YOLO11检测器...")
        detector = YOLOv11Detector(config)
        
        # 测试图像
        test_image_path = 'conversion/int8_images_test/500.jpg'
        if not Path(test_image_path).exists():
            print(f"测试图像不存在: {test_image_path}")
            sys.exit(1)
        
        # 读取图像
        image = cv2.imread(test_image_path)
        print(f"\n读取测试图像: {test_image_path}")
        print(f"图像尺寸: {image.shape}")
        
        # 执行检测
        print("\n执行检测...")
        detections = detector.detect(image)
        
        # 输出结果
        print(f"\n检测结果:")
        print(f"  检测到 {len(detections)} 个目标")
        print(f"  推理时间: {detector.get_avg_inference_time()*1000:.2f}ms")
        print(f"  FPS: {detector.get_fps():.2f}")
        
        # 绘制检测框
        for det in detections:
            x1, y1, x2, y2, conf, cls = det
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            label = f"Class{cls}: {conf:.2f}"
            cv2.putText(image, label, (int(x1), int(y1)-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果
        output_path = 'output/detection_result.jpg'
        Path('output').mkdir(exist_ok=True)
        cv2.imwrite(output_path, image)
        print(f"\n结果已保存: {output_path}")
        
        # 释放资源
        detector.release()
        
        print("\n✓ 检测器测试完成")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
