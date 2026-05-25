"""
ONNX模型推理测试脚本
使用ONNXRuntime进行推理
"""

import cv2
import numpy as np
from pathlib import Path
import time
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class ONNXInference:
    """ONNX模型推理类"""
    
    def __init__(self, model_path: str, input_size: int = 640):
        """
        初始化ONNX推理器
        
        Args:
            model_path: ONNX模型路径
            input_size: 输入尺寸
        """
        self.model_path = model_path
        self.input_size = input_size
        self.session = None
        self.input_name = None
        self.output_names = None
        
        # 加载模型
        self._load_model()
    
    def _load_model(self):
        """加载ONNX模型"""
        try:
            import onnxruntime as ort
            
            print(f"加载ONNX模型: {self.model_path}")
            
            # 检查模型文件
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
            # 创建推理会话
            self.session = ort.InferenceSession(
                self.model_path,
                providers=['CPUExecutionProvider']
            )
            
            # 获取输入输出名称
            self.input_name = self.session.get_inputs()[0].name
            self.output_names = [output.name for output in self.session.get_outputs()]
            
            print(f"✓ ONNX模型加载成功")
            print(f"  输入名称: {self.input_name}")
            print(f"  输出数量: {len(self.output_names)}")
            
        except ImportError:
            raise ImportError("无法导入onnxruntime，请安装: pip install onnxruntime")
        except Exception as e:
            raise RuntimeError(f"加载ONNX模型失败: {e}")
    
    def preprocess(self, image: np.ndarray):
        """
        图像预处理
        
        Args:
            image: 输入图像 (BGR格式)
            
        Returns:
            preprocessed: 预处理后的图像 (1, 3, H, W)
            scale: 缩放比例
            pad: 填充 (pad_w, pad_h)
        """
        h, w = image.shape[:2]
        
        # 计算缩放比例（保持宽高比）
        scale = min(self.input_size / w, self.input_size / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # 创建填充图像（灰色背景）
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        
        # 计算填充位置（居中）
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2
        
        # 放置图像
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # BGR转RGB
        rgb_image = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        
        # 归一化到 [0, 1] 并转换为 (1, 3, H, W)
        normalized = rgb_image.astype(np.float32) / 255.0
        transposed = np.transpose(normalized, (2, 0, 1))  # HWC -> CHW
        batched = np.expand_dims(transposed, axis=0)  # CHW -> NCHW
        
        return batched, scale, (pad_w, pad_h)
    
    def postprocess(self, outputs, scale, pad, orig_shape, conf_threshold=0.25, nms_threshold=0.45):
        """
        后处理：解析YOLO11多尺度输出 + NMS
        
        Args:
            outputs: ONNX推理输出 (9个特征图)
            scale: 缩放比例
            pad: 填充 (pad_w, pad_h)
            orig_shape: 原始图像尺寸 (h, w)
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            检测结果列表 [[x1, y1, x2, y2, conf, class_id], ...]
        """
        # YOLO11输出格式：9个特征图，每3个为一组（3个尺度）
        # 每组包含：[bbox_dist(64), class_scores(80), objectness(1)]
        # 尺度：80x80, 40x40, 20x20
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        
        # 处理3个尺度
        strides = [8, 16, 32]  # 对应80x80, 40x40, 20x20
        
        for scale_idx in range(3):
            # 获取当前尺度的3个输出
            bbox_idx = scale_idx * 3
            cls_idx = scale_idx * 3 + 1
            obj_idx = scale_idx * 3 + 2
            
            bbox_output = outputs[bbox_idx][0]  # (64, H, W)
            cls_output = outputs[cls_idx][0]    # (80, H, W)
            obj_output = outputs[obj_idx][0]    # (1, H, W)
            
            stride = strides[scale_idx]
            h, w = bbox_output.shape[1], bbox_output.shape[2]
            
            # 生成网格坐标
            grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
            grid_y = grid_y.flatten()
            grid_x = grid_x.flatten()
            
            # 展平特征图
            bbox_flat = bbox_output.reshape(64, -1).T  # (H*W, 64)
            cls_flat = cls_output.reshape(80, -1).T    # (H*W, 80)
            obj_flat = obj_output.reshape(1, -1).T     # (H*W, 1)
            
            # DFL解码：将64维分布转换为4个坐标
            # 简化版本：取每16个值的加权平均
            reg_max = 16
            boxes_decoded = np.zeros((bbox_flat.shape[0], 4))
            for i in range(4):
                dist = bbox_flat[:, i*reg_max:(i+1)*reg_max]
                # Softmax
                dist_exp = np.exp(dist - np.max(dist, axis=1, keepdims=True))
                dist_prob = dist_exp / np.sum(dist_exp, axis=1, keepdims=True)
                # 加权求和
                weights = np.arange(reg_max)
                boxes_decoded[:, i] = np.sum(dist_prob * weights, axis=1)
            
            # 转换为bbox坐标 (相对于输入图像)
            # boxes_decoded: [left, top, right, bottom] 距离
            x1 = (grid_x - boxes_decoded[:, 0]) * stride
            y1 = (grid_y - boxes_decoded[:, 1]) * stride
            x2 = (grid_x + boxes_decoded[:, 2]) * stride
            y2 = (grid_y + boxes_decoded[:, 3]) * stride
            
            # 获取类别分数和ID
            class_ids = np.argmax(cls_flat, axis=1)
            class_scores = np.max(cls_flat, axis=1)
            
            # 结合objectness分数
            obj_scores = obj_flat[:, 0]
            # Sigmoid激活
            obj_scores = 1 / (1 + np.exp(-obj_scores))
            class_scores = 1 / (1 + np.exp(-class_scores))
            
            # 最终置信度
            confidences = obj_scores * class_scores
            
            # 过滤低置信度
            mask = confidences >= conf_threshold
            
            if np.sum(mask) > 0:
                all_boxes.append(np.stack([x1[mask], y1[mask], x2[mask], y2[mask]], axis=1))
                all_scores.append(confidences[mask])
                all_class_ids.append(class_ids[mask])
        
        # 合并所有尺度的检测
        if len(all_boxes) == 0:
            return []
        
        boxes = np.vstack(all_boxes)
        confidences = np.concatenate(all_scores)
        class_ids = np.concatenate(all_class_ids)
        
        if len(boxes) == 0:
            return []
        
        # boxes已经是x1,y1,x2,y2格式，直接提取
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # 还原到原始图像坐标
        pad_w, pad_h = pad
        x1 = (x1 - pad_w) / scale
        y1 = (y1 - pad_h) / scale
        x2 = (x2 - pad_w) / scale
        y2 = (y2 - pad_h) / scale
        
        # 裁剪到图像边界
        orig_h, orig_w = orig_shape
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        # 按类别进行NMS
        boxes_for_nms = np.stack([x1, y1, x2, y2], axis=1)
        
        detections = []
        unique_classes = np.unique(class_ids)
        
        for cls in unique_classes:
            cls_mask = class_ids == cls
            cls_boxes = boxes_for_nms[cls_mask]
            cls_scores = confidences[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            
            # 对当前类别进行NMS
            keep_indices = self._nms(cls_boxes, cls_scores, nms_threshold)
            
            # 添加到结果
            for idx in keep_indices:
                orig_idx = cls_indices[idx]
                detections.append([
                    float(x1[orig_idx]),
                    float(y1[orig_idx]),
                    float(x2[orig_idx]),
                    float(y2[orig_idx]),
                    float(confidences[orig_idx]),
                    int(class_ids[orig_idx])
                ])
        
        return detections
    
    def _nms(self, boxes, scores, threshold):
        """非极大值抑制"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
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
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def inference(self, image: np.ndarray, conf_threshold=0.25, nms_threshold=0.45):
        """
        执行推理
        
        Args:
            image: 输入图像 (BGR格式)
            conf_threshold: 置信度阈值
            nms_threshold: NMS阈值
            
        Returns:
            detections: 检测结果
            inference_time: 推理时间（秒）
        """
        if self.session is None:
            raise RuntimeError("ONNX模型未加载")
        
        # 记录原始尺寸
        orig_h, orig_w = image.shape[:2]
        
        # 预处理
        preprocessed, scale, pad = self.preprocess(image)
        
        # 推理
        start_time = time.time()
        outputs = self.session.run(self.output_names, {self.input_name: preprocessed})
        inference_time = time.time() - start_time
        
        # 后处理
        detections = self.postprocess(outputs, scale, pad, (orig_h, orig_w), 
                                     conf_threshold, nms_threshold)
        
        return detections, inference_time


def draw_detections(image, detections, class_names=None):
    """
    在图像上绘制检测结果
    
    Args:
        image: 输入图像
        detections: 检测结果列表
        class_names: 类别名称列表
        
    Returns:
        绘制后的图像
    """
    result_image = image.copy()
    
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # 绘制边界框
        cv2.rectangle(result_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        # 绘制标签
        if class_names and cls < len(class_names):
            label = f"{class_names[cls]}: {conf:.2f}"
        else:
            label = f"Class {cls}: {conf:.2f}"
        
        # 计算文本大小
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # 绘制文本背景
        cv2.rectangle(result_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), (255, 0, 0), -1)
        
        # 绘制文本
        cv2.putText(result_image, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result_image


def main():
    """主函数"""
    print("=" * 60)
    print("ONNX模型推理测试")
    print("=" * 60)
    
    # 配置
    model_path = "conversion/models/yolo11n.onnx"
    test_images_dir = "conversion/int8_images_test"
    results_dir = "results"
    conf_threshold = 0.30  # 提高置信度阈值
    nms_threshold = 0.30
    
    # COCO类别名称（YOLOv8使用COCO数据集）
    class_names = [
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
    
    try:
        # 创建结果目录
        Path(results_dir).mkdir(exist_ok=True)
        print(f"\n结果保存目录: {results_dir}")
        
        # 创建推理器
        print(f"\n初始化ONNX推理器...")
        inferencer = ONNXInference(model_path)
        
        # 获取测试图片
        test_images = sorted(Path(test_images_dir).glob("*.jpg"))
        if not test_images:
            print(f"错误: 在 {test_images_dir} 中未找到测试图片")
            return
        
        print(f"\n找到 {len(test_images)} 张测试图片")
        print(f"置信度阈值: {conf_threshold}")
        print(f"NMS阈值: {nms_threshold}")
        print("\n" + "=" * 60)
        
        # 推理统计
        total_time = 0
        total_detections = 0
        
        # 处理每张图片
        for i, img_path in enumerate(test_images, 1):
            print(f"\n[{i}/{len(test_images)}] 处理: {img_path.name}")
            
            # 读取图像
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ✗ 无法读取图像")
                continue
            
            print(f"  图像尺寸: {image.shape[1]}x{image.shape[0]}")
            
            # 推理
            detections, inference_time = inferencer.inference(
                image, conf_threshold, nms_threshold
            )
            
            total_time += inference_time
            total_detections += len(detections)
            
            print(f"  推理时间: {inference_time*1000:.2f}ms")
            print(f"  检测到 {len(detections)} 个目标")
            
            # 显示检测详情（限制显示前20个）
            if detections:
                display_count = min(20, len(detections))
                for j, det in enumerate(detections[:display_count], 1):
                    x1, y1, x2, y2, conf, cls = det
                    cls_name = class_names[cls] if cls < len(class_names) else f"Class{cls}"
                    print(f"    {j}. {cls_name}: {conf:.3f} [{int(x1)},{int(y1)},{int(x2)},{int(y2)}]")
                if len(detections) > display_count:
                    print(f"    ... 还有 {len(detections) - display_count} 个检测结果")
            
            # 绘制检测结果
            result_image = draw_detections(image, detections, class_names)
            
            # 添加信息文本
            info_text = f"ONNX | {inference_time*1000:.1f}ms | {len(detections)} objects"
            cv2.putText(result_image, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
            
            # 保存结果
            output_path = Path(results_dir) / f"onnx_{img_path.name}"
            cv2.imwrite(str(output_path), result_image)
            print(f"  ✓ 结果已保存: {output_path}")
        
        # 统计信息
        print("\n" + "=" * 60)
        print("推理统计")
        print("=" * 60)
        print(f"总图片数: {len(test_images)}")
        print(f"总检测数: {total_detections}")
        print(f"平均推理时间: {(total_time/len(test_images))*1000:.2f}ms")
        print(f"平均FPS: {len(test_images)/total_time:.2f}")
        print(f"\n✓ 所有结果已保存到: {results_dir}/")
        
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
