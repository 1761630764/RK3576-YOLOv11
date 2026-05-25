#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11n 训练脚本 - 简化版
提供快速训练接口
"""

from ultralytics import YOLO
import torch


def train_yolov11n(
    model_path='models/yolo11n.pt',
    data_yaml='dataset/custom_dataset.yaml',
    epochs=100,
    batch_size=16,
    img_size=640,
    device='0',
    project='runs/train',
    name='yolov11n_custom',
    pretrained=True,
    **kwargs
):
    """
    训练 YOLOv11n 模型

    参数:
        model_path: 预训练模型路径
        data_yaml: 数据集配置文件路径
        epochs: 训练轮数
        batch_size: 批次大小
        img_size: 输入图像尺寸
        device: 训练设备 ('0', '1', 'cpu' 等)
        project: 项目保存路径
        name: 实验名称
        pretrained: 是否使用预训练权重
        **kwargs: 其他训练参数

    返回:
        训练结果
    """

    print("=" * 70)
    print("YOLOv11n 模型训练")
    print("=" * 70)
    print(f"模型路径: {model_path}")
    print(f"数据集配置: {data_yaml}")
    print(f"训练轮数: {epochs}")
    print(f"批次大小: {batch_size}")
    print(f"图像尺寸: {img_size}")
    print(f"训练设备: {device}")
    print(f"使用预训练: {pretrained}")

    # 检查CUDA是否可用
    if device != 'cpu':
        if torch.cuda.is_available():
            print(f"CUDA可用: {torch.cuda.get_device_name(0)}")
        else:
            print("警告: CUDA不可用，将使用CPU训练")
            device = 'cpu'

    print("=" * 70)

    # 加载模型
    model = YOLO(model_path)

    # 开始训练
    # 设置 amp 参数，使用当前模型进行 AMP 检查而不是下载 yolov8n
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        device=device,
        project=project,
        name=name,
        pretrained=pretrained,
        verbose=True,
        plots=True,
        amp=False,  # 启用自动混合精度训练
        **kwargs
    )

    print("\n" + "=" * 70)
    print("训练完成!")
    print("=" * 70)
    print(f"最佳模型: {project}/{name}/weights/best.pt")
    print(f"最后模型: {project}/{name}/weights/last.pt")
    print("=" * 70)

    return results


def validate_model(model_path, data_yaml, device='0', **kwargs):
    """
    验证模型性能

    参数:
        model_path: 模型路径
        data_yaml: 数据集配置文件
        device: 验证设备
        **kwargs: 其他验证参数

    返回:
        验证结果
    """
    print("\n" + "=" * 70)
    print("模型验证")
    print("=" * 70)

    model = YOLO(model_path)
    results = model.val(data=data_yaml, device=device, **kwargs)

    print("=" * 70)
    print("验证完成!")
    print("=" * 70)

    return results


def export_model(model_path, format='onnx', **kwargs):
    """
    导出模型到其他格式

    参数:
        model_path: 模型路径
        format: 导出格式 ('onnx', 'torchscript', 'coreml', 'tflite' 等)
        **kwargs: 其他导出参数

    返回:
        导出路径
    """
    print("\n" + "=" * 70)
    print(f"导出模型为 {format.upper()} 格式")
    print("=" * 70)

    model = YOLO(model_path)
    export_path = model.export(format=format, **kwargs)

    print("=" * 70)
    print(f"模型已导出: {export_path}")
    print("=" * 70)

    return export_path


def predict_image(model_path, source, save=True, conf=0.25, **kwargs):
    """
    使用模型进行预测

    参数:
        model_path: 模型路径
        source: 图像路径或目录
        save: 是否保存结果
        conf: 置信度阈值
        **kwargs: 其他预测参数

    返回:
        预测结果
    """
    print("\n" + "=" * 70)
    print("模型预测")
    print("=" * 70)

    model = YOLO(model_path)
    results = model.predict(source=source, save=save, conf=conf, **kwargs)

    print("=" * 70)
    print("预测完成!")
    print("=" * 70)

    return results


if __name__ == '__main__':
    # 示例: 快速训练
    train_yolov11n(
        model_path='models/yolo11n.pt',
        data_yaml='dataset/custom_dataset.yaml',
        epochs=200,
        batch_size=16,
        img_size=640,
        device='0',
        pretrained=True,

        # 可选: 自定义训练参数
        lr0=0.01,           # 初始学习率
        optimizer='SGD',    # 优化器
        patience=50,        # 早停耐心值
        save_period=50,     # 每10个epoch保存一次
        workers=1,          # 数据加载线程数
    )
