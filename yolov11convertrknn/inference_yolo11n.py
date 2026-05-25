#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11n 推理脚本 - 苹果检测
支持 PyTorch (.pt) 和 ONNX (.onnx) 模型
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from ultralytics import YOLO


def inference_image(model_path, image_path, output_path=None, conf=0.25, show=False):
    """
    图片推理

    参数:
        model_path: 模型路径 (.pt 或 .onnx)
        image_path: 图片路径
        output_path: 输出路径
        conf: 置信度阈值
        show: 是否显示结果
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"推理图片: {image_path}")
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        show=show,
        project="runs/detect" if not output_path else os.path.dirname(output_path),
        name="predict" if not output_path else os.path.basename(output_path).split('.')[0],
    )

    print(f"✓ 推理完成")
    if results and len(results) > 0:
        boxes = results[0].boxes
        if boxes is not None:
            print(f"  检测到 {len(boxes)} 个目标")
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"    - {results[0].names[cls]}: {conf:.2f}")

    return results


def inference_video(model_path, video_path, output_path=None, conf=0.25, show=False):
    """
    视频推理

    参数:
        model_path: 模型路径 (.pt 或 .onnx)
        video_path: 视频路径或摄像头ID
        output_path: 输出路径
        conf: 置信度阈值
        show: 是否显示结果
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"推理视频: {video_path}")
    results = model.predict(
        source=video_path,
        conf=conf,
        save=True if output_path else False,
        show=show,
        stream=True,
    )

    frame_count = 0
    for r in results:
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"处理帧: {frame_count}", end='\r')

    print(f"\n✓ 推理完成，共处理 {frame_count} 帧")
    return results


def inference_camera(model_path, camera_id=0, conf=0.25):
    """
    摄像头实时推理

    参数:
        model_path: 模型路径 (.pt 或 .onnx)
        camera_id: 摄像头ID
        conf: 置信度阈值
    """
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)

    print(f"打开摄像头: {camera_id}")
    print("按 'q' 退出")

    results = model.predict(
        source=camera_id,
        conf=conf,
        show=True,
        stream=True,
    )

    for r in results:
        pass  # 实时显示由 show=True 处理

    print("✓ 推理结束")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="YOLOv11n 苹果检测推理")
    parser.add_argument("--model", type=str, required=True,
                        help="模型路径 (.pt 或 .onnx)")
    parser.add_argument("--source", type=str, required=True,
                        help="输入源（图片/视频/摄像头ID）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--show", action="store_true",
                        help="显示结果")

    args = parser.parse_args()

    print("=" * 60)
    print("YOLOv11n 苹果检测推理")
    print("=" * 60)

    # 检查模型文件
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        exit(1)

    # 判断输入类型
    if args.source.isdigit():
        # 摄像头
        inference_camera(args.model, int(args.source), args.conf)
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # 视频
        inference_video(args.model, args.source, args.output, args.conf, args.show)
    else:
        # 图片
        inference_image(args.model, args.source, args.output, args.conf, args.show)

    print("\n" + "=" * 60)
    print("推理完成")
    print("=" * 60)
