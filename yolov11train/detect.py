#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YOLOv11n 推理检测脚本
支持图片、视频、摄像头检测，支持自定义参数和可视化

# 检测图片
python detect.py --source path/to/image.jpg --weights runs/train/yolov11n_custom/weights/best.pt

# 检测视频
python detect.py --source path/to/video.mp4 --conf 0.5

# 使用摄像头
python detect.py --source 0 --show

# 检测图片目录
python detect.py --source path/to/images/ --save-txt --save-crop

# 自定义参数
python detect.py --source video.mp4 --conf 0.3 --iou 0.5 --device 0 --img-size 640

"""

import os
import argparse
from ultralytics import YOLO


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv11n 推理检测')

    # 模型参数
    parser.add_argument('--weights', type=str, default='runs/train/yolov11n_custom/weights/best.pt',
                        help='模型权重路径 (default: runs/train/yolov11n_custom/weights/best.pt)')

    # 输入源参数
    parser.add_argument('--source', type=str, default=r'E:\Code\rkyolov11\yolov11train\dataset\images\test\3_jpg.rf.ad0800cd21cd1a13450d7f552ab5417a.jpg',
                        help='输入源: 图片路径/视频路径/摄像头ID(0,1,...)/RTSP流 (default: 0)')
    parser.add_argument('--img-size', '--imgsz', type=int, default=640,
                        help='推理图像尺寸 (default: 640)')

    # 检测参数
    parser.add_argument('--conf', '--conf-thres', type=float, default=0.70,
                        help='置信度阈值 (default: 0.65)')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.45,
                        help='NMS IOU阈值 (default: 0.65)')
    parser.add_argument('--max-det', type=int, default=300,
                        help='每张图片最大检测数量 (default: 300)')
    parser.add_argument('--classes', nargs='+', type=int, default=None,
                        help='只检测指定类别，例如: --classes 0 1 2')

    # 设备参数
    parser.add_argument('--device', type=str, default='0',
                        help='推理设备: 0, 1, 2... 或 cpu (default: 0)')
    parser.add_argument('--half', action='store_true',
                        help='使用FP16半精度推理')

    # 输出参数
    parser.add_argument('--project', type=str, default='runs/detect',
                        help='结果保存项目路径 (default: runs/detect)')
    parser.add_argument('--name', type=str, default='exp',
                        help='结果保存名称 (default: exp)')
    parser.add_argument('--exist-ok', action='store_true',
                        help='允许覆盖已存在的项目/名称')
    parser.add_argument('--save', action='store_true', default=True,
                        help='保存检测结果 (default: True)')
    parser.add_argument('--save-txt', action='store_true',
                        help='保存结果为txt文件')
    parser.add_argument('--save-conf', action='store_true',
                        help='在txt文件中保存置信度')
    parser.add_argument('--save-crop', action='store_true',
                        help='保存裁剪的检测框')
    parser.add_argument('--nosave', action='store_true',
                        help='不保存图片/视频')

    # 可视化参数
    parser.add_argument('--show', action='store_true',
                        help='显示检测结果')
    parser.add_argument('--show-labels', action='store_true', default=True,
                        help='显示标签 (default: True)')
    parser.add_argument('--show-conf', action='store_true', default=True,
                        help='显示置信度 (default: True)')
    parser.add_argument('--show-boxes', action='store_true', default=True,
                        help='显示边界框 (default: True)')
    parser.add_argument('--line-width', type=int, default=None,
                        help='边界框线宽 (None表示自动)')

    # 视频/摄像头参数
    parser.add_argument('--vid-stride', type=int, default=1,
                        help='视频帧率步长 (default: 1)')
    parser.add_argument('--stream', action='store_true',
                        help='流式处理视频/摄像头')

    # 其他参数
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细信息 (default: True)')
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='类别无关的NMS')

    return parser.parse_args()


def detect_image(model, source, args):
    """
    检测单张图片或图片目录

    参数:
        model: YOLO模型
        source: 图片路径或目录
        args: 命令行参数
    """
    print("\n" + "=" * 70)
    print("图片检测模式")
    print("=" * 70)
    print(f"输入源: {source}")

    results = model.predict(
        source=source,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        classes=args.classes,
        device=args.device,
        half=args.half,
        save=not args.nosave and args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        show_boxes=args.show_boxes,
        line_width=args.line_width,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=args.verbose,
        agnostic_nms=args.agnostic_nms,
    )

    # 打印检测结果统计
    print("\n检测结果统计:")
    for i, result in enumerate(results):
        print(f"图片 {i+1}: 检测到 {len(result.boxes)} 个目标")
        if len(result.boxes) > 0:
            for box in result.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                print(f"  - 类别: {result.names[cls]}, 置信度: {conf:.2f}")

    return results


def detect_video(model, source, args):
    """
    检测视频文件

    参数:
        model: YOLO模型
        source: 视频路径
        args: 命令行参数
    """
    print("\n" + "=" * 70)
    print("视频检测模式")
    print("=" * 70)
    print(f"输入源: {source}")

    results = model.predict(
        source=source,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        classes=args.classes,
        device=args.device,
        half=args.half,
        save=not args.nosave and args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=args.show,
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        show_boxes=args.show_boxes,
        line_width=args.line_width,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=args.verbose,
        vid_stride=args.vid_stride,
        stream=args.stream,
        agnostic_nms=args.agnostic_nms,
    )

    print("\n视频检测完成!")
    return results


def detect_camera(model, camera_id, args):
    """
    检测摄像头实时画面

    参数:
        model: YOLO模型
        camera_id: 摄像头ID
        args: 命令行参数
    """
    print("\n" + "=" * 70)
    print("摄像头检测模式")
    print("=" * 70)
    print(f"摄像头ID: {camera_id}")
    print("按 'q' 键退出")

    results = model.predict(
        source=camera_id,
        imgsz=args.img_size,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        classes=args.classes,
        device=args.device,
        half=args.half,
        save=not args.nosave and args.save,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
        save_crop=args.save_crop,
        show=True,  # 摄像头模式强制显示
        show_labels=args.show_labels,
        show_conf=args.show_conf,
        show_boxes=args.show_boxes,
        line_width=args.line_width,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        verbose=args.verbose,
        stream=True,  # 摄像头模式使用流式处理
        agnostic_nms=args.agnostic_nms,
    )

    return results


def main():
    """主函数"""
    args = parse_args()

    # 检查模型权重是否存在
    if not os.path.exists(args.weights):
        print(f"错误: 模型权重文件不存在: {args.weights}")
        print("请先训练模型或指定正确的权重路径")
        return

    print("=" * 70)
    print("YOLOv11n 推理检测")
    print("=" * 70)
    print(f"模型权重: {args.weights}")
    print(f"输入源: {args.source}")
    print(f"推理设备: {args.device}")
    print(f"图像尺寸: {args.img_size}")
    print(f"置信度阈值: {args.conf}")
    print(f"IOU阈值: {args.iou}")
    print(f"最大检测数: {args.max_det}")
    if args.classes:
        print(f"检测类别: {args.classes}")
    print(f"结果保存: {args.project}/{args.name}")
    print("=" * 70)

    # 加载模型
    print("\n加载模型...")
    model = YOLO(args.weights)
    print("模型加载成功!")

    # 判断输入源类型
    source = args.source

    # 检查是否为摄像头ID
    if source.isdigit():
        camera_id = int(source)
        results = detect_camera(model, camera_id, args)

    # 检查是否为RTSP流
    elif source.startswith('rtsp://') or source.startswith('http://') or source.startswith('https://'):
        print("\n检测到网络流输入")
        results = detect_video(model, source, args)

    # 检查是否为视频文件
    elif os.path.isfile(source) and source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')):
        results = detect_video(model, source, args)

    # 检查是否为图片文件或目录
    elif os.path.isfile(source) or os.path.isdir(source):
        results = detect_image(model, source, args)

    else:
        print(f"错误: 无法识别的输入源: {source}")
        print("支持的输入源类型:")
        print("  - 摄像头: 0, 1, 2, ...")
        print("  - 图片: path/to/image.jpg")
        print("  - 图片目录: path/to/images/")
        print("  - 视频: path/to/video.mp4")
        print("  - RTSP流: rtsp://...")
        return

    print("\n" + "=" * 70)
    print("检测完成!")
    print("=" * 70)
    if not args.nosave and args.save:
        print(f"结果已保存到: {args.project}/{args.name}")


if __name__ == '__main__':
    main()
