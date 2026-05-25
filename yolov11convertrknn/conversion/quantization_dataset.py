"""
准备RKNN量化数据集
从训练图片中选择部分图片用于INT8量化
"""

import cv2
import numpy as np
from pathlib import Path
import random
from tqdm import tqdm


def prepare_quantization_dataset(
    image_dir: str,
    output_file: str,
    num_images: int = 100,
    input_size: int = 640,
    random_seed: int = 42,
    generate_txt: bool = True
):
    """
    准备量化数据集
    
    Args:
        image_dir: 图像目录（训练集图片）
        output_file: 输出文件路径（.npy或.txt）
        num_images: 使用的图像数量（推荐100-500张）
        input_size: 输入尺寸
        random_seed: 随机种子
        generate_txt: 是否生成txt格式（RKNN推荐）
    """
    print("=" * 60)
    print("准备RKNN量化数据集")
    print("=" * 60)
    
    # 设置随机种子
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 查找所有图片
    image_path = Path(image_dir)
    if not image_path.exists():
        print(f"错误: 图像目录不存在: {image_dir}")
        return False
    
    # 支持的图片格式
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    all_images = []
    for ext in image_extensions:
        all_images.extend(list(image_path.glob(f'*{ext}')))
        all_images.extend(list(image_path.glob(f'*{ext.upper()}')))
    
    if len(all_images) == 0:
        print(f"错误: 在 {image_dir} 中没有找到图片")
        return False
    
    print(f"\n找到 {len(all_images)} 张图片")
    
    # 随机选择图片
    if len(all_images) > num_images:
        selected_images = random.sample(all_images, num_images)
        print(f"随机选择 {num_images} 张图片用于量化")
    else:
        selected_images = all_images
        print(f"使用全部 {len(all_images)} 张图片")
    
    # 如果生成txt格式，直接写入图片路径
    if generate_txt:
        txt_file = output_file.replace('.npy', '.txt')
        print(f"\n生成dataset.txt文件...")
        
        with open(txt_file, 'w') as f:
            for img_path in selected_images:
                # 写入绝对路径
                abs_path = img_path.absolute()
                f.write(f"{abs_path}\n")
        
        print(f"  ✓ 保存成功: {txt_file}")
        print(f"  图片数量: {len(selected_images)}")
        
        print("\n" + "=" * 60)
        print("量化数据集准备完成！")
        print("=" * 60)
        print(f"\n下一步:")
        print(f"运行: python conversion/onnx_to_rknn.py --dataset {txt_file}")
        print()
        
        return True
    
    # 准备数据集（npy格式）
    dataset = []
    print(f"\n处理图片...")
    
    for img_path in tqdm(selected_images):
        try:
            # 读取图像
            img = cv2.imread(str(img_path))
            
            if img is None:
                print(f"警告: 无法读取图片 {img_path}")
                continue
            
            # 预处理（与YOLOv8训练时一致）
            # 1. Resize到目标尺寸
            img = cv2.resize(img, (input_size, input_size))
            
            # 2. BGR转RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # 3. 归一化到[0, 1]
            img = img.astype(np.float32) / 255.0
            
            # 4. HWC转CHW (Height, Width, Channel -> Channel, Height, Width)
            img = np.transpose(img, (2, 0, 1))
            
            dataset.append(img)
            
        except Exception as e:
            print(f"警告: 处理图片失败 {img_path}: {e}")
            continue
    
    if len(dataset) == 0:
        print("错误: 没有成功处理任何图片")
        return False
    
    # 转换为numpy数组
    dataset = np.array(dataset, dtype=np.float32)
    
    print(f"\n数据集信息:")
    print(f"  形状: {dataset.shape}")
    print(f"  数据类型: {dataset.dtype}")
    print(f"  数值范围: [{dataset.min():.3f}, {dataset.max():.3f}]")
    print(f"  内存占用: {dataset.nbytes / 1024 / 1024:.2f} MB")
    
    # 保存数据集
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n保存量化数据集...")
    np.save(output_file, dataset)
    
    # 验证保存
    saved_size = Path(output_file).stat().st_size / 1024 / 1024
    print(f"  ✓ 保存成功: {output_file}")
    print(f"  文件大小: {saved_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("量化数据集准备完成！")
    print("=" * 60)
    print(f"\n下一步:")
    print(f"运行: python conversion/onnx_to_rknn.py")
    print()
    
    return True


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='准备RKNN量化数据集')
    parser.add_argument('--image-dir', type=str, required=True,
                       help='图像目录（例如: conversion/int8_images）')
    parser.add_argument('--output', type=str, default='conversion/int8_dataset.txt',
                       help='输出文件路径（.txt或.npy）')
    parser.add_argument('--num-images', type=int, default=200,
                       help='使用的图像数量（推荐100-500）')
    parser.add_argument('--input-size', type=int, default=640,
                       help='输入图像尺寸')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--format', type=str, default='txt', choices=['txt', 'npy'],
                       help='输出格式：txt（推荐）或npy')
    
    args = parser.parse_args()
    
    # 执行准备
    success = prepare_quantization_dataset(
        image_dir=args.image_dir,
        output_file=args.output,
        num_images=args.num_images,
        input_size=args.input_size,
        random_seed=args.seed,
        generate_txt=(args.format == 'txt')
    )
    
    import sys
    sys.exit(0 if success else 1)
