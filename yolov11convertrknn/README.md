# YOLO11 在 RK35xx(默认3576) 的模型转工具部署指南

`本项目库仅用于模型转换：pt - onnx - rknn ， 因对官方YOLOv11 - ultralytics 8.3.0进行了修改，移除了NMS和DFL模块，此模块对量化rknn转换极其不友好，缺少NMS模块会导致无法训练和推理，训练使用官方版本：https://github.com/ultralytics/ultralytics`

为了方便在 RKNN 开发板上部署 YOLO11 模型，整理了完整流程，避免配置错误和遗忘关键步骤。

---

## 1. 配置 YOLO11 环境

在PC端 Ubuntu系统中，首先配置 YOLO11 环境（Python 3.11 + PyTorch + Ultralytics）。

⚠️ 注意：RKNN 官方修改过 Ultralytics，导出的 ONNX 模型与官方原版不同。  
可使用 **Netron** 查看 ONNX 文件差异。

推荐流程：

1. 正常安装 YOLO11 环境，使用 Anaconda创建`yolov11convertrknn`虚拟环境。
```
conda create -n yolov11convertrknn python=3.11

conda activate yolov11convertrknn

pip install ultralytics==8.3.0

```

2. 卸载环境中的官方 `ultralytics`：
```
pip uninstall ultralytics

pip install requirements.txt

pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cpu

pip install conversion/rknn_toolkit2-2.3.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

```

3. 下载本仓库的 `rkyolov11` 到本地，在该工作空间中进行 PT - ONNX - RKNN 转换。

---

## 2. 导出 ONNX 模型

确保环境满足 `requirements.txt` 后执行：
修改`./ultralytics/cfg/default.yaml`中 `model` 路径：

```
修改`./ultralytics/cfg/default.yaml`中 model 路径：
# - yolo11n.pt        → 检测模型
# - yolo11n-seg.pt    → 分割模型
# - yolo11n-pose.pt   → 姿态模型
# - yolo11n-obb.pt    → 旋转框检测模型

```


导出命令：

```
export PYTHONPATH=./
python ./ultralytics/engine/exporter.py
```

执行完成后生成 ONNX 模型，例如：
```
yolo11n.pt → yolo11n.onnx
```

> 使用本仓库的 Ultralytics 确保 ONNX 与 RKNN 官方兼容，避免转换失败。

---

## 3. 准备量化数据集合
化数据集用于INT8量化，需要100-400张代表性图片。

```bash

cd /home/zhao/yolov11convertrknn

python conversion/quantization_dataset.py \
    --image-dir conversion/int8_images \
    --output conversion/int8_dataset.txt \
    --num-images 376 \
    --format txt
```

**参数说明：**

- `--image-dir`: 图片目录路径
- `--output`: 输出文件路径（.txt格式）
- `--num-images`: 使用的图片数量（推荐100-200张）
- `--format`: 输出格式，使用 `txt`（推荐）

**预期输出：**
```
============================================================
准备RKNN量化数据集
============================================================

找到 100 张图片
使用全部 100 张图片

生成dataset.txt文件...
  ✓ 保存成功: conversion/int8_dataset.txt
  图片数量: 100

============================================================
量化数据集准备完成！
============================================================
```

**生成的文件：**
- `conversion/int8_dataset.txt` - 包含图片绝对路径的文本文件


---

## 4. ONNX → RKNN 模型转换

使用准备好的量化数据集将ONNX模型转换为RKNN INT8模型。

```bash
cd /home/zhao/yolov11convertrknn

# 修改为自己的onnx路径，/home/zhao/rkyolov11/models/yolo11n.onnx
python conversion/convert_onnx_rknn.py /home/zhao/rkyolov11/models/yolo11n.onnx rk3576 i8
```

**参数说明：**
========================================
使用方式：
========================================

# INT8 量化（默认）
python conversion/convert_onnx_rknn.py model.onnx rk3576

# 或明确指定 INT8
python conversion/convert_onnx_rknn.py model.onnx rk3576 i8

# FP16 不量化
python conversion/convert_onnx_rknn.py model.onnx rk3576 fp



## 📚 参考资料

- [RKNN Toolkit2 文档](https://github.com/rockchip-linux/rknn-toolkit2)
- [YOLOv8 文档](https://docs.ultralytics.com/)
- [ONNX 文档](https://onnx.ai/)
- [RK3576 NPU 性能指南](https://www.rock-chips.com/a/cn/product/RK35xilie/2023/0428/1661.html)

---

**更新日期**: 2026-05-19  
**适用平台**: RK3576  
**Python版本**: 3.11  
**RKNN Toolkit版本**: 2.3.2