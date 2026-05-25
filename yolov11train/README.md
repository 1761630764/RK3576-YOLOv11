# YOLOv11n 自定义模型训练与推理

这是一个完整的 YOLOv11n 目标检测项目，支持自定义数据集训练和多种场景推理检测。

## 项目结构

```
yolov11train/
├── train.py                     # 训练脚本
├── detect.py                    # 推理检测脚本
├── requirements.txt             # 依赖包
├── models/
│   └── yolo11n.pt              # 预训练模型
├── dataset/
│   ├── custom_dataset.yaml     # 数据集配置
│   ├── images/                 # 图片目录
│   │   ├── train/
│   │   └── valid/
│   └── labels/                 # 标签目录
│       ├── train/
│       └── valid/
└── runs/
    ├── train/                  # 训练结果
    └── detect/                 # 检测结果
```

---

## 一、环境部署

### 1. 安装依赖

```bash
# 创建环境
conda create -n yolov11 python=3.11

#激活环境
conda activate yolov11

#安装依赖
pip install ultralytics==8.3.0     或者   pip install -r requirements.txt
```

### 2. 安装 PyTorch（GPU版本，推荐）

```bash
# CUDA 11.8
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118

#这个需要根据设备硬件主要是显卡驱动、显卡型号、python版本和cuda版本来确定
```

### 3. 验证安装

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 二、数据集准备

### 1. 数据集格式

采用 YOLO 格式，标签文件为 TXT，每行格式：

```
class_id center_x center_y width height
```

坐标为归一化值（0-1之间）。

### 2. 目录结构

```
dataset/
├── images/
│   ├── train/          # 训练图片
│   └── valid/          # 验证图片
└── labels/
    ├── train/          # 训练标签（.txt）
    └── valid/          # 验证标签（.txt）
```

### 3. 配置数据集

编辑 `dataset/custom_dataset.yaml`：

```yaml
path: E:/Code/rkyolov11/yolov11train/dataset
train: images/train
val: images/valid

nc: 1  # 类别数量
names:
  0: apple  # 类别名称
```

---

## 三、模型训练

### 快速开始

```bash
python train.py
```

### 自定义训练参数

编辑 `train.py` 文件底部的参数：

```python
train_yolov11n(
    model_path='models/yolo11n.pt',
    data_yaml='dataset/custom_dataset.yaml',
    epochs=200,              # 训练轮数
    batch_size=16,           # 批次大小
    img_size=640,            # 图像尺寸
    device='0',              # GPU设备（'0', '1', 'cpu'）
    pretrained=True,         # 使用预训练权重
    lr0=0.01,               # 初始学习率
    optimizer='SGD',         # 优化器
    patience=50,            # 早停耐心值
    workers=1,              # 数据加载线程数
)
```

### 训练结果

训练完成后，结果保存在 `runs/train/yolov11n_custom/`：

- `weights/best.pt` - 最佳模型
- `weights/last.pt` - 最后一轮模型
- `results.png` - 训练曲线
- `confusion_matrix.png` - 混淆矩阵

---

## 四、推理检测

### 1. 检测图片

```bash
python detect.py --source path/to/image.jpg --weights runs/train/yolov11n_custom/weights/best.pt
```

### 2. 检测视频

```bash
python detect.py --source path/to/video.mp4 --conf 0.5
```

### 3. 使用摄像头

```bash
python detect.py --source 0 --show
```

### 4. 检测图片目录

```bash
python detect.py --source path/to/images/ --save-txt --save-crop
```

### 5. 自定义参数

```bash
python detect.py \
    --source video.mp4 \
    --weights runs/train/yolov11n_custom/weights/best.pt \
    --conf 0.3 \
    --iou 0.5 \
    --device 0 \
    --img-size 640 \
    --save \
    --show
```

### 常用参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--source` | 输入源（图片/视频/摄像头/RTSP流） | 0 |
| `--weights` | 模型权重路径 | best.pt |
| `--conf` | 置信度阈值 | 0.25 |
| `--iou` | NMS IOU阈值 | 0.45 |
| `--device` | 推理设备（0, 1, cpu） | 0 |
| `--img-size` | 推理图像尺寸 | 640 |
| `--save` | 保存检测结果 | True |
| `--show` | 实时显示结果 | False |
| `--save-txt` | 保存为txt文件 | False |
| `--save-crop` | 保存裁剪的检测框 | False |

### 检测结果

结果保存在 `runs/detect/exp/`：

- 检测后的图片/视频
- `labels/` - txt标签文件（如果使用 `--save-txt`）
- `crops/` - 裁剪的目标（如果使用 `--save-crop`）

---

## 五、常见问题

### 1. CUDA out of memory

减小批次大小：

```python
batch_size=8  # 或更小
```

### 2. 训练速度慢

- 使用 GPU 训练
- 增加 workers 数量
- 减小图像尺寸

### 3. 模型不收敛

- 检查数据集标注是否正确
- 降低学习率：`lr0=0.001`
- 使用预训练权重：`pretrained=True`

### 4. 检测效果不好

- 增加训练轮数
- 调整置信度阈值：`--conf 0.3`
- 使用更多训练数据

---

## 六、训练技巧

### 批次大小选择

- GPU 4GB: `batch_size=8`
- GPU 8GB: `batch_size=16`
- GPU 12GB+: `batch_size=32`

### 学习率调整

- 小数据集（<1000张）: `lr0=0.001`
- 中等数据集（1000-10000张）: `lr0=0.01`
- 大数据集（>10000张）: `lr0=0.01-0.1`

### 迁移学习 vs 从头训练

- **迁移学习**（推荐）: `pretrained=True`，适用于大多数场景
- **从头训练**: `pretrained=False`，适用于特殊领域数据

---

## 七、参考资源

- [Ultralytics YOLOv11 官方文档](https://docs.ultralytics.com/)
- [YOLO 数据集格式说明](https://docs.ultralytics.com/datasets/)

---

## 许可证

本项目基于 AGPL-3.0 许可证。
