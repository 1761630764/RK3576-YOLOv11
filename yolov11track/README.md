# YOLOv11n + ByteTrack 实时目标检测与计数系统

基于RK3576或者RK35xx系列开发板的实时目标检测、跟踪与划线计数系统，采用YOLOv11n模型和ByteTrack算法，支持Web可视化界面和Modbus通信。

---

## 项目说明

本项目是一个完整的视觉检测与计数解决方案，主要特点：

- **高性能检测**：基于YOLOv11n模型，在RK3576 NPU上实现实时推理
- **稳定跟踪**：采用ByteTrack算法，支持多目标跟踪
- **灵活计数**：支持划线计数，提供track和line两种计数模式
- **Web可视化**：实时视频流展示和统计数据监控
- **工业通信**：支持Modbus TCP通信，可与PLC等设备集成
- **高度可配置**：通过YAML配置文件灵活调整所有参数

### 技术栈

- **深度学习框架**：RKNN (Rockchip Neural Network)
- **目标跟踪**：ByteTrack
- **Web框架**：Flask
- **工业通信**：Modbus TCP (pymodbus)
- **图像处理**：OpenCV
- **开发语言**：Python 3.11.x

---

## 项目目录

```
yolov11track/
├── config/
│   └── config.yaml                 # 主配置文件
├── conversion/
│   └── models/
│       └── yolov11n_int8.rknn     # RKNN模型文件
├── src/
│   ├── communication/
│   │   └── modbus_client.py       # Modbus TCP客户端
│   ├── counter/
│   │   └── object_counter.py      # 目标计数器
│   ├── detector/
│   │   └── yolov11_detector.py    # YOLOv11检测器
│   ├── tracker/
│   │   └── byte_tracker.py        # ByteTrack跟踪器
│   ├── utils/
│   │   └── logger.py              # 日志工具
│   └── main_web.py                # 主程序（Web版）
├── test_modbus*.py                # Modbus测试脚本
├── requirements.txt               # Python依赖
└── README.md                      # 本文件
```

---

## 部署步骤

### 1. 环境要求

- **硬件**：RK3576开发板（或其他支持RKNN的Rockchip平台）
- **操作系统**：Linux /Debian /Ubuntu 20.04
- **Python版本**：3.11.x
- **摄像头**：USB摄像头或RTSP网络摄像头

### 2. 安装依赖

```bash
# 克隆项目
git clone <repository_url>
cd /home/yolov11track/

# 创建环境
python -m venv myenv    

# 激活环境
source myenv/bin/activate

# 安装Python依赖
pip install -r requirements.txt

# 安装RKNN运行时（根据开发板型号）
# 参考：https://github.com/rockchip-linux/rknn-toolkit2
# RK3576   对应   conversion/rknn_toolkit_lite2-2.3.2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

### 3. 准备模型

确保RKNN模型文件存在：
```bash
ls conversion/models/yolov11n_int8.rknn
```

### 4. 配置参数

编辑配置文件 `config/config.yaml`：

```yaml
# 相机配置
camera:
  device_id: 0              # 摄像头ID或视频文件路径
  width: 640
  height: 480
  fps: 30

# 检测器配置
detector:
  model_path: "conversion/models/yolov11n_int8.rknn"
  conf_threshold: 0.15
  target_class: 0           # 目标类别ID

# 计数器配置
counter:
  count_threshold: 5        # 触发阈值
  counting_mode: "line"     # track或line模式
  counting_line:
    position: 0.5           # 线的位置（0-1）
    direction: "horizontal" # horizontal或vertical

# Modbus配置（可选）
modbus:
  enabled: true
  host: "192.168.137.235"    #  PLC的IP，修改PLC与rk3576同局域网（192.168.137.xxx）
  port: 502
  heartbeat_enabled: true
  heartbeat_interval: 10
```

### 5. 运行程序

```bash
# 启动Web服务
python src/main_web.py

# 访问Web界面  连接Wan口
# Wan IP：192.168.137.230  / Lan IP：192.168.2.103
# 浏览器打开：http://192.168.137.230:8080
```



## 功能说明

### 1. 目标检测

- 使用YOLOv11n模型进行实时目标检测
- 支持RKNN硬件加速，在RK3576上达到13-17 FPS
- 可配置置信度阈值和NMS阈值
- 支持指定目标类别检测

### 2. 目标跟踪

- 采用ByteTrack算法进行多目标跟踪
- 为每个目标分配唯一的track_id
- 支持遮挡处理和ID保持
- 可配置跟踪参数（track_thresh、match_thresh等）

### 3. 划线计数

提供两种计数模式：

#### Track模式（追踪计数）
- 每个track_id只计数一次
- 适合人流、车流等场景
- 避免目标变换穿越被重复和遗漏计数
- 显示目标ID

#### Line模式（划线计数）
- 每次穿越都计数
- 适合流水线、传送带等场景
- 不依赖ID稳定性
- 不显示目标ID

**计数方向**：
- **IN方向**：水平线向下穿越 / 垂直线向右穿越
- **OUT方向**：水平线向上穿越 / 垂直线向左穿越

**触发模式**：
- `total`：总计数达到阈值触发
- `in`：仅IN方向计数达到阈值触发
- `out`：仅OUT方向计数达到阈值触发
- `net`：净流量（IN-OUT）达到阈值触发

### 4. Web可视化

- 实时视频流展示
- 实时统计数据（FPS、IN/OUT计数、总计数）
- 响应式设计，支持移动端访问
- 美观的渐变UI界面

### 5. Modbus通信

- 支持Modbus TCP协议
- 达到阈值时自动发送指令
- 支持连接保持（keep-alive）
- 心跳机制防止连接超时
- 异步通信不阻塞主线程
- 可配置重试次数和超时时间

**优化特性**：
- 异步通信：不影响检测FPS
- 连接复用：通信耗时从22ms降至7ms
- 心跳保活：定期读取寄存器保持连接
- 并发保护：线程安全的发送机制

---

## 参数说明

### 相机参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `device_id` | int/str | 0 | 摄像头ID、视频文件路径或RTSP地址 |
| `width` | int | 640 | 图像宽度 |
| `height` | int | 480 | 图像高度 |
| `fps` | int | 30 | 目标帧率 |
| `auto_exposure` | bool | true | 自动曝光 |
| `rotation` | int | 0 | 旋转角度（0/90/180/270） |

### 检测器参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `model_path` | str | - | RKNN模型路径 |
| `input_size` | int | 640 | 输入尺寸 |
| `conf_threshold` | float | 0.15 | 置信度阈值 |
| `nms_threshold` | float | 0.15 | NMS阈值 |
| `target_class` | int | 0 | 目标类别ID |

### 跟踪器参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `track_thresh` | float | 0.45 | 高置信度阈值 |
| `low_thresh` | float | 0.1 | 低置信度阈值 |
| `match_thresh` | float | 0.8 | IoU匹配阈值 |
| `track_buffer` | int | 30 | 跟踪缓冲帧数 |

### 计数器参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `count_threshold` | int | 5 | 触发阈值 |
| `reset_on_trigger` | bool | true | 触发后是否清零 |
| `trigger_mode` | str | "total" | 触发模式（total/in/out/net） |
| `counting_mode` | str | "track" | 计数模式（track/line） |
| `counting_line.position` | float | 0.5 | 线的位置（0-1） |
| `counting_line.direction` | str | "horizontal" | 线的方向 |

### Modbus参数

| 参数 | 类型 | 默认值 | 说明 |
|-----|------|--------|------|
| `enabled` | bool | true | 是否启用Modbus |
| `host` | str | - | Modbus服务器IP |
| `port` | int | 502 | Modbus端口 |
| `keep_alive` | bool | true | 保持连接 |
| `heartbeat_enabled` | bool | true | 启用心跳 |
| `heartbeat_interval` | int | 10 | 心跳间隔（秒） |
| `timeout` | int | 5 | 超时时间（秒） |
| `retry_times` | int | 3 | 重试次数 |

---

## 性能指标

### 检测性能

- **平台**：RK3576开发板
- **模型**：YOLOv11n INT8量化
- **分辨率**：640x480
- **FPS**：15.2 FPS（稳定）
- **延迟**：66ms/帧

### 性能分解

| 模块 | 耗时 |
|-----|------|
| 相机采集 | 4ms |
| 目标检测 | 55ms |
| 目标跟踪 | 5ms |
| 计数处理 | 1ms |
| **总计** | **65ms** | **15.2 FPS** |

### Modbus通信性能

| 指标 | 优化前 | 优化后 | 提升 |
|-----|--------|--------|------|
| 通信耗时 | 22ms | 7ms | -68% |
| 通信帧FPS | 12.5 | 15.2 | +22% |
| 连接建立 | 每次 | 仅1次 | ✅ |

---

## 常见问题

### 1. 相机无法打开

**问题**：`Failed to open camera`

**解决**：
```bash
# 检查相机设备
ls /dev/video*

# 测试相机
v4l2-ctl --list-devices

# 修改config.yaml中的device_id
```

### 2. RKNN模型加载失败

**问题**：`Failed to load RKNN model`

**解决**：
- 确认模型文件存在
- 检查RKNN运行时是否正确安装
- 确认模型与开发板型号匹配

### 3. FPS过低

**问题**：FPS低于10

**解决**：
- 降低输入分辨率
- 调整`conf_threshold`和`nms_threshold`
- 检查CPU/NPU负载
- 关闭不必要的可视化

### 4. Modbus连接失败

**问题**：`Modbus连接失败`

**解决**：
```bash
# 检查网络连通性
ping <modbus_server_ip>

# 检查端口
telnet <modbus_server_ip> 502

```

### 5. 计数不准确

**问题**：计数重复或遗漏

**解决**：
- **Track模式**：调整`track_thresh`和`match_thresh`
- **Line模式**：调整计数线位置
- 检查目标是否被正确检测
- 查看DEBUG日志分析穿越事件

---

## 作者

**Luqiang Zhao**

---

## 开发时间

**2026年21月**

---

## 致谢

- [YOLOv11](https://github.com/ultralytics/ultralytics) - 目标检测模型
- [YOLOv11-RK](https://github.com/yuking926/RKNN-YOLO11) - Ultralytics RKNN 官方修改版 
- [ByteTrack](https://github.com/ifzhang/ByteTrack) - 多目标跟踪算法
- [RKNN Toolkit](https://github.com/rockchip-linux/rknn-toolkit2) - Rockchip NPU工具链
- [pymodbus](https://github.com/pymodbus-dev/pymodbus) - Modbus通信库

---

## 特殊说明
关于YOLOv11自定义训练及权重转换 pt - onnx - rknn(fp16/int8) 需要参考在ubuntu上部署环境。
具体训练代码和转换方法参考rkyolov11项目，链接如下：
- [rkyolov11](xxxxxxx) 
- 基于 [YOLOv11-RK] Ultralytics RKNN 官方修改版  二次开发

## 更新日志

### v1.0.0 (2025-05)

- ✅ 实现YOLOv11n + ByteTrack检测跟踪
- ✅ 支持track和line两种计数模式
- ✅ Web可视化界面
- ✅ Modbus TCP通信
- ✅ 异步通信优化
- ✅ 连接保持和心跳机制
- ✅ 位置跳变检测防止误判
- ✅ 完整的配置系统
- ✅ 详细的文档和测试工具

---

**Happy Coding! 🚀**
