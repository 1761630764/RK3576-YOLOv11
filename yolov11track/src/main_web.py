"""
YOLOv11n + ByteTrack 实时检测系统 - 基于划线计数
"""

import os
import sys
import yaml
import signal
import argparse
import threading
import time
import queue
from pathlib import Path
from flask import Flask, Response, render_template_string, jsonify
import cv2
import numpy as np

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.detector.yolov11_detector import YOLOv11Detector
from src.tracker.byte_tracker import ByteTracker
from src.counter.object_counter import ObjectCounter
from src.communication.modbus_client import ModbusClient
from src.utils.logger import Logger


class SimpleWebApp:
    """简化的Web流媒体应用"""

    def __init__(self, config_path: str, host: str = '0.0.0.0', port: int = 8080):
        self.config_path = config_path
        self.host = host
        self.port = port
        self.config = None

        # Flask应用
        self.app = Flask(__name__)

        # 运行状态
        self.running = False

        # 相机
        self.cap = None

        # 检测器、跟踪器、计数器
        self.detector = None
        self.tracker = None
        self.counter = None
        self.modbus_client = None

        # 帧队列（用于Web流）
        self.frame_queue = queue.Queue(maxsize=2)

        # 统计信息
        self.stats = {
            'fps': 0,
            'in_count': 0,
            'out_count': 0,
            'current_count': 0,
            'total_count': 0,
            'current_detections': 0,
            'uptime': 0
        }
        self.start_time = None

        # 设置路由
        self._setup_routes()

        # 注册信号处理
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """信号处理函数"""
        Logger.info(f"\n收到信号: {signal.Signals(signum).name}")
        self.stop()
        # 不要立即退出，让finally块执行
        raise KeyboardInterrupt()

    def _setup_routes(self):
        """设置Flask路由"""

        @self.app.route('/')
        def index():
            """主页"""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>YOLOv11n + ByteTrack 实时检测系统</title>
                <style>
                    * { margin: 0; padding: 0; box-sizing: border-box; }
                    body {
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        min-height: 100vh;
                        padding: 20px;
                    }
                    .container { max-width: 1400px; margin: 0 auto; }
                    .header {
                        text-align: center;
                        color: white;
                        margin-bottom: 30px;
                    }
                    .header h1 {
                        font-size: 2.5em;
                        margin-bottom: 10px;
                        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
                    }
                    .main-content {
                        display: grid;
                        grid-template-columns: 1fr 350px;
                        gap: 20px;
                    }
                    .video-container {
                        background: white;
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    }
                    .video-wrapper {
                        position: relative;
                        width: 100%;
                        padding-bottom: 75%;
                        background: #000;
                        border-radius: 10px;
                        overflow: hidden;
                    }
                    .video-wrapper img {
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        object-fit: contain;
                    }
                    .stats-panel {
                        background: white;
                        border-radius: 15px;
                        padding: 20px;
                        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                    }
                    .stats-panel h2 {
                        color: #667eea;
                        margin-bottom: 20px;
                        font-size: 1.5em;
                    }
                    .stat-item {
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        color: white;
                        padding: 15px;
                        border-radius: 10px;
                        margin-bottom: 15px;
                    }
                    .stat-label {
                        font-size: 0.9em;
                        opacity: 0.9;
                        margin-bottom: 5px;
                    }
                    .stat-value {
                        font-size: 2em;
                        font-weight: bold;
                    }
                    .status-indicator {
                        display: inline-block;
                        width: 12px;
                        height: 12px;
                        border-radius: 50%;
                        background: #4ade80;
                        animation: pulse 2s infinite;
                        margin-right: 8px;
                    }
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                    @media (max-width: 1024px) {
                        .main-content { grid-template-columns: 1fr; }
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>🎯 YOLOv11n 划线计数系统</h1>
                        <p>RK3576 开发板 | 实时目标检测与划线计数</p>
                    </div>

                    <div class="main-content">
                        <div class="video-container">
                            <h2 style="color: #667eea; margin-bottom: 15px;">
                                <span class="status-indicator"></span>实时视频流
                            </h2>
                            <div class="video-wrapper">
                                <img src="/video_feed" alt="视频流">
                            </div>
                        </div>

                        <div class="stats-panel">
                            <h2>📊 实时统计</h2>

                            <div class="stat-item">
                                <div class="stat-label">帧率 (FPS)</div>
                                <div class="stat-value" id="fps">0</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">IN方向计数</div>
                                <div class="stat-value" id="in-count">0</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">OUT方向计数</div>
                                <div class="stat-value" id="out-count">0</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">当前总计数</div>
                                <div class="stat-value" id="current-count">0</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">累计总计数</div>
                                <div class="stat-value" id="total-count">0</div>
                            </div>

                            <div class="stat-item">
                                <div class="stat-label">当前检测目标</div>
                                <div class="stat-value" id="current-tracked">0</div>
                            </div>
                        </div>
                    </div>
                </div>

                <script>
                    function updateStats() {
                        fetch('/stats')
                            .then(response => response.json())
                            .then(data => {
                                document.getElementById('fps').textContent = data.fps.toFixed(1);
                                document.getElementById('in-count').textContent = data.in_count;
                                document.getElementById('out-count').textContent = data.out_count;
                                document.getElementById('current-count').textContent = data.current_count;
                                document.getElementById('total-count').textContent = data.total_count;
                                document.getElementById('current-tracked').textContent = data.current_tracked;
                            })
                            .catch(error => console.error('Error:', error));
                    }
                    setInterval(updateStats, 1000);
                    updateStats();
                </script>
            </body>
            </html>
            """
            return render_template_string(html)

        @self.app.route('/video_feed')
        def video_feed():
            """视频流端点"""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/stats')
        def stats():
            """统计信息API"""
            return jsonify(self.stats)

    def _generate_frames(self):
        """生成MJPEG流"""
        Logger.info("MJPEG流生成器已启动")

        while self.running:
            try:
                # 从队列获取帧（超时1秒）
                frame = self.frame_queue.get(timeout=1.0)

                # 编码为JPEG
                ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except queue.Empty:
                continue
            except Exception as e:
                Logger.error(f"MJPEG流生成异常: {e}")
                break

        Logger.info("MJPEG流生成器结束")

    def _detection_thread(self):
        """检测线程：相机采集 -> 检测 -> 计数"""
        Logger.info("检测线程已启动")

        frame_count = 0

        # 性能统计
        camera_times = []
        detect_times = []
        track_times = []
        count_times = []
        total_times = []

        try:
            while self.running:
                frame_start = time.time()

                # 1. 读取相机帧
                camera_start = time.time()
                ret, frame = self.cap.read()
                camera_time = time.time() - camera_start

                if not ret or frame is None:
                    Logger.warning("相机读取失败")
                    time.sleep(0.1)
                    continue

                frame_count += 1
                camera_times.append(camera_time)

                # 2. YOLO检测
                detect_start = time.time()
                detections_list = self.detector.detect(frame)
                detect_time = time.time() - detect_start
                detect_times.append(detect_time)

                # 过滤目标类别
                target_class = self.config['detector'].get('target_class', 0)
                if len(detections_list) > 0:
                    detections = np.array(detections_list)
                    class_mask = detections[:, 5] == target_class
                    detections = detections[class_mask]
                    filtered_detections = detections.tolist() if len(detections) > 0 else []
                else:
                    filtered_detections = []

                # 3. ByteTrack跟踪
                track_start = time.time()
                tracks = self.tracker.update(filtered_detections)
                track_time = time.time() - track_start
                track_times.append(track_time)

                # 4. 划线计数（使用track_id）
                count_start = time.time()
                counter_result = self.counter.update(tracks, frame.shape[:2])
                count_time = time.time() - count_start
                count_times.append(count_time)

                # 5. Modbus通信（如果触发）- 异步执行
                if counter_result['is_triggered'] and self.modbus_client.enabled:
                    # 在单独线程中发送，不阻塞主检测线程
                    threading.Thread(
                        target=self.modbus_client.send_command,
                        daemon=True
                    ).start()

                # 6. 绘制可视化
                vis_frame = self._draw_frame(frame, tracks, counter_result)

                # 7. 更新帧队列（非阻塞）
                try:
                    self.frame_queue.put_nowait(vis_frame)
                except queue.Full:
                    # 队列满了，丢弃旧帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(vis_frame)
                    except:
                        pass

                # 8. 更新统计信息
                self.stats['current_detections'] = len(tracks)
                self.stats['in_count'] = counter_result['in_count']
                self.stats['out_count'] = counter_result['out_count']
                self.stats['current_count'] = counter_result['current_count']
                self.stats['total_count'] = counter_result['total_count']

                # 总耗时
                total_time = time.time() - frame_start
                total_times.append(total_time)

                # 每10帧打印详细性能统计
                if frame_count % 10 == 0:
                    # 计算最近10帧的平均值
                    recent_camera = camera_times[-10:]
                    recent_detect = detect_times[-10:]
                    recent_track = track_times[-10:]
                    recent_count = count_times[-10:]
                    recent_total = total_times[-10:]

                    avg_camera = sum(recent_camera) / len(recent_camera)
                    avg_detect = sum(recent_detect) / len(recent_detect)
                    avg_track = sum(recent_track) / len(recent_track)
                    avg_count = sum(recent_count) / len(recent_count)
                    avg_total = sum(recent_total) / len(recent_total)

                    camera_fps = 1.0 / avg_camera if avg_camera > 0 else 0
                    detect_fps = 1.0 / avg_detect if avg_detect > 0 else 0
                    track_fps = 1.0 / avg_track if avg_track > 0 else 0
                    count_fps = 1.0 / avg_count if avg_count > 0 else 0
                    total_fps = 1.0 / avg_total if avg_total > 0 else 0

                    Logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                    Logger.info(f"帧 #{frame_count} | 跟踪: {len(tracks)} | IN: {counter_result['in_count']} | OUT: {counter_result['out_count']} | 总计: {counter_result['total_count']}")
                    Logger.info(f"  相机: {avg_camera*1000:6.1f}ms ({camera_fps:5.1f} FPS)")
                    Logger.info(f"  检测: {avg_detect*1000:6.1f}ms ({detect_fps:5.1f} FPS)")
                    Logger.info(f"  跟踪: {avg_track*1000:6.1f}ms ({track_fps:5.1f} FPS)")
                    Logger.info(f"  计数: {avg_count*1000:6.1f}ms ({count_fps:5.1f} FPS)")
                    Logger.info(f"  总计: {avg_total*1000:6.1f}ms ({total_fps:5.1f} FPS)")
                    Logger.info(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

                    # 更新Web统计
                    self.stats['fps'] = total_fps

        except Exception as e:
            Logger.error(f"检测线程异常: {e}")
            import traceback
            Logger.error(traceback.format_exc())

        Logger.info(f"检测线程结束 (共处理 {frame_count} 帧)")

    def _draw_frame(self, frame: np.ndarray, tracks: list, counter_result: dict) -> np.ndarray:
        """绘制跟踪结果"""
        vis_frame = frame.copy()

        # 绘制计数线
        line_coord = counter_result.get('line_coord', 0)
        line_direction = counter_result.get('line_direction', 'horizontal')
        line_color = self.counter.get_line_color()
        line_thickness = self.counter.get_line_thickness()

        if line_direction == 'horizontal':
            cv2.line(vis_frame, (0, line_coord), (vis_frame.shape[1], line_coord),
                    line_color, line_thickness)
            # 绘制线标签
            cv2.putText(vis_frame, "Line", (10, line_coord - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)
        else:
            cv2.line(vis_frame, (line_coord, 0), (line_coord, vis_frame.shape[0]),
                    line_color, line_thickness)
            cv2.putText(vis_frame, "Line", (line_coord + 10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, line_color, 2)

        # 绘制跟踪框和ID（仅在track模式下显示ID）
        counting_mode = self.config['counter'].get('counting_mode', 'track')

        for track in tracks:
            if len(track) < 5:
                continue

            x1, y1, x2, y2, track_id = track[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)

            # 绘制边界框
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制中心点
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            cv2.circle(vis_frame, (center_x, center_y), 4, (0, 0, 255), -1)

            # 仅在track模式下绘制track_id
            if counting_mode == 'track':
                label = f"ID:{track_id}"
                cv2.putText(vis_frame, label, (x1, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 绘制统计信息（显示IN/OUT计数）
        texts = [
            f"FPS: {self.stats['fps']:.1f}",
            f"Tracks: {len(tracks)}",
            f"IN: {counter_result['in_count']}",
            f"OUT: {counter_result['out_count']}",
            f"Current: {counter_result['current_count']}",
            f"Total: {counter_result['total_count']}",
            f"Triggers: {counter_result['trigger_count']}"
        ]

        y_offset = 20
        for i, text in enumerate(texts):
            y = y_offset + i * 20

            # 半透明背景
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (5, y - 15), (150, y + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, vis_frame, 0.4, 0, vis_frame)

            # 绘制文本
            cv2.putText(vis_frame, text, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        return vis_frame

    def load_config(self) -> bool:
        """加载配置文件"""
        try:
            Logger.info(f"加载配置文件: {self.config_path}")

            if not os.path.exists(self.config_path):
                Logger.error(f"配置文件不存在: {self.config_path}")
                return False

            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)

            Logger.info("配置文件加载成功")
            return True

        except Exception as e:
            Logger.error(f"加载配置文件失败: {e}")
            return False

    def start(self) -> bool:
        """启动应用"""
        try:
            # 加载配置
            if not self.load_config():
                return False

            # 1. 打开相机
            Logger.info("打开相机...")
            camera_config = self.config['camera']
            device_id = camera_config['device_id']

            # 判断是摄像头还是视频文件
            if isinstance(device_id, int) or (isinstance(device_id, str) and device_id.isdigit()):
                # 摄像头设备，使用V4L2后端
                device_id = int(device_id) if isinstance(device_id, str) else device_id
                self.cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)
            else:
                # 视频文件或RTSP流，不指定后端
                self.cap = cv2.VideoCapture(device_id)

            if not self.cap.isOpened():
                Logger.error("相机打开失败")
                return False

            # 设置相机参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_config['width'])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_config['height'])
            self.cap.set(cv2.CAP_PROP_FPS, camera_config['fps'])

            Logger.info(f"相机已打开: {camera_config['width']}x{camera_config['height']} @ {camera_config['fps']}fps")

            # 2. 初始化检测器
            Logger.info("初始化检测器...")
            self.detector = YOLOv11Detector(self.config['detector'])

            # 3. 初始化ByteTrack跟踪器
            Logger.info("初始化ByteTrack跟踪器...")
            self.tracker = ByteTracker(self.config['tracker'])

            # 4. 初始化计数器
            Logger.info("初始化计数器...")
            self.counter = ObjectCounter(self.config['counter'])

            # 5. 初始化Modbus
            Logger.info("初始化Modbus...")
            self.modbus_client = ModbusClient(self.config['modbus'])
            if self.modbus_client.enabled:
                self.modbus_client.connect()

            # 6. 设置运行标志
            self.running = True
            self.start_time = time.time()

            # 7. 启动检测线程
            detection_thread = threading.Thread(target=self._detection_thread, daemon=True)
            detection_thread.start()
            Logger.info("检测线程已启动")

            # 7. 启动Flask服务器
            Logger.info("=" * 60)
            Logger.info(f"Web服务器启动: http://{self.host}:{self.port}")
            Logger.info("=" * 60)
            Logger.info("在浏览器中打开上述地址查看实时检测画面")
            Logger.info("按 Ctrl+C 停止服务器")
            Logger.info("=" * 60)

            self.app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                debug=False,
                use_reloader=False
            )

            return True

        except KeyboardInterrupt:
            Logger.info("收到中断信号，正在关闭...")
            return True
        except Exception as e:
            Logger.error(f"启动应用失败: {e}")
            import traceback
            Logger.error(traceback.format_exc())
            return False
        finally:
            self.stop()

    def stop(self):
        """停止应用"""
        if not self.running:
            return

        Logger.info("正在停止应用...")
        self.running = False

        # 等待线程结束
        time.sleep(1.0)

        # 释放相机资源
        if self.cap:
            try:
                self.cap.release()
                cv2.destroyAllWindows()
                Logger.info("相机已关闭")
            except Exception as e:
                Logger.error(f"关闭相机时出错: {e}")

        # 断开Modbus连接
        if self.modbus_client:
            try:
                self.modbus_client.disconnect()
            except Exception as e:
                Logger.error(f"断开Modbus时出错: {e}")

        Logger.info("应用已停止")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='YOLOv11n + ByteTrack Web流媒体版本 V2')

    parser.add_argument('-c', '--config', type=str, default='config/config.yaml',
                       help='配置文件路径 (默认: config/config.yaml)')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                       help='服务器地址 (默认: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=8080,
                       help='服务器端口 (默认: 8080)')

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志
    Logger.setup(level='INFO')

    # 创建应用
    app = SimpleWebApp(
        config_path=args.config,
        host=args.host,
        port=args.port
    )

    # 启动应用
    try:
        app.start()
    except KeyboardInterrupt:
        Logger.info("\n收到键盘中断")
        app.stop()
    except Exception as e:
        Logger.error(f"应用运行异常: {e}")
        app.stop()
        sys.exit(1)


if __name__ == '__main__':
    main()
