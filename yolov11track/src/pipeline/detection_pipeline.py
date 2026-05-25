"""
检测流水线模块
主线程顺序执行：相机采集 -> YOLO检测 -> 目标跟踪 -> ID去重计数
子线程异步执行：Modbus通信
"""

import cv2
import time
import threading
import queue
import numpy as np
from typing import Optional, Dict, List
import logging

from src.camera.camera_capture import CameraCapture
from src.detector.yolov11_detector import YOLOv11Detector
from src.tracker.byte_tracker import ByteTracker
from src.counter.object_counter import ObjectCounter
from src.communication.modbus_client import ModbusClient
from src.monitor.performance_monitor import PerformanceMonitor
from src.utils.mjpeg_server import MJPEGServer
from src.utils.video_recorder import VideoRecorder
from src.utils.logger import Logger

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    检测流水线类
    主线程顺序执行架构：采集 -> 检测 -> 跟踪 -> 计数
    子线程异步执行：Modbus通信
    """
    
    def __init__(self, config: dict):
        """
        初始化流水线
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 运行状态
        self.running = False
        self.comm_thread = None
        
        # 通信队列（用于主线程向通信线程传递触发信号）
        self.comm_queue = queue.Queue(maxsize=100)
        
        # 初始化各模块
        logger.info("初始化流水线模块...")
        
        # 1. 相机
        self.camera = CameraCapture(config['camera'])
        
        # 2. 检测器
        self.detector = YOLOv11Detector(config['detector'])
        
        # 3. 跟踪器
        self.tracker = ByteTracker(
            track_thresh=config['tracker']['track_thresh'],
            track_buffer=config['tracker']['track_buffer'],
            match_thresh=config['tracker']['match_thresh'],
            min_box_area=config['tracker']['min_box_area']
        )
        
        # 4. 计数器
        self.counter = ObjectCounter(config['counter'])
        
        # 5. Modbus通信
        self.modbus_client = ModbusClient(config['modbus'])
        
        # 6. 性能监控
        self.performance_monitor = PerformanceMonitor(config['performance_monitor'])
        
        # 可视化配置
        self.vis_config = config['visualization']

        # MJPEG服务器
        self.mjpeg_server = None
        if self.vis_config.get('enable_mjpeg_stream', False):
            mjpeg_port = self.vis_config.get('mjpeg_port', 8080)
            self.mjpeg_server = MJPEGServer(port=mjpeg_port)

        # 视频录制器
        self.video_recorder = None
        if self.vis_config.get('save_video', False):
            self.video_recorder = VideoRecorder(
                output_dir=self.vis_config.get('output_path', 'output/videos'),
                fps=config['camera'].get('fps', 30),
                codec='mp4v',
                max_duration=self.vis_config.get('max_video_duration', 0)
            )

        # 目标类别ID
        self.target_class = config['detector'].get('target_class', 0)
        
        # 统计信息
        self.stats = {
            'frame_count': 0,
            'detection_count': 0,
            'tracking_count': 0,
            'trigger_count': 0
        }

        # 最新的可视化帧（用于Web流）
        self.latest_frame = None
        self._frame_lock = threading.Lock()

        logger.info("流水线初始化完成")
    
    def start(self) -> bool:
        """
        启动流水线
        
        Returns:
            是否成功启动
        """
        if self.running:
            logger.warning("流水线已在运行")
            return False
        
        logger.info("=" * 60)
        logger.info("启动检测流水线")
        logger.info("=" * 60)
        
        # 1. 打开相机
        if not self.camera.open():
            logger.error("相机打开失败")
            return False
        
        # 2. 启动MJPEG服务器（如果启用）
        if self.mjpeg_server:
            self.mjpeg_server.start()

        # 3. 启动视频录制（如果启用）
        if self.video_recorder:
            width = self.camera.width
            height = self.camera.height
            if self.video_recorder.start_recording(width, height, prefix="detection"):
                logger.info("视频录制已启动")
            else:
                logger.warning("视频录制启动失败")

        # 4. 连接Modbus（如果启用）
        if self.modbus_client.enabled:
            self.modbus_client.connect()
        
        # 5. 设置运行标志
        self.running = True

        # 6. 启动通信线程
        self.comm_thread = threading.Thread(
            target=self._communication_thread, 
            name="CommunicationThread", 
            daemon=True
        )
        self.comm_thread.start()
        logger.info("启动线程: CommunicationThread")

        # 7. 创建显示窗口（如果启用）
        if self.vis_config['show_window']:
            cv2.namedWindow('Detection Pipeline', cv2.WINDOW_NORMAL)
        
        logger.info("流水线启动成功")
        return True
    
    def stop(self) -> None:
        """停止流水线"""
        if not self.running:
            logger.warning("流水线未运行")
            return
        
        logger.info("=" * 60)
        logger.info("停止检测流水线")
        logger.info("=" * 60)
        
        # 1. 设置停止标志
        self.running = False
        
        # 2. 等待通信线程结束
        if self.comm_thread and self.comm_thread.is_alive():
            logger.info("等待线程结束: CommunicationThread")
            self.comm_thread.join(timeout=2.0)

        # 3. 停止视频录制
        if self.video_recorder and self.video_recorder.is_recording_active():
            video_file = self.video_recorder.stop_recording()
            if video_file:
                logger.info(f"视频已保存: {video_file}")

        # 4. 停止MJPEG服务器
        if self.mjpeg_server:
            self.mjpeg_server.stop()

        # 5. 关闭相机
        self.camera.close()

        # 6. 断开Modbus
        self.modbus_client.disconnect()

        # 7. 关闭显示窗口
        if self.vis_config['show_window']:
            cv2.destroyAllWindows()
        
        logger.info("流水线已停止")

        # 8. 打印统计信息
        self._print_statistics()
    
    def run_once(self) -> bool:
        """
        执行一次完整的检测流程（主线程顺序执行）

        Returns:
            是否成功执行
        """
        if not self.running:
            return False

        try:
            # 记录开始时间
            start_time = time.time()

            # ===== 步骤1: 相机采集（直接读取，不使用队列）=====
            Logger.info(f"[DEBUG] 开始读取相机帧 #{self.stats['frame_count'] + 1}")
            read_start = time.time()
            ret, frame = self.camera.cap.read()
            read_time = time.time() - read_start
            Logger.info(f"[DEBUG] 相机读取完成: ret={ret}, 耗时={read_time*1000:.1f}ms")

            if not ret or frame is None:
                Logger.warning(f"相机读取失败: ret={ret}, frame={'None' if frame is None else frame.shape}")
                time.sleep(0.1)  # 短暂等待后重试
                return True  # 返回True继续循环，而不是False退出

            self.stats['frame_count'] += 1
            
            # ===== 步骤2: YOLO检测 =====
            detections_list = self.detector.detect(frame)
            
            # 转换为numpy数组并过滤目标类别
            if len(detections_list) > 0:
                detections = np.array(detections_list)
                class_mask = detections[:, 5] == self.target_class
                detections = detections[class_mask]
            else:
                detections = np.empty((0, 6))
            
            self.stats['detection_count'] += len(detections)
            
            # ===== 步骤3: ByteTrack跟踪 =====
            # ByteTracker需要格式：[x1, y1, x2, y2, score]
            if len(detections) > 0:
                det_for_track = detections[:, :5]  # 去掉class列
            else:
                det_for_track = np.empty((0, 5))
            
            tracks = self.tracker.update(det_for_track)
            
            # 转换为列表格式：[[x1, y1, x2, y2, track_id], ...]
            track_list = []
            for track in tracks:
                if track.is_activated:
                    tlbr = track.tlbr
                    track_list.append([
                        tlbr[0], tlbr[1], tlbr[2], tlbr[3], track.track_id
                    ])
            
            self.stats['tracking_count'] += len(track_list)
            
            # ===== 步骤4: ID去重计数 =====
            counter_result = self.counter.update(track_list)
            
            # 如果触发，发送到通信队列
            if counter_result['is_triggered']:
                self.stats['trigger_count'] += 1
                try:
                    self.comm_queue.put_nowait({
                        'current_count': counter_result['current_count'],
                        'total_count': counter_result['total_count']
                    })
                except queue.Full:
                    logger.warning("通信队列已满，丢弃触发信号")
            
            # ===== 步骤5: 可视化 =====
            # 始终生成可视化帧（即使没有检测到目标）
            vis_frame = self._draw_frame(frame, track_list, counter_result)

            # 存储最新帧（用于Web流）
            with self._frame_lock:
                self.latest_frame = vis_frame.copy()

            # 显示窗口
            if self.vis_config['show_window']:
                cv2.imshow('Detection Pipeline', vis_frame)

                # 按'q'退出
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("用户按下'q'键，停止流水线")
                    self.running = False
                    return False

            # 更新MJPEG流
            if self.mjpeg_server:
                self.mjpeg_server.update_frame(vis_frame)

            # 录制视频
            if self.video_recorder and self.video_recorder.is_recording_active():
                self.video_recorder.write_frame(vis_frame)
            
            # ===== 步骤6: 简单性能统计 =====
            frame_time = time.time() - start_time
            if self.performance_monitor.enabled:
                self.performance_monitor.record_time('total', frame_time)
            
            return True
            
        except Exception as e:
            logger.error(f"主循环异常: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _communication_thread(self) -> None:
        """通信线程：异步处理Modbus指令发送"""
        logger.info("通信线程启动")
        
        while self.running:
            try:
                # 从队列获取触发信号（阻塞等待）
                data = self.comm_queue.get(timeout=0.5)
                
                if data is None:
                    continue
                
                logger.info(f"触发Modbus指令: 当前计数={data['current_count']}, "
                          f"累计计数={data['total_count']}")
                
                # 发送Modbus指令
                success = self.modbus_client.send_command()
                
                if success:
                    logger.info("Modbus指令发送成功")
                else:
                    logger.error("Modbus指令发送失败")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"通信线程异常: {e}")
                time.sleep(0.1)
        
        logger.info("通信线程结束")
    
    def _draw_frame(self, frame: np.ndarray, tracks: List, 
                    counter_result: Dict) -> np.ndarray:
        """
        绘制检测结果
        
        Args:
            frame: 原始图像帧
            tracks: 跟踪结果列表
            counter_result: 计数结果
            
        Returns:
            绘制后的图像帧
        """
        vis_frame = frame.copy()
        
        # 绘制跟踪框
        for track in tracks:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 绘制边界框
            color = tuple(self.vis_config['box_color'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 
                        self.vis_config['thickness'])
            
            # 绘制ID
            label = f"ID:{int(track_id)}"
            text_color = tuple(self.vis_config['text_color'])
            cv2.putText(vis_frame, label, (x1, y1 - 5),
                      cv2.FONT_HERSHEY_SIMPLEX,
                      self.vis_config['font_scale'],
                      text_color,
                      self.vis_config['thickness'])
        
        # 绘制统计信息
        self._draw_statistics(vis_frame, counter_result)
        
        return vis_frame
    
    def _draw_statistics(self, frame: np.ndarray, counter_result: Dict) -> None:
        """
        在帧上绘制统计信息
        
        Args:
            frame: 图像帧
            counter_result: 计数结果
        """
        # 获取性能统计
        perf_stats = self.performance_monitor.get_summary()
        
        # 准备文本
        fps = perf_stats.get('total', {}).get('fps', 0.0)
        texts = [
            f"FPS: {fps:.1f}",
            f"Current: {counter_result['current_count']}",
            f"Total: {counter_result['total_count']}",
            f"Triggers: {counter_result['trigger_count']}"
        ]
        
        # 绘制背景
        y_offset = 30
        for i, text in enumerate(texts):
            y = y_offset + i * 30
            
            # 半透明背景
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, y - 20), (250, y + 5), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            
            # 绘制文本
            cv2.putText(frame, text, (15, y),
                       cv2.FONT_HERSHEY_SIMPLEX,
                       0.7,
                       (0, 255, 0),
                       2)
    
    def _print_statistics(self) -> None:
        """打印统计信息"""
        logger.info("=" * 60)
        logger.info("流水线统计信息")
        logger.info("=" * 60)
        
        logger.info(f"总帧数: {self.stats['frame_count']}")
        logger.info(f"检测目标数: {self.stats['detection_count']}")
        logger.info(f"跟踪目标数: {self.stats['tracking_count']}")
        logger.info(f"触发次数: {self.stats['trigger_count']}")
        
        # 计数器统计
        counter_stats = self.counter.get_statistics()
        logger.info(f"当前计数: {counter_stats['current_count']}")
        logger.info(f"累计计数: {counter_stats['total_count']}")
        
        # 性能统计
        perf_stats = self.performance_monitor.get_summary()
        total_stats = perf_stats.get('total', {})
        logger.info(f"平均FPS: {total_stats.get('fps', 0.0):.1f}")
        logger.info(f"平均帧时间: {total_stats.get('avg_time', 0.0)*1000:.1f}ms")
        
        # Modbus统计
        modbus_stats = self.modbus_client.get_statistics()
        logger.info(f"Modbus发送: {modbus_stats['send_count']}")
        logger.info(f"Modbus成功: {modbus_stats['success_count']}")
        logger.info(f"Modbus失败: {modbus_stats['fail_count']}")
        
        # 通信队列统计
        logger.info(f"通信队列大小: {self.comm_queue.qsize()}")
        
        logger.info("=" * 60)
    
    def get_statistics(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        stats = self.stats.copy()
        stats['counter'] = self.counter.get_statistics()
        stats['performance'] = self.performance_monitor.get_summary()
        stats['modbus'] = self.modbus_client.get_statistics()
        stats['comm_queue_size'] = self.comm_queue.qsize()
        
        return stats
    
    def is_running(self) -> bool:
        """
        检查流水线是否运行

        Returns:
            是否运行
        """
        return self.running

    def get_latest_frame(self) -> Optional[np.ndarray]:
        """
        获取最新的可视化帧

        Returns:
            最新的可视化帧，如果没有则返回None
        """
        with self._frame_lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None


if __name__ == "__main__":
    # 测试流水线
    import yaml
    
    print("=" * 60)
    print("检测流水线测试")
    print("=" * 60)
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    try:
        with open('config/config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("\n配置加载成功")
        
        # 创建流水线
        pipeline = DetectionPipeline(config)
        
        # 启动流水线
        if pipeline.start():
            print("\n流水线已启动，按Ctrl+C停止...")
            
            try:
                # 主循环
                while pipeline.is_running():
                    pipeline.run_once()
                
            except KeyboardInterrupt:
                print("\n\n收到停止信号...")
            
            # 停止流水线
            pipeline.stop()
        else:
            print("\n流水线启动失败")
    
    except FileNotFoundError:
        print("\n错误: 配置文件不存在 (config/config.yaml)")
        print("请确保配置文件存在")
    
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
