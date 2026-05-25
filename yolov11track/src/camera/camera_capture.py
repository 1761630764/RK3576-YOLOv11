"""
相机采集模块
支持USB摄像头和RTSP流的多线程采集
"""

import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple
from queue import Queue, Full
from src.utils.logger import Logger


class CameraCapture:
    """相机采集类"""
    
    def __init__(self, config: dict):
        """
        初始化相机采集器
        
        Args:
            config: 相机配置字典
        """
        self.device_id = config.get('device_id', 0)
        self.width = config.get('width', 640)
        self.height = config.get('height', 480)
        self.target_fps = config.get('fps', 30)
        self.buffer_size = config.get('buffer_size', 10)
        
        # 相机对象
        self.cap = None
        
        # 帧队列
        self.frame_queue = Queue(maxsize=self.buffer_size)
        
        # 线程控制
        self.thread = None
        self.running = False
        self.paused = False
        
        # 统计信息
        self.frame_count = 0
        self.drop_count = 0
        self.fps = 0.0
        self.last_time = time.time()
        self.fps_update_interval = 1.0  # 每秒更新一次FPS
        
        # 锁
        self._lock = threading.Lock()
        
        Logger.info(f"相机采集器初始化: device={self.device_id}, "
                   f"分辨率={self.width}x{self.height}, "
                   f"目标FPS={self.target_fps}")
    
    def open(self) -> bool:
        """
        打开相机
        
        Returns:
            是否成功打开
        """
        try:
            # 判断是RTSP流还是USB摄像头
            if isinstance(self.device_id, str) and self.device_id.startswith('rtsp'):
                Logger.info(f"打开RTSP流: {self.device_id}")
                self.cap = cv2.VideoCapture(self.device_id)
            else:
                Logger.info(f"打开USB摄像头: {self.device_id}")
                self.cap = cv2.VideoCapture(self.device_id)
            
            if not self.cap.isOpened():
                Logger.error(f"无法打开摄像头: {self.device_id}")
                return False
            
            # 设置MJPEG编码格式（提高带宽和帧率）
            # MJPEG通常比YUYV有更高的帧率支持
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M','J','P','G'))
            
            # 设置分辨率和帧率
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # 减少缓冲区大小，降低延迟
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 禁用自动曝光和自动对焦以提高性能（可选）
            # 注意：某些相机可能不支持这些属性
            try:
                self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)  # 禁用自动对焦
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 手动曝光模式
            except:
                pass  # 忽略不支持的属性
            
            # 获取实际配置
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
            
            Logger.info(f"相机已打开: 实际分辨率={actual_width}x{actual_height}, "
                       f"实际FPS={actual_fps}, 编码格式={fourcc_str}")
            
            # 如果实际FPS仍然很低，给出警告和建议
            if actual_fps < self.target_fps * 0.5:
                Logger.warning(f"实际FPS({actual_fps})远低于目标FPS({self.target_fps})")
                Logger.warning("建议检查:")
                Logger.warning("  1. USB连接是否为USB 3.0")
                Logger.warning("  2. 相机驱动是否正确安装")
                Logger.warning("  3. 系统资源是否充足")
                Logger.warning("  4. 尝试降低分辨率以提高帧率")
            
            return True
            
        except Exception as e:
            Logger.error(f"打开相机失败", e)
            return False
    
    def close(self) -> None:
        """关闭相机"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            Logger.info("相机已关闭")
    
    def start(self) -> bool:
        """
        启动采集线程
        
        Returns:
            是否成功启动
        """
        if self.running:
            Logger.warning("采集线程已在运行")
            return True
        
        # 打开相机
        if not self.open():
            return False
        
        # 启动线程
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()
        
        Logger.info("采集线程已启动")
        return True
    
    def stop(self) -> None:
        """停止采集线程"""
        if not self.running:
            return
        
        self.running = False
        
        # 等待线程结束
        if self.thread is not None:
            self.thread.join(timeout=2.0)
        
        # 关闭相机
        self.close()
        
        # 清空队列
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except:
                break
        
        Logger.info("采集线程已停止")
    
    def pause(self) -> None:
        """暂停采集"""
        self.paused = True
        Logger.info("采集已暂停")
    
    def resume(self) -> None:
        """恢复采集"""
        self.paused = False
        Logger.info("采集已恢复")
    
    def _capture_loop(self) -> None:
        """采集循环（在独立线程中运行）"""
        Logger.info("采集循环开始")
        
        consecutive_failures = 0
        max_failures = 10
        
        while self.running:
            try:
                # 如果暂停，则等待
                if self.paused:
                    time.sleep(0.1)
                    continue
                
                # 读取帧
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    consecutive_failures += 1
                    Logger.warning(f"读取帧失败 ({consecutive_failures}/{max_failures})")
                    
                    if consecutive_failures >= max_failures:
                        Logger.error("连续读取失败次数过多，尝试重新连接")
                        self._reconnect()
                        consecutive_failures = 0
                    
                    time.sleep(0.1)
                    continue
                
                # 重置失败计数
                consecutive_failures = 0
                
                # 更新统计信息
                with self._lock:
                    self.frame_count += 1
                    
                    # 更新FPS
                    current_time = time.time()
                    elapsed = current_time - self.last_time
                    if elapsed >= self.fps_update_interval:
                        self.fps = self.frame_count / elapsed
                        self.frame_count = 0
                        self.last_time = current_time
                
                # 放入队列
                try:
                    self.frame_queue.put(frame, block=False)
                except Full:
                    # 队列已满，丢弃最旧的帧
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put(frame, block=False)
                        with self._lock:
                            self.drop_count += 1
                    except:
                        pass
                
                # 控制帧率
                time.sleep(1.0 / self.target_fps)
                
            except Exception as e:
                Logger.error(f"采集循环异常", e)
                time.sleep(0.1)
        
        Logger.info("采集循环结束")
    
    def _reconnect(self) -> bool:
        """
        重新连接相机
        
        Returns:
            是否成功重连
        """
        Logger.info("尝试重新连接相机...")
        
        # 关闭当前连接
        if self.cap is not None:
            self.cap.release()
            time.sleep(1.0)
        
        # 重新打开
        return self.open()
    
    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        获取一帧图像
        
        Args:
            timeout: 超时时间（秒）
            
        Returns:
            图像帧，如果超时则返回None
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except:
            return None
    
    def get_fps(self) -> float:
        """
        获取当前FPS
        
        Returns:
            FPS值
        """
        with self._lock:
            return self.fps
    
    def get_stats(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'fps': self.fps,
                'queue_size': self.frame_queue.qsize(),
                'buffer_size': self.buffer_size,
                'drop_count': self.drop_count,
                'running': self.running,
                'paused': self.paused
            }
    
    def is_running(self) -> bool:
        """
        检查是否正在运行
        
        Returns:
            是否运行中
        """
        return self.running
    
    def __del__(self):
        """析构函数"""
        self.stop()


if __name__ == '__main__':
    # 测试相机采集
    print("测试相机采集模块...")
    
    # 初始化日志
    Logger.setup()
    
    # 配置
    config = {
        'device_id': 0,
        'width': 640,
        'height': 480,
        'fps': 30,
        'buffer_size': 10
    }
    
    # 创建采集器
    camera = CameraCapture(config)
    
    # 启动采集
    if camera.start():
        print("采集已启动，按Ctrl+C停止...")
        
        try:
            frame_count = 0
            while True:
                # 获取帧
                frame = camera.get_frame(timeout=1.0)
                
                if frame is not None:
                    frame_count += 1
                    
                    # 显示帧
                    cv2.imshow('Camera Test', frame)
                    
                    # 每30帧输出一次统计
                    if frame_count % 30 == 0:
                        stats = camera.get_stats()
                        print(f"FPS: {stats['fps']:.1f}, "
                              f"队列: {stats['queue_size']}/{stats['buffer_size']}, "
                              f"丢帧: {stats['drop_count']}")
                    
                    # 按'q'退出
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("获取帧超时")
                    
        except KeyboardInterrupt:
            print("\n停止采集...")
        finally:
            camera.stop()
            cv2.destroyAllWindows()
            print("测试完成")
    else:
        print("启动采集失败")
