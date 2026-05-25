"""
视频录制模块
用于保存相机视频流，便于调试和健康检查
"""

import cv2
import numpy as np
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional
from src.utils.logger import Logger


class VideoRecorder:
    """视频录制器类"""

    def __init__(self, output_dir: str = "output/videos",
                 fps: int = 30,
                 codec: str = "mp4v",
                 max_duration: int = 300):
        """
        初始化视频录制器

        Args:
            output_dir: 输出目录
            fps: 视频帧率
            codec: 视频编码器 (mp4v, xvid, h264)
            max_duration: 最大录制时长（秒），0表示无限制
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.fps = fps
        self.codec = codec
        self.max_duration = max_duration

        # 视频写入器
        self.writer = None
        self.current_file = None

        # 录制状态
        self.is_recording = False
        self.start_time = None
        self.frame_count = 0

        # 线程锁
        self._lock = threading.Lock()

        Logger.info(f"视频录制器初始化: 输出目录={output_dir}, FPS={fps}, 编码器={codec}")

    def start_recording(self, width: int, height: int, prefix: str = "video") -> bool:
        """
        开始录制

        Args:
            width: 视频宽度
            height: 视频高度
            prefix: 文件名前缀

        Returns:
            是否成功开始录制
        """
        with self._lock:
            if self.is_recording:
                Logger.warning("已在录制中")
                return False

            try:
                # 生成文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{prefix}_{timestamp}.mp4"
                self.current_file = self.output_dir / filename

                # 创建视频写入器
                fourcc = cv2.VideoWriter_fourcc(*self.codec)
                self.writer = cv2.VideoWriter(
                    str(self.current_file),
                    fourcc,
                    self.fps,
                    (width, height)
                )

                if not self.writer.isOpened():
                    Logger.error("无法创建视频写入器")
                    self.writer = None
                    return False

                self.is_recording = True
                self.start_time = time.time()
                self.frame_count = 0

                Logger.info(f"开始录制: {self.current_file}")
                return True

            except Exception as e:
                Logger.error(f"开始录制失败", e)
                return False

    def write_frame(self, frame: np.ndarray) -> bool:
        """
        写入一帧

        Args:
            frame: 图像帧

        Returns:
            是否成功写入
        """
        with self._lock:
            if not self.is_recording or self.writer is None:
                return False

            try:
                self.writer.write(frame)
                self.frame_count += 1

                # 检查是否超过最大时长
                if self.max_duration > 0:
                    elapsed = time.time() - self.start_time
                    if elapsed >= self.max_duration:
                        Logger.info(f"达到最大录制时长({self.max_duration}秒)，停止录制")
                        self.stop_recording()
                        return False

                return True

            except Exception as e:
                Logger.error(f"写入帧失败", e)
                return False

    def stop_recording(self) -> Optional[str]:
        """
        停止录制

        Returns:
            录制的视频文件路径，如果未录制则返回None
        """
        with self._lock:
            if not self.is_recording:
                return None

            try:
                # 释放写入器
                if self.writer is not None:
                    self.writer.release()
                    self.writer = None

                # 计算统计信息
                duration = time.time() - self.start_time
                actual_fps = self.frame_count / duration if duration > 0 else 0

                Logger.info(f"录制完成: {self.current_file}")
                Logger.info(f"  帧数: {self.frame_count}")
                Logger.info(f"  时长: {duration:.1f}秒")
                Logger.info(f"  实际FPS: {actual_fps:.1f}")

                file_path = str(self.current_file)

                # 重置状态
                self.is_recording = False
                self.start_time = None
                self.frame_count = 0
                self.current_file = None

                return file_path

            except Exception as e:
                Logger.error(f"停止录制失败", e)
                return None

    def is_recording_active(self) -> bool:
        """
        检查是否正在录制

        Returns:
            是否正在录制
        """
        with self._lock:
            return self.is_recording

    def get_stats(self) -> dict:
        """
        获取录制统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            if not self.is_recording:
                return {
                    'is_recording': False,
                    'frame_count': 0,
                    'duration': 0.0,
                    'fps': 0.0
                }

            duration = time.time() - self.start_time
            actual_fps = self.frame_count / duration if duration > 0 else 0

            return {
                'is_recording': True,
                'frame_count': self.frame_count,
                'duration': duration,
                'fps': actual_fps,
                'file': str(self.current_file)
            }

    def __del__(self):
        """析构函数"""
        if self.is_recording:
            self.stop_recording()


if __name__ == '__main__':
    # 测试视频录制器
    print("测试视频录制模块...")

    # 初始化日志
    Logger.setup()

    # 创建录制器
    recorder = VideoRecorder(
        output_dir="output/videos",
        fps=30,
        codec="mp4v",
        max_duration=10  # 测试10秒
    )

    # 打开摄像头
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit(1)

    # 获取摄像头参数
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"摄像头分辨率: {width}x{height}")

    # 开始录制
    if recorder.start_recording(width, height, prefix="test"):
        print("录制已开始，按'q'停止...")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 写入帧
                recorder.write_frame(frame)

                # 显示帧
                cv2.imshow('Recording', frame)

                # 显示录制状态
                stats = recorder.get_stats()
                if stats['is_recording']:
                    print(f"\r录制中: {stats['frame_count']}帧, "
                          f"{stats['duration']:.1f}秒, "
                          f"{stats['fps']:.1f}FPS", end='')

                # 按'q'退出
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                # 检查是否自动停止
                if not recorder.is_recording_active():
                    print("\n达到最大录制时长，自动停止")
                    break

        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            # 停止录制
            video_file = recorder.stop_recording()
            if video_file:
                print(f"\n视频已保存: {video_file}")

            # 释放资源
            cap.release()
            cv2.destroyAllWindows()
            print("测试完成")
    else:
        print("启动录制失败")
        cap.release()
