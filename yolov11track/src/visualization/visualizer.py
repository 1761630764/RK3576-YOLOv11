"""
可视化模块
绘制检测框、跟踪ID、计数信息和性能指标
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from src.utils.logger import Logger


class Visualizer:
    """可视化器类"""
    
    def __init__(self, config: dict):
        """
        初始化可视化器
        
        Args:
            config: 可视化配置字典
        """
        self.show_window = config.get('show_window', True)
        self.save_video = config.get('save_video', False)
        self.output_path = config.get('output_path', 'output/')
        self.show_fps = config.get('show_fps', True)
        self.box_color = tuple(config.get('box_color', [0, 255, 0]))
        self.text_color = tuple(config.get('text_color', [255, 255, 255]))
        self.font_scale = config.get('font_scale', 0.6)
        self.thickness = config.get('thickness', 2)
        
        # 视频写入器
        self.video_writer = None
        self.video_initialized = False
        
        # 窗口名称
        self.window_name = "YOLOv8n + ByteTrack"
        
        # 保存最后一帧（用于Web流媒体）
        self.last_frame = None
        
        # 创建输出目录
        if self.save_video:
            Path(self.output_path).mkdir(parents=True, exist_ok=True)
        
        Logger.info(f"可视化器初始化: 显示窗口={self.show_window}, "
                   f"保存视频={self.save_video}")
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       tracks: List[List[float]],
                       show_id: bool = True) -> np.ndarray:
        """
        绘制检测框和跟踪ID
        
        Args:
            image: 输入图像
            tracks: 跟踪结果列表，格式：[[x1, y1, x2, y2, track_id], ...]
            show_id: 是否显示跟踪ID
            
        Returns:
            绘制后的图像
        """
        vis_image = image.copy()
        
        for track in tracks:
            if len(track) < 5:
                continue
            
            x1, y1, x2, y2, track_id = track[:5]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)
            
            # 绘制检测框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), 
                         self.box_color, self.thickness)
            
            # 绘制跟踪ID
            if show_id:
                label = f"ID:{track_id}"
                
                # 计算文字大小
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 
                    self.font_scale, self.thickness
                )
                
                # 绘制文字背景
                cv2.rectangle(vis_image, 
                            (x1, y1 - text_height - baseline - 5),
                            (x1 + text_width, y1),
                            self.box_color, -1)
                
                # 绘制文字
                cv2.putText(vis_image, label, 
                           (x1, y1 - baseline - 2),
                           cv2.FONT_HERSHEY_SIMPLEX,
                           self.font_scale, self.text_color, 
                           self.thickness)
        
        return vis_image
    
    def draw_info_panel(self,
                       image: np.ndarray,
                       counter_stats: Optional[Dict] = None,
                       perf_stats: Optional[Dict] = None) -> np.ndarray:
        """
        绘制信息面板
        
        Args:
            image: 输入图像
            counter_stats: 计数统计信息
            perf_stats: 性能统计信息
            
        Returns:
            绘制后的图像
        """
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        # 面板参数
        panel_height = 120
        panel_color = (0, 0, 0)
        panel_alpha = 0.7
        
        # 创建半透明面板
        overlay = vis_image.copy()
        cv2.rectangle(overlay, (0, h - panel_height), (w, h), 
                     panel_color, -1)
        cv2.addWeighted(overlay, panel_alpha, vis_image, 1 - panel_alpha, 
                       0, vis_image)
        
        # 文字参数
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)
        line_height = 25
        start_y = h - panel_height + 20
        
        # 绘制计数信息
        if counter_stats:
            current = counter_stats.get('current_count', 0)
            total = counter_stats.get('total_count', 0)
            triggers = counter_stats.get('trigger_count', 0)
            
            text = f"Count: {current} | Total: {total} | Triggers: {triggers}"
            cv2.putText(vis_image, text, (10, start_y),
                       font, font_scale, color, thickness)
        
        # 绘制性能信息
        if perf_stats and self.show_fps:
            y_offset = start_y + line_height
            
            # 相机FPS
            if 'camera' in perf_stats:
                fps = perf_stats['camera'].get('fps', 0)
                text = f"Camera: {fps:.1f} FPS"
                cv2.putText(vis_image, text, (10, y_offset),
                           font, font_scale * 0.8, color, thickness - 1)
            
            # 检测FPS
            if 'detection' in perf_stats:
                fps = perf_stats['detection'].get('fps', 0)
                text = f"Detection: {fps:.1f} FPS"
                cv2.putText(vis_image, text, (200, y_offset),
                           font, font_scale * 0.8, color, thickness - 1)
            
            # 跟踪FPS
            if 'tracking' in perf_stats:
                fps = perf_stats['tracking'].get('fps', 0)
                text = f"Tracking: {fps:.1f} FPS"
                cv2.putText(vis_image, text, (400, y_offset),
                           font, font_scale * 0.8, color, thickness - 1)
            
            # 总FPS
            if 'total' in perf_stats:
                fps = perf_stats['total'].get('fps', 0)
                y_offset += line_height
                text = f"Total FPS: {fps:.1f}"
                cv2.putText(vis_image, text, (10, y_offset),
                           font, font_scale, (0, 255, 0), thickness)
        
        return vis_image
    
    def draw_detection_line(self, 
                           image: np.ndarray,
                           line_position: float = 0.5,
                           orientation: str = 'horizontal') -> np.ndarray:
        """
        绘制检测线（用于线穿越计数）
        
        Args:
            image: 输入图像
            line_position: 线的位置（0-1之间）
            orientation: 方向（'horizontal' 或 'vertical'）
            
        Returns:
            绘制后的图像
        """
        vis_image = image.copy()
        h, w = vis_image.shape[:2]
        
        line_color = (0, 0, 255)  # 红色
        line_thickness = 2
        
        if orientation == 'horizontal':
            y = int(h * line_position)
            cv2.line(vis_image, (0, y), (w, y), line_color, line_thickness)
        else:  # vertical
            x = int(w * line_position)
            cv2.line(vis_image, (x, 0), (x, h), line_color, line_thickness)
        
        return vis_image
    
    def show(self, image: np.ndarray, wait_key: int = 1) -> int:
        """
        显示图像
        
        Args:
            image: 要显示的图像
            wait_key: 等待按键时间（毫秒）
            
        Returns:
            按键值
        """
        # 保存最后一帧
        self.last_frame = image.copy()
        
        if not self.show_window:
            return -1
        
        cv2.imshow(self.window_name, image)
        key = cv2.waitKey(wait_key) & 0xFF
        
        return key
    
    def get_last_frame(self) -> Optional[np.ndarray]:
        """
        获取最后一帧图像（用于Web流媒体）
        
        Returns:
            最后一帧图像，如果没有则返回None
        """
        return self.last_frame
    
    def save_frame(self, image: np.ndarray) -> None:
        """
        保存帧到视频文件
        
        Args:
            image: 要保存的图像
        """
        if not self.save_video:
            return
        
        # 初始化视频写入器
        if not self.video_initialized:
            h, w = image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # 生成输出文件名
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = Path(self.output_path) / f"output_{timestamp}.mp4"
            
            self.video_writer = cv2.VideoWriter(
                str(output_file), fourcc, 20.0, (w, h)
            )
            
            self.video_initialized = True
            Logger.info(f"开始保存视频: {output_file}")
        
        # 写入帧
        if self.video_writer is not None:
            self.video_writer.write(image)
    
    def release(self) -> None:
        """释放资源"""
        if self.video_writer is not None:
            self.video_writer.release()
            Logger.info("视频已保存")
        
        if self.show_window:
            cv2.destroyAllWindows()
        
        Logger.info("可视化器已释放")
    
    def __del__(self):
        """析构函数"""
        self.release()


if __name__ == '__main__':
    # 测试可视化器
    print("测试可视化模块...")
    
    # 初始化日志
    from src.utils.logger import Logger
    Logger.setup()
    
    # 配置
    config = {
        'show_window': True,
        'save_video': False,
        'output_path': 'output/',
        'show_fps': True,
        'box_color': [0, 255, 0],
        'text_color': [255, 255, 255],
        'font_scale': 0.6,
        'thickness': 2
    }
    
    # 创建可视化器
    visualizer = Visualizer(config)
    
    # 创建测试图像
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 模拟跟踪结果
    tracks = [
        [100, 100, 200, 200, 1],
        [300, 150, 400, 250, 2],
        [150, 300, 250, 400, 3]
    ]
    
    # 模拟统计信息
    counter_stats = {
        'current_count': 5,
        'total_count': 125,
        'trigger_count': 12
    }
    
    perf_stats = {
        'camera': {'fps': 30.2},
        'detection': {'fps': 25.5},
        'tracking': {'fps': 28.1},
        'total': {'fps': 24.8}
    }
    
    print("\n绘制测试图像...")
    
    # 绘制检测框
    vis_image = visualizer.draw_detections(image, tracks)
    
    # 绘制信息面板
    vis_image = visualizer.draw_info_panel(vis_image, counter_stats, perf_stats)
    
    # 绘制检测线
    vis_image = visualizer.draw_detection_line(vis_image, 0.5, 'horizontal')
    
    # 显示
    print("显示图像（按'q'退出）...")
    while True:
        key = visualizer.show(vis_image, 1)
        if key == ord('q'):
            break
    
    # 释放资源
    visualizer.release()
    
    print("\n测试完成！")
