"""
性能监控模块
使用滑动窗口统计各模块的FPS和耗时
"""

import time
import threading
from collections import deque
from typing import Dict, Optional
from src.utils.logger import Logger


class PerformanceMonitor:
    """性能监控器类"""
    
    def __init__(self, config: dict):
        """
        初始化性能监控器
        
        Args:
            config: 性能监控配置字典
        """
        self.enabled = config.get('enabled', True)
        self.log_interval = config.get('log_interval', 10)  # 日志输出间隔（秒）
        self.window_size = config.get('window_size', 30)    # 滑动窗口大小（帧数）
        self.show_in_ui = config.get('show_in_ui', True)
        
        # 各模块的耗时记录（滑动窗口）
        self._time_windows = {}
        
        # 锁
        self._lock = threading.Lock()
        
        # 日志定时器
        self._last_log_time = time.time()
        
        Logger.info(f"性能监控器初始化: 窗口大小={self.window_size}, "
                   f"日志间隔={self.log_interval}秒")
    
    def record_time(self, module_name: str, elapsed_time: float) -> None:
        """
        记录模块耗时
        
        Args:
            module_name: 模块名称
            elapsed_time: 耗时（秒）
        """
        if not self.enabled:
            return
        
        with self._lock:
            # 如果模块不存在，创建新的滑动窗口
            if module_name not in self._time_windows:
                self._time_windows[module_name] = deque(maxlen=self.window_size)
            
            # 添加耗时记录
            self._time_windows[module_name].append(elapsed_time)
    
    def get_fps(self, module_name: str) -> float:
        """
        获取模块的FPS
        
        Args:
            module_name: 模块名称
            
        Returns:
            FPS值，如果没有数据则返回0.0
        """
        avg_time = self.get_avg_time(module_name)
        if avg_time > 0:
            return 1.0 / avg_time
        return 0.0
    
    def get_avg_time(self, module_name: str) -> float:
        """
        获取模块的平均耗时
        
        Args:
            module_name: 模块名称
            
        Returns:
            平均耗时（秒），如果没有数据则返回0.0
        """
        with self._lock:
            if module_name not in self._time_windows:
                return 0.0
            
            window = self._time_windows[module_name]
            if len(window) == 0:
                return 0.0
            
            return sum(window) / len(window)
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """
        获取所有模块的性能摘要
        
        Returns:
            性能摘要字典，格式：
            {
                'module_name': {
                    'fps': float,
                    'avg_time': float,
                    'avg_time_ms': float
                },
                ...
            }
        """
        summary = {}
        
        with self._lock:
            for module_name in self._time_windows.keys():
                avg_time = self.get_avg_time(module_name)
                fps = self.get_fps(module_name)
                
                summary[module_name] = {
                    'fps': fps,
                    'avg_time': avg_time,
                    'avg_time_ms': avg_time * 1000  # 转换为毫秒
                }
        
        return summary
    
    def log_performance(self, force: bool = False) -> None:
        """
        输出性能日志
        
        Args:
            force: 是否强制输出（忽略时间间隔）
        """
        if not self.enabled:
            return
        
        current_time = time.time()
        elapsed = current_time - self._last_log_time
        
        # 检查是否到达日志输出间隔
        if not force and elapsed < self.log_interval:
            return
        
        # 获取性能摘要
        summary = self.get_summary()
        
        if not summary:
            return
        
        # 输出日志
        Logger.log_performance("=" * 60)
        Logger.log_performance("性能统计:")
        
        for module_name, stats in summary.items():
            Logger.log_performance(
                f"  {module_name:15s}: FPS={stats['fps']:6.1f}, "
                f"平均耗时={stats['avg_time_ms']:6.1f}ms"
            )
        
        Logger.log_performance("=" * 60)
        
        # 更新最后日志时间
        self._last_log_time = current_time
    
    def reset(self) -> None:
        """重置所有统计数据"""
        with self._lock:
            self._time_windows.clear()
            self._last_log_time = time.time()
        
        Logger.info("性能监控器已重置")
    
    def reset_module(self, module_name: str) -> None:
        """
        重置指定模块的统计数据
        
        Args:
            module_name: 模块名称
        """
        with self._lock:
            if module_name in self._time_windows:
                self._time_windows[module_name].clear()
        
        Logger.info(f"模块 '{module_name}' 的性能统计已重置")
    
    def get_module_stats(self, module_name: str) -> Optional[Dict[str, float]]:
        """
        获取指定模块的统计信息
        
        Args:
            module_name: 模块名称
            
        Returns:
            统计信息字典，如果模块不存在则返回None
        """
        summary = self.get_summary()
        return summary.get(module_name)
    
    def print_summary(self) -> None:
        """打印性能摘要到控制台"""
        summary = self.get_summary()
        
        if not summary:
            print("没有性能数据")
            return
        
        print("\n" + "=" * 60)
        print("性能统计摘要:")
        print("-" * 60)
        print(f"{'模块':<15} {'FPS':>8} {'平均耗时(ms)':>15}")
        print("-" * 60)
        
        for module_name, stats in summary.items():
            print(f"{module_name:<15} {stats['fps']:>8.1f} {stats['avg_time_ms']:>15.1f}")
        
        print("=" * 60 + "\n")


class PerformanceTimer:
    """性能计时器上下文管理器"""
    
    def __init__(self, monitor: PerformanceMonitor, module_name: str):
        """
        初始化计时器
        
        Args:
            monitor: 性能监控器实例
            module_name: 模块名称
        """
        self.monitor = monitor
        self.module_name = module_name
        self.start_time = None
    
    def __enter__(self):
        """进入上下文，开始计时"""
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文，记录耗时"""
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            self.monitor.record_time(self.module_name, elapsed_time)
        return False


if __name__ == '__main__':
    # 测试性能监控器
    print("测试性能监控模块...")
    
    # 初始化日志
    Logger.setup()
    
    # 配置
    config = {
        'enabled': True,
        'log_interval': 5,
        'window_size': 30,
        'show_in_ui': True
    }
    
    # 创建监控器
    monitor = PerformanceMonitor(config)
    
    # 模拟记录性能数据
    print("\n模拟记录性能数据...")
    
    import random
    
    for i in range(100):
        # 模拟相机采集耗时
        with PerformanceTimer(monitor, 'camera'):
            time.sleep(random.uniform(0.02, 0.04))  # 20-40ms
        
        # 模拟检测耗时
        with PerformanceTimer(monitor, 'detection'):
            time.sleep(random.uniform(0.03, 0.05))  # 30-50ms
        
        # 模拟跟踪耗时
        with PerformanceTimer(monitor, 'tracking'):
            time.sleep(random.uniform(0.01, 0.03))  # 10-30ms
        
        # 每30帧输出一次统计
        if (i + 1) % 30 == 0:
            print(f"\n处理了 {i + 1} 帧")
            monitor.print_summary()
            
            # 输出日志
            monitor.log_performance(force=True)
    
    # 最终统计
    print("\n最终统计:")
    monitor.print_summary()
    
    # 测试重置
    print("\n测试重置...")
    monitor.reset_module('camera')
    print("已重置 'camera' 模块")
    
    monitor.print_summary()
    
    print("\n测试完成！")
