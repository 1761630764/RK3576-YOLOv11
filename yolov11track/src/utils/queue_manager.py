"""
队列管理器模块
提供线程安全的队列管理功能
"""

from queue import Queue, Empty, Full
from typing import Any, Optional
import threading
from src.utils.logger import Logger


class QueueManager:
    """队列管理器类"""
    
    def __init__(self, maxsize: int = 10):
        """
        初始化队列管理器
        
        Args:
            maxsize: 队列最大大小
        """
        self.queue = Queue(maxsize=maxsize)
        self.maxsize = maxsize
        self._lock = threading.Lock()
        self._stats = {
            'put_count': 0,
            'get_count': 0,
            'drop_count': 0
        }
    
    def put(self, item: Any, block: bool = True, timeout: Optional[float] = None) -> bool:
        """
        向队列中放入元素
        
        Args:
            item: 要放入的元素
            block: 是否阻塞
            timeout: 超时时间（秒）
            
        Returns:
            是否成功放入
        """
        try:
            self.queue.put(item, block=block, timeout=timeout)
            with self._lock:
                self._stats['put_count'] += 1
            return True
        except Full:
            with self._lock:
                self._stats['drop_count'] += 1
            return False
    
    def get(self, block: bool = True, timeout: Optional[float] = None) -> Optional[Any]:
        """
        从队列中获取元素
        
        Args:
            block: 是否阻塞
            timeout: 超时时间（秒）
            
        Returns:
            获取的元素，如果队列为空则返回None
        """
        try:
            item = self.queue.get(block=block, timeout=timeout)
            with self._lock:
                self._stats['get_count'] += 1
            return item
        except Empty:
            return None
    
    def put_nowait(self, item: Any) -> bool:
        """
        非阻塞方式放入元素
        
        Args:
            item: 要放入的元素
            
        Returns:
            是否成功放入
        """
        return self.put(item, block=False)
    
    def get_nowait(self) -> Optional[Any]:
        """
        非阻塞方式获取元素
        
        Returns:
            获取的元素，如果队列为空则返回None
        """
        return self.get(block=False)
    
    def clear(self) -> int:
        """
        清空队列
        
        Returns:
            清空的元素数量
        """
        count = 0
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                count += 1
            except Empty:
                break
        return count
    
    def size(self) -> int:
        """
        获取队列当前大小
        
        Returns:
            队列大小
        """
        return self.queue.qsize()
    
    def is_empty(self) -> bool:
        """
        判断队列是否为空
        
        Returns:
            是否为空
        """
        return self.queue.empty()
    
    def is_full(self) -> bool:
        """
        判断队列是否已满
        
        Returns:
            是否已满
        """
        return self.queue.full()
    
    def get_stats(self) -> dict:
        """
        获取队列统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'size': self.size(),
                'maxsize': self.maxsize,
                'put_count': self._stats['put_count'],
                'get_count': self._stats['get_count'],
                'drop_count': self._stats['drop_count'],
                'usage': f"{self.size()}/{self.maxsize}"
            }
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        with self._lock:
            self._stats = {
                'put_count': 0,
                'get_count': 0,
                'drop_count': 0
            }


class QueueManagerPool:
    """队列管理器池"""
    
    def __init__(self):
        """初始化队列管理器池"""
        self._queues = {}
        self._lock = threading.Lock()
    
    def create_queue(self, name: str, maxsize: int = 10) -> QueueManager:
        """
        创建队列
        
        Args:
            name: 队列名称
            maxsize: 队列最大大小
            
        Returns:
            队列管理器实例
        """
        with self._lock:
            if name in self._queues:
                Logger.warning(f"队列 '{name}' 已存在，将返回现有队列")
                return self._queues[name]
            
            queue = QueueManager(maxsize=maxsize)
            self._queues[name] = queue
            Logger.info(f"创建队列: {name}, 最大大小: {maxsize}")
            return queue
    
    def get_queue(self, name: str) -> Optional[QueueManager]:
        """
        获取队列
        
        Args:
            name: 队列名称
            
        Returns:
            队列管理器实例，如果不存在则返回None
        """
        with self._lock:
            return self._queues.get(name)
    
    def remove_queue(self, name: str) -> bool:
        """
        移除队列
        
        Args:
            name: 队列名称
            
        Returns:
            是否成功移除
        """
        with self._lock:
            if name in self._queues:
                # 清空队列
                self._queues[name].clear()
                del self._queues[name]
                Logger.info(f"移除队列: {name}")
                return True
            return False
    
    def clear_all(self) -> None:
        """清空所有队列"""
        with self._lock:
            for name, queue in self._queues.items():
                count = queue.clear()
                Logger.info(f"清空队列 '{name}': {count} 个元素")
    
    def get_all_stats(self) -> dict:
        """
        获取所有队列的统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            stats = {}
            for name, queue in self._queues.items():
                stats[name] = queue.get_stats()
            return stats
    
    def print_stats(self) -> None:
        """打印所有队列的统计信息"""
        stats = self.get_all_stats()
        Logger.info("=" * 60)
        Logger.info("队列统计信息:")
        for name, stat in stats.items():
            Logger.info(f"  {name}: {stat['usage']}, "
                       f"放入: {stat['put_count']}, "
                       f"取出: {stat['get_count']}, "
                       f"丢弃: {stat['drop_count']}")
        Logger.info("=" * 60)


if __name__ == '__main__':
    # 测试队列管理器
    print("测试队列管理器...")
    
    # 初始化日志
    Logger.setup()
    
    # 创建队列池
    pool = QueueManagerPool()
    
    # 创建队列
    frame_queue = pool.create_queue('frame_queue', maxsize=10)
    detection_queue = pool.create_queue('detection_queue', maxsize=10)
    
    # 测试放入和获取
    print("\n测试放入和获取...")
    for i in range(5):
        frame_queue.put(f"frame_{i}")
        print(f"放入: frame_{i}, 队列大小: {frame_queue.size()}")
    
    for i in range(3):
        item = frame_queue.get()
        print(f"取出: {item}, 队列大小: {frame_queue.size()}")
    
    # 测试统计信息
    print("\n队列统计信息:")
    stats = frame_queue.get_stats()
    print(f"  大小: {stats['size']}/{stats['maxsize']}")
    print(f"  放入次数: {stats['put_count']}")
    print(f"  取出次数: {stats['get_count']}")
    print(f"  丢弃次数: {stats['drop_count']}")
    
    # 测试队列池统计
    print("\n所有队列统计:")
    pool.print_stats()
    
    # 清空队列
    print("\n清空队列...")
    count = frame_queue.clear()
    print(f"清空了 {count} 个元素")
    
    print("\n测试完成！")
