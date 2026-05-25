"""
日志系统模块
使用loguru实现统一的日志管理
"""

from loguru import logger
from pathlib import Path
import sys
from typing import Optional
from datetime import datetime


class Logger:
    """日志管理器类"""
    
    _initialized = False
    
    @classmethod
    def setup(cls, log_dir: str = "logs", level: str = "INFO") -> None:
        """
        初始化日志系统（仅终端输出）

        Args:
            log_dir: 日志目录（已废弃，保留参数兼容性）
            level: 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        if cls._initialized:
            return

        # 移除默认处理器
        logger.remove()

        # 只添加控制台输出（彩色）
        logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            level=level,
            colorize=True
        )

        cls._initialized = True
        logger.info("日志系统初始化完成（仅终端输出）")
    
    @staticmethod
    def get_logger():
        """
        获取logger实例
        
        Returns:
            logger实例
        """
        if not Logger._initialized:
            Logger.setup()
        return logger
    
    @staticmethod
    def log_performance(message: str) -> None:
        """
        记录性能日志
        
        Args:
            message: 日志消息
        """
        logger.bind(performance=True).info(message)
    
    @staticmethod
    def log_event(message: str) -> None:
        """
        记录事件日志
        
        Args:
            message: 日志消息
        """
        logger.bind(event=True).info(message)
    
    @staticmethod
    def debug(message: str) -> None:
        """记录DEBUG级别日志"""
        logger.debug(message)
    
    @staticmethod
    def info(message: str) -> None:
        """记录INFO级别日志"""
        logger.info(message)
    
    @staticmethod
    def warning(message: str) -> None:
        """记录WARNING级别日志"""
        logger.warning(message)
    
    @staticmethod
    def error(message: str, exception: Optional[Exception] = None) -> None:
        """
        记录ERROR级别日志
        
        Args:
            message: 日志消息
            exception: 异常对象（可选）
        """
        if exception:
            logger.error(f"{message}: {exception}")
            logger.exception(exception)
        else:
            logger.error(message)
    
    @staticmethod
    def critical(message: str) -> None:
        """记录CRITICAL级别日志"""
        logger.critical(message)


# 便捷函数
def setup_logger(log_dir: str = "logs", level: str = "INFO") -> None:
    """
    初始化日志系统（便捷函数）
    
    Args:
        log_dir: 日志目录
        level: 日志级别
    """
    Logger.setup(log_dir, level)


def get_logger():
    """获取logger实例（便捷函数）"""
    return Logger.get_logger()


if __name__ == '__main__':
    # 测试日志系统
    print("测试日志系统...")
    
    # 初始化日志
    Logger.setup(log_dir="logs", level="DEBUG")
    
    # 测试不同级别的日志
    Logger.debug("这是一条DEBUG日志")
    Logger.info("这是一条INFO日志")
    Logger.warning("这是一条WARNING日志")
    Logger.error("这是一条ERROR日志")
    
    # 测试性能日志
    Logger.log_performance("相机FPS: 30.5, 平均耗时: 33.1ms")
    Logger.log_performance("检测FPS: 25.2, 平均耗时: 39.7ms")
    
    # 测试事件日志
    Logger.log_event("当前计数达到阈值: 10")
    Logger.log_event("Modbus指令发送成功")
    
    # 测试异常日志
    try:
        raise ValueError("测试异常")
    except Exception as e:
        Logger.error("捕获到异常", e)
    
    print("\n日志测试完成！请查看 logs/ 目录")
