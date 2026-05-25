"""
Modbus TCP通信模块
用于发送控制指令到PLC或其他设备
"""

import time
import threading
from typing import Optional
from pymodbus.client import ModbusTcpClient
from pymodbus.exceptions import ModbusException
from src.utils.logger import Logger


class ModbusClient:
    """Modbus TCP客户端类"""
    
    def __init__(self, config: dict):
        """
        初始化Modbus客户端
        
        Args:
            config: Modbus配置字典
        """
        self.enabled = config.get('enabled', True)
        self.host = config.get('host', '192.168.1.100')
        self.port = config.get('port', 502)
        self.unit_id = config.get('unit_id', 1)
        self.register_address = config.get('register_address', 0)
        self.command_value = config.get('command_value', 1)
        self.timeout = config.get('timeout', 5)
        self.retry_times = config.get('retry_times', 3)
        self.keep_alive = config.get('keep_alive', True)  # 保持连接

        # Modbus客户端对象
        self.client: Optional[ModbusTcpClient] = None
        self.connected = False
        
        # 统计信息
        self.send_count = 0
        self.success_count = 0
        self.fail_count = 0

        # 锁
        self._lock = threading.Lock()
        self._connect_lock = threading.Lock()  # 连接锁，防止并发重连

        # 心跳机制
        self.heartbeat_interval = config.get('heartbeat_interval', 10)  # 心跳间隔（秒）
        self.heartbeat_enabled = config.get('heartbeat_enabled', True) and self.keep_alive
        self._heartbeat_thread = None
        self._heartbeat_stop = threading.Event()
        self._last_activity = time.time()  # 最后活动时间

        if self.enabled:
            Logger.info(f"Modbus客户端初始化: {self.host}:{self.port}, "
                       f"从站ID={self.unit_id}, "
                       f"寄存器地址={self.register_address}, "
                       f"保持连接={'是' if self.keep_alive else '否'}")

            # 如果启用保持连接，立即建立连接
            if self.keep_alive:
                self.connect()

                # 启动心跳线程
                if self.heartbeat_enabled:
                    self._start_heartbeat()
                    Logger.info(f"心跳机制已启动，间隔={self.heartbeat_interval}秒")
        else:
            Logger.info("Modbus通信已禁用")
    
    def connect(self) -> bool:
        """
        连接到Modbus服务器

        Returns:
            是否成功连接
        """
        if not self.enabled:
            return False

        # 使用连接锁，防止多个线程同时尝试连接
        with self._connect_lock:
            # 再次检查连接状态（可能其他线程已经连接）
            if self.connected and self.client and self.client.connected:
                return True

            try:
                Logger.info(f"连接Modbus服务器: {self.host}:{self.port}")

                self.client = ModbusTcpClient(
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout
                )

                # 尝试连接
                self.connected = self.client.connect()

                if self.connected:
                    Logger.info("Modbus连接成功")
                else:
                    Logger.error("Modbus连接失败")

                return self.connected

            except Exception as e:
                Logger.error(f"Modbus连接异常", e)
                self.connected = False
                return False
    
    def disconnect(self) -> None:
        """断开Modbus连接"""
        # 停止心跳线程
        if self._heartbeat_thread is not None:
            self._stop_heartbeat()

        if self.client is not None:
            try:
                self.client.close()
                self.connected = False
                Logger.info("Modbus连接已断开")
            except Exception as e:
                Logger.error(f"断开Modbus连接异常", e)

    def _start_heartbeat(self) -> None:
        """启动心跳线程"""
        if self._heartbeat_thread is None or not self._heartbeat_thread.is_alive():
            self._heartbeat_stop.clear()
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop,
                daemon=True,
                name="ModbusHeartbeat"
            )
            self._heartbeat_thread.start()

    def _stop_heartbeat(self) -> None:
        """停止心跳线程"""
        if self._heartbeat_thread is not None:
            self._heartbeat_stop.set()
            self._heartbeat_thread.join(timeout=2)
            self._heartbeat_thread = None
            Logger.debug("心跳线程已停止")

    def _heartbeat_loop(self) -> None:
        """心跳循环"""
        while not self._heartbeat_stop.is_set():
            try:
                # 检查距离上次活动的时间
                time_since_activity = time.time() - self._last_activity

                # 如果超过心跳间隔且连接有效，发送心跳
                if time_since_activity >= self.heartbeat_interval and self.connected:
                    self._send_heartbeat()

                # 等待一段时间再检查
                self._heartbeat_stop.wait(timeout=1)

            except Exception as e:
                Logger.warning(f"心跳循环异常: {str(e)}")
                time.sleep(1)

    def _send_heartbeat(self) -> None:
        """发送心跳（读取寄存器）"""
        try:
            with self._connect_lock:
                if self.client and self.connected:
                    # 读取一个寄存器作为心跳
                    response = self.client.read_holding_registers(
                        address=self.register_address,
                        count=1
                    )

                    if not response.isError():
                        self._last_activity = time.time()
                        Logger.debug(f"心跳发送成功")
                    else:
                        Logger.debug(f"心跳失败: {response}")
                        # 不标记为断开，让正常发送时处理

        except Exception as e:
            Logger.debug(f"心跳异常: {str(e)}")
            # 不标记为断开，让正常发送时处理
    
    def send_command(self, value: Optional[int] = None) -> bool:
        """
        发送控制指令
        
        Args:
            value: 要写入的值，如果为None则使用配置的默认值
            
        Returns:
            是否成功发送
        """
        if not self.enabled:
            Logger.debug("Modbus通信已禁用，跳过发送")
            return True
        
        if value is None:
            value = self.command_value
        
        with self._lock:
            self.send_count += 1
        
        # 尝试发送（带重试）
        success = self._retry_send(value, self.retry_times)
        
        with self._lock:
            if success:
                self.success_count += 1
            else:
                self.fail_count += 1
        
        return success
    
    def _retry_send(self, value: int, max_retries: int) -> bool:
        """
        带重试的发送

        Args:
            value: 要写入的值
            max_retries: 最大重试次数

        Returns:
            是否成功发送
        """
        # 使用连接锁保护整个发送过程，防止并发冲突
        with self._connect_lock:
            for attempt in range(max_retries):
                try:
                    # 检查连接状态（使用pymodbus的connected属性）
                    if not self.connected or (self.client and not self.client.connected):
                        if not self.connect():
                            Logger.warning(f"Modbus未连接，重试 {attempt + 1}/{max_retries}")
                            time.sleep(0.5)
                            continue

                    # 记录开始时间
                    start_time = time.time()

                    # 写单个寄存器（Function Code 0x06）
                    # pymodbus 3.x 不需要unit/slave参数
                    response = self.client.write_register(
                        address=self.register_address,
                        value=value
                    )

                    # 计算耗时
                    elapsed_time = time.time() - start_time

                    # 检查响应
                    if response.isError():
                        Logger.error(f"Modbus写入失败: {response}")
                        # 不要设置 self.connected = False，因为连接可能仍然有效
                        # 只有在连接异常时才标记为断开

                        if attempt < max_retries - 1:
                            Logger.warning(f"重试 {attempt + 1}/{max_retries}")
                            time.sleep(0.5)
                            continue
                        else:
                            return False
                    else:
                        Logger.log_event(f"Modbus指令发送成功: "
                                       f"寄存器={self.register_address}, "
                                       f"值={value}, "
                                       f"耗时={elapsed_time*1000:.1f}ms")

                        # 更新最后活动时间
                        self._last_activity = time.time()

                        # 如果不保持连接，发送成功后断开
                        if not self.keep_alive:
                            self.disconnect()

                        return True

                except ModbusException as e:
                    Logger.warning(f"Modbus异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    # 只有在连接相关的异常时才标记为断开
                    error_msg = str(e).lower()
                    if 'connection' in error_msg or 'timeout' in error_msg or 'closed' in error_msg:
                        self.connected = False
                        Logger.warning("检测到连接异常，标记为断开")

                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        Logger.error(f"Modbus发送失败，已重试{max_retries}次")
                        return False

                except Exception as e:
                    Logger.warning(f"发送指令异常 (尝试 {attempt + 1}/{max_retries}): {str(e)}")
                    # 只有在连接相关的异常时才标记为断开
                    error_msg = str(e).lower()
                    if 'connection' in error_msg or 'timeout' in error_msg or 'closed' in error_msg:
                        self.connected = False
                        Logger.warning("检测到连接异常，标记为断开")

                    if attempt < max_retries - 1:
                        time.sleep(0.5)
                    else:
                        Logger.error(f"Modbus发送失败，已重试{max_retries}次")
                        return False

            return False
    
    def read_register(self, address: Optional[int] = None) -> Optional[int]:
        """
        读取寄存器值
        
        Args:
            address: 寄存器地址，如果为None则使用配置的地址
            
        Returns:
            寄存器值，失败则返回None
        """
        if not self.enabled:
            return None
        
        if address is None:
            address = self.register_address
        
        try:
            # 如果未连接，尝试连接
            if not self.connected:
                if not self.connect():
                    return None
            
            # 读保持寄存器（Function Code 0x03）
            # pymodbus 3.x 不需要slave参数
            response = self.client.read_holding_registers(
                address=address,
                count=1
            )
            
            if response.isError():
                Logger.error(f"Modbus读取失败: {response}")
                return None
            else:
                value = response.registers[0]
                Logger.debug(f"读取寄存器 {address}: {value}")
                return value
                
        except Exception as e:
            Logger.error(f"读取寄存器异常", e)
            return None
    
    def get_connection_status(self) -> bool:
        """
        获取连接状态
        
        Returns:
            是否已连接
        """
        return self.connected
    
    def get_statistics(self) -> dict:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'enabled': self.enabled,
                'connected': self.connected,
                'send_count': self.send_count,
                'success_count': self.success_count,
                'fail_count': self.fail_count,
                'success_rate': (self.success_count / self.send_count * 100) 
                               if self.send_count > 0 else 0.0
            }
    
    def print_statistics(self) -> None:
        """打印统计信息"""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("Modbus统计:")
        print("-" * 60)
        print(f"  启用状态: {'是' if stats['enabled'] else '否'}")
        print(f"  连接状态: {'已连接' if stats['connected'] else '未连接'}")
        print(f"  发送次数: {stats['send_count']}")
        print(f"  成功次数: {stats['success_count']}")
        print(f"  失败次数: {stats['fail_count']}")
        print(f"  成功率: {stats['success_rate']:.1f}%")
        print("=" * 60 + "\n")
    
    def __del__(self):
        """析构函数"""
        self.disconnect()


if __name__ == '__main__':
    # 测试Modbus客户端
    print("测试Modbus通信模块...")
    
    # 初始化日志
    Logger.setup()
    
    # 配置（使用模拟服务器地址）
    config = {
        'enabled': True,
        'host': '127.0.0.1',  # 本地测试
        'port': 502,
        'unit_id': 1,
        'register_address': 0,
        'command_value': 1,
        'timeout': 5,
        'retry_times': 3
    }
    
    # 创建客户端
    client = ModbusClient(config)
    
    # 测试连接
    print("\n测试连接...")
    if client.connect():
        print("连接成功")
        
        # 测试发送指令
        print("\n测试发送指令...")
        for i in range(5):
            success = client.send_command(i + 1)
            print(f"发送指令 {i + 1}: {'成功' if success else '失败'}")
            time.sleep(1)
        
        # 测试读取寄存器
        print("\n测试读取寄存器...")
        value = client.read_register()
        if value is not None:
            print(f"寄存器值: {value}")
        else:
            print("读取失败")
        
        # 断开连接
        client.disconnect()
    else:
        print("连接失败（这是正常的，因为没有实际的Modbus服务器）")
    
    # 打印统计
    print("\n统计信息:")
    client.print_statistics()
    
    print("\n测试完成！")
    print("注意：实际使用时需要配置正确的Modbus服务器地址")
