"""
配置加载器模块
负责加载和验证YAML配置文件
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import os


class ConfigLoader:
    """配置加载器类"""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载YAML配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML格式错误
        """
        config_file = Path(config_path)
        
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证配置
            if not ConfigLoader.validate_config(config):
                raise ValueError("配置验证失败")
            
            # 处理路径
            config = ConfigLoader._process_paths(config)
            
            return config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"YAML解析错误: {e}")
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """
        验证配置文件的完整性
        
        Args:
            config: 配置字典
            
        Returns:
            验证是否通过
        """
        required_sections = [
            'camera',
            'detector',
            'tracker',
            'counter',
            'visualization',
            'performance_monitor',
            'modbus'
        ]
        
        # 检查必需的配置节
        for section in required_sections:
            if section not in config:
                print(f"警告: 缺少配置节 '{section}'")
                return False
        
        # 检查相机配置
        if 'device_id' not in config['camera']:
            print("错误: 缺少 camera.device_id")
            return False
        
        # 检查检测器配置
        if 'model_path' not in config['detector']:
            print("错误: 缺少 detector.model_path")
            return False
        
        return True
    
    @staticmethod
    def _process_paths(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理配置中的路径，转换为绝对路径
        
        Args:
            config: 配置字典
            
        Returns:
            处理后的配置字典
        """
        # 获取项目根目录
        project_root = Path(__file__).parent.parent.parent
        
        # 处理模型路径
        if 'detector' in config and 'model_path' in config['detector']:
            model_path = config['detector']['model_path']
            if not Path(model_path).is_absolute():
                config['detector']['model_path'] = str(project_root / model_path)
        
        # 处理输出路径
        if 'visualization' in config and 'output_path' in config['visualization']:
            output_path = config['visualization']['output_path']
            if not Path(output_path).is_absolute():
                config['visualization']['output_path'] = str(project_root / output_path)
        
        return config
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        获取默认配置
        
        Returns:
            默认配置字典
        """
        return {
            'camera': {
                'device_id': 0,
                'width': 640,
                'height': 480,
                'fps': 30,
                'buffer_size': 10
            },
            'detector': {
                'model_path': 'models/yolov8n_int8.rknn',
                'input_size': 640,
                'conf_threshold': 0.25,
                'nms_threshold': 0.45,
                'target_class': 0
            },
            'tracker': {
                'track_thresh': 0.5,
                'track_buffer': 30,
                'match_thresh': 0.8,
                'min_box_area': 100
            },
            'counter': {
                'count_threshold': 10,
                'counting_mode': 'id',
                'reset_on_trigger': True,
                'max_tracked_ids': 1000
            },
            'visualization': {
                'show_window': True,
                'save_video': False,
                'output_path': 'output/',
                'show_fps': True,
                'box_color': [0, 255, 0],
                'text_color': [255, 255, 255]
            },
            'performance_monitor': {
                'enabled': True,
                'log_interval': 10,
                'window_size': 30,
                'show_in_ui': True
            },
            'modbus': {
                'enabled': True,
                'host': '192.168.1.100',
                'port': 502,
                'unit_id': 1,
                'register_address': 0,
                'command_value': 1,
                'timeout': 5,
                'retry_times': 3
            }
        }
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str) -> None:
        """
        保存配置到文件
        
        Args:
            config: 配置字典
            config_path: 保存路径
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        print(f"配置已保存到: {config_path}")


if __name__ == '__main__':
    # 测试配置加载器
    print("测试配置加载器...")
    
    # 获取默认配置
    default_config = ConfigLoader.get_default_config()
    print("\n默认配置:")
    print(yaml.dump(default_config, default_flow_style=False))
    
    # 保存默认配置
    ConfigLoader.save_config(default_config, 'config/config.yaml')
    
    # 加载配置
    try:
        config = ConfigLoader.load_config('config/config.yaml')
        print("\n配置加载成功!")
        print(f"检测器模型路径: {config['detector']['model_path']}")
    except Exception as e:
        print(f"配置加载失败: {e}")
