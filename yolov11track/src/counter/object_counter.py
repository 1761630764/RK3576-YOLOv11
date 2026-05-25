"""
目标计数模块
实现基于划线的计数逻辑，结合ByteTrack的track_id避免重复计数
"""

import threading
from typing import List, Dict, Tuple
from src.utils.logger import Logger


class ObjectCounter:
    """目标计数器类 - 基于划线计数，使用track_id避免重复计数"""

    def __init__(self, config: dict):
        """
        初始化计数器

        Args:
            config: 计数器配置字典
        """
        self.count_threshold = config.get('count_threshold', 10)
        self.reset_on_trigger = config.get('reset_on_trigger', True)
        self.trigger_mode = config.get('trigger_mode', 'total')  # total/in/out/net
        self.counting_mode = config.get('counting_mode', 'track')  # track/line

        # 计数线配置
        line_config = config.get('counting_line', {})
        self.line_position = line_config.get('position', 0.5)  # 线的位置（0-1，相对于图像高度/宽度）
        self.line_direction = line_config.get('direction', 'horizontal')  # horizontal/vertical
        self.line_thickness = line_config.get('thickness', 2)
        self.line_color = tuple(line_config.get('color', [0, 255, 0]))  # BGR格式

        # 计数统计（区分IN/OUT方向）
        self.in_count = 0           # IN方向计数（水平线：向下；垂直线：向右）
        self.out_count = 0          # OUT方向计数（水平线：向上；垂直线：向左）
        self.current_count = 0      # 当前总计数（in_count + out_count）
        self.total_count = 0        # 累计总计数
        self.trigger_count = 0      # 触发次数

        # 记录每个track_id的历史位置（用于判断是否穿越线）
        # 格式: {track_id: {'prev_center': (x, y), 'counted': False}}
        self.track_history: Dict[int, Dict] = {}
        self.counted_ids = []       # 已计数的track_id列表（仅track模式使用）

        # 锁
        self._lock = threading.Lock()

        Logger.info(f"计数器初始化: 阈值={self.count_threshold}, "
                   f"线位置={self.line_position}, "
                   f"方向={self.line_direction}, "
                   f"计数模式={self.counting_mode}, "
                   f"触发模式={self.trigger_mode}, "
                   f"触发后清零={self.reset_on_trigger}")

    def update(self, tracks: List[List[float]], frame_shape: Tuple[int, int]) -> Dict:
        """
        更新计数（使用ByteTrack的track_id，支持IN/OUT方向区分）

        Args:
            tracks: 跟踪结果列表，格式：[[x1, y1, x2, y2, track_id], ...]
            frame_shape: 图像尺寸 (height, width)

        Returns:
            统计信息字典，包含：
            {
                'in_count': int,           # IN方向计数
                'out_count': int,          # OUT方向计数
                'current_count': int,      # 当前总计数
                'total_count': int,        # 累计总计数
                'trigger_count': int,      # 触发次数
                'is_triggered': bool,      # 是否触发
                'new_objects': int,        # 本次新增目标数
                'line_coord': int,         # 计数线的坐标（用于绘制）
                'line_direction': str      # 计数线方向
            }
        """
        with self._lock:
            new_objects = 0
            frame_height, frame_width = frame_shape

            # 计算计数线的位置
            if self.line_direction == 'horizontal':
                line_coord = int(frame_height * self.line_position)
            else:
                line_coord = int(frame_width * self.line_position)

            # 处理每个跟踪目标
            for track in tracks:
                if len(track) < 5:
                    continue

                x1, y1, x2, y2, track_id = track[:5]
                track_id = int(track_id)

                # 计算中心点
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                # 如果是新的track_id，记录其位置
                if track_id not in self.track_history:
                    self.track_history[track_id] = {
                        'prev_center': (center_x, center_y),
                        'counted': False
                    }
                    continue

                # 获取历史位置
                history = self.track_history[track_id]
                prev_center = history['prev_center']

                # 检查是否穿越计数线
                crossed = False
                direction = None  # 'IN' or 'OUT'

                if self.line_direction == 'horizontal':
                    # 水平线：判断是否穿越
                    if (prev_center[1] < line_coord <= center_y) or \
                       (prev_center[1] > line_coord >= center_y):
                        crossed = True
                        # 判断方向：向下为IN，向上为OUT
                        if center_y > prev_center[1]:
                            direction = 'IN'
                        else:
                            direction = 'OUT'
                else:
                    # 垂直线：判断是否穿越
                    if (prev_center[0] < line_coord <= center_x) or \
                       (prev_center[0] > line_coord >= center_x):
                        crossed = True
                        # 判断方向：向右为IN，向左为OUT
                        if center_x > prev_center[0]:
                            direction = 'IN'
                        else:
                            direction = 'OUT'

                # 根据计数模式处理
                if crossed:
                    should_count = False

                    if self.counting_mode == 'track':
                        # track模式：每个track_id只计数一次
                        if track_id not in self.counted_ids:
                            should_count = True
                            self.counted_ids.append(track_id)
                    else:
                        # line模式：每次穿越都计数（可能重复计数同一目标）
                        should_count = True

                    if should_count:
                        if direction == 'IN':
                            self.in_count += 1
                        else:
                            self.out_count += 1

                        self.current_count = self.in_count + self.out_count
                        self.total_count += 1
                        new_objects += 1

                        Logger.debug(f"目标 ID={track_id} 穿越计数线 [{direction}] (模式={self.counting_mode}): "
                                   f"IN={self.in_count}, OUT={self.out_count}, "
                                   f"当前总计={self.current_count}, 累计={self.total_count}")

                # 更新位置
                history['prev_center'] = (center_x, center_y)

            # 清理不再活跃的track_id（可选，避免内存无限增长）
            active_track_ids = {int(track[4]) for track in tracks if len(track) >= 5}
            ids_to_remove = [tid for tid in self.track_history.keys() if tid not in active_track_ids]
            for tid in ids_to_remove:
                del self.track_history[tid]

            # 检查是否触发
            is_triggered = self._check_trigger()

            # 返回统计信息
            return {
                'in_count': self.in_count,
                'out_count': self.out_count,
                'current_count': self.current_count,
                'total_count': self.total_count,
                'trigger_count': self.trigger_count,
                'is_triggered': is_triggered,
                'new_objects': new_objects,
                'line_coord': line_coord,
                'line_direction': self.line_direction
            }

    def _check_trigger(self) -> bool:
        """
        检查是否达到触发阈值

        Returns:
            是否触发
        """
        # 根据触发模式选择计数值
        if self.trigger_mode == 'in':
            trigger_value = self.in_count
        elif self.trigger_mode == 'out':
            trigger_value = self.out_count
        elif self.trigger_mode == 'net':
            trigger_value = self.in_count - self.out_count
        else:  # 'total' 或其他默认值
            trigger_value = self.current_count

        if trigger_value >= self.count_threshold:
            self.trigger_count += 1

            Logger.log_event(f"计数达到阈值: 触发模式={self.trigger_mode}, "
                           f"触发值={trigger_value}, "
                           f"阈值={self.count_threshold}, "
                           f"触发次数={self.trigger_count}")

            # 如果配置为触发后清零
            if self.reset_on_trigger:
                Logger.log_event(f"当前计数清零，累计计数={self.total_count}")
                self.in_count = 0
                self.out_count = 0
                self.current_count = 0
                # 清空历史记录和已计数ID，重新开始计数
                self.track_history.clear()
                self.counted_ids.clear()

            return True

        return False

    def reset_current_count(self) -> None:
        """重置当前计数"""
        with self._lock:
            self.in_count = 0
            self.out_count = 0
            self.current_count = 0
            self.counted_ids.clear()
            self.track_history.clear()
            Logger.info("当前计数已重置")

    def reset_all(self) -> None:
        """重置所有计数"""
        with self._lock:
            self.in_count = 0
            self.out_count = 0
            self.current_count = 0
            self.total_count = 0
            self.trigger_count = 0
            self.counted_ids.clear()
            self.track_history.clear()
            Logger.info("所有计数已重置")

    def get_statistics(self) -> Dict:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        with self._lock:
            return {
                'in_count': self.in_count,
                'out_count': self.out_count,
                'current_count': self.current_count,
                'total_count': self.total_count,
                'trigger_count': self.trigger_count,
                'tracked_count': len(self.track_history),
                'counted_ids_count': len(self.counted_ids),
                'count_threshold': self.count_threshold,
                'trigger_mode': self.trigger_mode,
                'line_position': self.line_position,
                'line_direction': self.line_direction
            }

    def set_threshold(self, threshold: int) -> None:
        """
        设置触发阈值

        Args:
            threshold: 新的阈值
        """
        with self._lock:
            old_threshold = self.count_threshold
            self.count_threshold = threshold
            Logger.info(f"触发阈值已更新: {old_threshold} -> {threshold}")

    def get_line_color(self) -> Tuple[int, int, int]:
        """获取计数线颜色"""
        return self.line_color

    def get_line_thickness(self) -> int:
        """获取计数线粗细"""
        return self.line_thickness

    def print_statistics(self) -> None:
        """打印统计信息"""
        stats = self.get_statistics()

        print("\n" + "=" * 60)
        print("计数统计:")
        print("-" * 60)
        print(f"  IN方向计数: {stats['in_count']}")
        print(f"  OUT方向计数: {stats['out_count']}")
        print(f"  当前总计数: {stats['current_count']}")
        print(f"  累计总计数: {stats['total_count']}")
        print(f"  触发次数: {stats['trigger_count']}")
        print(f"  跟踪目标数: {stats['tracked_count']}")
        print(f"  已计数ID数: {stats['counted_ids_count']}")
        print(f"  触发阈值: {stats['count_threshold']}")
        print(f"  触发模式: {stats['trigger_mode']}")
        print(f"  线位置: {stats['line_position']}")
        print(f"  线方向: {stats['line_direction']}")
        print("=" * 60 + "\n")
