"""
ByteTrack跟踪器实现
优化参数配置，适配工业场景
"""

import numpy as np
from collections import deque
import lap


class STrack:
    """单个跟踪目标"""

    shared_kalman = None
    track_id_count = 0

    def __init__(self, tlwh, score):
        """
        初始化跟踪目标

        Args:
            tlwh: [x, y, w, h] 边界框
            score: 置信度分数
        """
        # 边界框 [x, y, w, h]
        self._tlwh = np.asarray(tlwh, dtype=np.float32)

        # 卡尔曼滤波器状态
        self.kalman_filter = None
        self.mean, self.covariance = None, None

        self.is_activated = False
        self.score = score
        self.tracklet_len = 0

        self.track_id = 0
        self.state = 'new'  # new, tracked, lost, removed

        self.frame_id = 0
        self.start_frame = 0

    def predict(self):
        """使用卡尔曼滤波器预测下一帧位置"""
        mean_state = self.mean.copy()
        if self.state != 'tracked':
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """批量预测多个目标"""
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != 'tracked':
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """激活新的跟踪目标"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xyah(self._tlwh))

        self.tracklet_len = 0
        self.state = 'tracked'
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """重新激活丢失的目标"""
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_track.tlwh)
        )
        self.tracklet_len = 0
        self.state = 'tracked'
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """更新跟踪目标"""
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xyah(new_tlwh)
        )
        self.state = 'tracked'
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """获取当前边界框 [x, y, w, h]"""
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """获取边界框 [x1, y1, x2, y2]"""
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """转换 [x, y, w, h] 到 [cx, cy, aspect_ratio, h]"""
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def to_xyah(self):
        """转换到 [cx, cy, aspect_ratio, h]"""
        return self.tlwh_to_xyah(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """转换 [x1, y1, x2, y2] 到 [x, y, w, h]"""
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        """转换 [x, y, w, h] 到 [x1, y1, x2, y2]"""
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return f'OT_{self.track_id}_({self.start_frame}-{self.frame_id})'

    @staticmethod
    def next_id():
        """生成下一个track_id"""
        STrack.track_id_count += 1
        return STrack.track_id_count


class KalmanFilter:
    """卡尔曼滤波器用于目标跟踪"""

    def __init__(self):
        ndim, dt = 4, 1.

        # 创建卡尔曼滤波器矩阵
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # 运动和观测不确定性权重
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """初始化跟踪状态"""
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """预测下一帧状态"""
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """投影状态到测量空间"""
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """批量预测"""
        std_pos = [
            self._std_weight_position * mean[:, 3],
            self._std_weight_position * mean[:, 3],
            1e-2 * np.ones_like(mean[:, 3]),
            self._std_weight_position * mean[:, 3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[:, 3],
            self._std_weight_velocity * mean[:, 3],
            1e-5 * np.ones_like(mean[:, 3]),
            self._std_weight_velocity * mean[:, 3]
        ]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = []
        for i in range(len(mean)):
            motion_cov.append(np.diag(sqr[i]))
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """更新状态"""
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(covariance, self._update_mat.T).T,
            check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance


class ByteTracker:
    """ByteTrack跟踪器 - 优化参数配置"""

    def __init__(self, config: dict):
        """
        初始化ByteTrack跟踪器

        Args:
            config: 跟踪器配置字典
        """
        # 优化后的参数配置
        self.track_thresh = config.get('track_thresh', 0.5)  # 高置信度阈值
        self.track_buffer = config.get('track_buffer', 30)   # 跟踪缓冲帧数
        self.match_thresh = config.get('match_thresh', 0.8)  # 匹配阈值
        self.min_box_area = config.get('min_box_area', 10)   # 最小框面积

        # 低置信度阈值（ByteTrack关键参数）
        self.low_thresh = config.get('low_thresh', 0.1)

        # 跟踪状态
        self.tracked_stracks = []  # 正在跟踪的目标
        self.lost_stracks = []     # 丢失的目标
        self.removed_stracks = []  # 移除的目标

        self.frame_id = 0
        self.kalman_filter = KalmanFilter()

        # 设置共享卡尔曼滤波器
        STrack.shared_kalman = self.kalman_filter

        from src.utils.logger import Logger
        Logger.info(f"ByteTrack初始化: track_thresh={self.track_thresh}, "
                   f"low_thresh={self.low_thresh}, "
                   f"match_thresh={self.match_thresh}, "
                   f"track_buffer={self.track_buffer}")

    def update(self, detections):
        """
        更新跟踪器

        Args:
            detections: 检测结果 [[x1, y1, x2, y2, score, class_id], ...]

        Returns:
            跟踪结果 [[x1, y1, x2, y2, track_id], ...]
        """
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(detections) == 0:
            detections = np.empty((0, 6))
        else:
            detections = np.array(detections)

        # 分离高低置信度检测
        scores = detections[:, 4]
        remain_inds = scores > self.track_thresh
        inds_low = scores > self.low_thresh
        inds_high = scores < self.track_thresh

        inds_second = np.logical_and(inds_low, inds_high)
        dets_second = detections[inds_second]
        dets = detections[remain_inds]

        # 转换为tlwh格式
        if len(dets) > 0:
            detections_tlwh = self.tlbr_to_tlwh(dets[:, :4])
            detections_first = [STrack(tlwh, score) for tlwh, score in zip(detections_tlwh, dets[:, 4])]
        else:
            detections_first = []

        if len(dets_second) > 0:
            detections_second_tlwh = self.tlbr_to_tlwh(dets_second[:, :4])
            detections_second = [STrack(tlwh, score) for tlwh, score in zip(detections_second_tlwh, dets_second[:, 4])]
        else:
            detections_second = []

        # 添加新检测到跟踪池
        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # 第一次关联：高置信度检测 + 跟踪目标
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
        STrack.multi_predict(strack_pool)
        dists = matching_iou_distance(strack_pool, detections_first)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections_first[idet]
            if track.state == 'tracked':
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        # 第二次关联：低置信度检测 + 未匹配的跟踪目标
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 'tracked']
        dists_second = []
        if len(dets_second) > 0 and len(r_tracked_stracks) > 0:
            detections_second_filtered = [detections_second[i] for i in u_detection if i < len(detections_second)]
            dists_second = matching_iou_distance(r_tracked_stracks, detections_second_filtered)
            matches, u_track_second, u_detection_second = linear_assignment(dists_second, thresh=0.5)
            for itracked, idet in matches:
                track = r_tracked_stracks[itracked]
                det = detections_second_filtered[idet]
                if track.state == 'tracked':
                    track.update(det, self.frame_id)
                    activated_starcks.append(track)
                else:
                    track.re_activate(det, self.frame_id, new_id=False)
                    refind_stracks.append(track)

            # 处理未匹配的跟踪目标（使用第二次匹配后的未匹配索引）
            for it in u_track_second:
                track = r_tracked_stracks[it]
                if not track.state == 'lost':
                    track.state = 'lost'
                    lost_stracks.append(track)
        else:
            # 如果没有进行第二次匹配，所有r_tracked_stracks都标记为lost
            for track in r_tracked_stracks:
                if not track.state == 'lost':
                    track.state = 'lost'
                    lost_stracks.append(track)

        # 处理未确认的跟踪目标
        detections_first = [detections_first[i] for i in u_detection]
        dists = matching_iou_distance(unconfirmed, detections_first)
        matches, u_unconfirmed, u_detection = linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections_first[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.state = 'removed'
            removed_stracks.append(track)

        # 初始化新的跟踪目标
        for inew in u_detection:
            track = detections_first[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        # 移除长时间丢失的目标
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                track.state = 'removed'
                removed_stracks.append(track)

        # 更新跟踪状态
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 'tracked']
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # 输出结果
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]

        # 转换为 [[x1, y1, x2, y2, track_id], ...]
        results = []
        for track in output_stracks:
            tlbr = track.tlbr
            results.append([tlbr[0], tlbr[1], tlbr[2], tlbr[3], track.track_id])

        return results

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        """转换 [x1, y1, x2, y2] 到 [x, y, w, h]"""
        ret = np.asarray(tlbr).copy()
        ret[:, 2:] -= ret[:, :2]
        return ret


def matching_iou_distance(atracks, btracks):
    """计算IoU距离矩阵"""
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]

    _ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if _ious.size == 0:
        return _ious

    _ious = bbox_ious(np.ascontiguousarray(atlbrs, dtype=np.float32),
                      np.ascontiguousarray(btlbrs, dtype=np.float32))

    cost_matrix = 1 - _ious
    return cost_matrix


def bbox_ious(atlbrs, btlbrs):
    """计算IoU"""
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    if ious.size == 0:
        return ious

    for i, atlbr in enumerate(atlbrs):
        for j, btlbr in enumerate(btlbrs):
            ious[i, j] = bbox_iou(atlbr, btlbr)
    return ious


def bbox_iou(boxA, boxB):
    """计算两个框的IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou


def linear_assignment(cost_matrix, thresh):
    """线性分配算法"""
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))

    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def joint_stracks(tlista, tlistb):
    """合并跟踪列表"""
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    """从列表a中移除列表b中的跟踪"""
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    """移除重复的跟踪"""
    pdist = matching_iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


# 简化的scipy替代（避免依赖scipy）
class scipy:
    class linalg:
        @staticmethod
        def cho_factor(a, lower=True, check_finite=True):
            """Cholesky分解"""
            L = np.linalg.cholesky(a)
            return (L, lower)

        @staticmethod
        def cho_solve(c_and_lower, b, check_finite=True):
            """使用Cholesky分解求解"""
            c, lower = c_and_lower
            if lower:
                y = np.linalg.solve(c, b)
                x = np.linalg.solve(c.T, y)
            else:
                y = np.linalg.solve(c.T, b)
                x = np.linalg.solve(c, y)
            return x
