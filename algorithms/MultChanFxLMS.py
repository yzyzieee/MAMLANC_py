import numpy as np
from scipy import signal
from typing import Tuple, Optional
import warnings

class MultChanFxLMS:
    """
    多参考多通道NFxLMS算法的Python实现
    
    这个类封装了NFxLMS算法，提供了更清晰的接口和更好的性能
    """
    
    def __init__(self, ref_num: int, err_num: int, ctrl_num: int, 
                 filter_len: int, sec_path: np.ndarray, stepsize: float = 0.01,
                 initial_weights: Optional[np.ndarray] = None):
        """
        初始化NFxLMS算法
        
        参数:
        ref_num: 参考信号数量
        err_num: 误差信号数量  
        ctrl_num: 控制源数量
        filter_len: 滤波器长度
        sec_path: 次级路径脉冲响应 (ls, ctrl_num*err_num)
        stepsize: 步长
        initial_weights: 初始控制滤波器权重 (filter_len, ctrl_num*ref_num)
                        如果为None，则从零开始
        """
        self.ref_num = ref_num
        self.err_num = err_num
        self.ctrl_num = ctrl_num
        self.filter_len = filter_len
        self.sec_path = np.array(sec_path)
        self.stepsize = stepsize
        
        # 检查输入参数
        self._validate_inputs()
        
        # 初始化滤波器权重
        self.w_sum = ctrl_num * ref_num
        self.weights = np.zeros((filter_len, self.w_sum))
        
        # 初始化缓存
        self._init_weights(initial_weights)
        self._init_buffers()
        
    def _init_weights(self, initial_weights: Optional[np.ndarray] = None):
        """
        初始化滤波器权重
        
        参数:
        initial_weights: 初始权重矩阵 (filter_len, ctrl_num*ref_num)
                        如果为None，则初始化为零
        """
        if initial_weights is None:
            # 默认初始化为零
            self.weights = np.zeros((self.filter_len, self.w_sum))
        else:
            initial_weights = np.array(initial_weights)
            
            # 检查初始权重的形状
            expected_shape = (self.filter_len, self.w_sum)
            if initial_weights.shape != expected_shape:
                raise ValueError(f"初始权重形状错误: 期望 {expected_shape}, "
                                f"实际 {initial_weights.shape}")
            
            # 使用给定的初始权重
            self.weights = initial_weights.copy()
            
        print(f"权重初始化完成，形状: {self.weights.shape}")
        if initial_weights is not None:
            print(f"使用给定的初始权重，权重范围: [{np.min(self.weights):.4f}, {np.max(self.weights):.4f}]")
        else:
            print("权重初始化为零")
        
    def _validate_inputs(self):
        """验证输入参数的有效性"""
        if self.sec_path.shape[1] != self.ctrl_num * self.err_num:
            raise ValueError(f"次级路径维度错误: 期望 (ls, {self.ctrl_num * self.err_num}), "
                           f"实际 {self.sec_path.shape}")
        
        if self.stepsize <= 0:
            raise ValueError("步长必须大于0")
            
        if any(x <= 0 for x in [self.ref_num, self.err_num, self.ctrl_num, self.filter_len]):
            raise ValueError("所有数量参数必须大于0")
    
    def _init_buffers(self):
        """初始化所有缓存区"""
        self.ls = self.sec_path.shape[0]
        
        # 参考信号缓存
        self.ref_buffer = np.zeros((self.filter_len, self.ref_num))
        
        # 滤波参考信号缓存
        self.filtered_ref_buffer = np.zeros((self.ls, self.ref_num))
        
        # 控制信号缓存
        self.ctrl_buffer = np.zeros((self.ls, self.ctrl_num))
        
        # 权重更新缓存
        self.update_buffer = np.zeros((self.filter_len, self.ctrl_num * self.ref_num))
        
    def process_sample(self, ref_sample: np.ndarray, error_sample: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        处理单个样本
        
        参数:
        ref_sample: 参考信号样本 (ref_num,)
        error_sample: 期望信号样本 (err_num,)
        
        返回:
        control_signal: 控制信号 (ctrl_num,)
        error_signal: 误差信号 (err_num,)
        """
        # 更新参考信号缓存
        self.ref_buffer = np.roll(self.ref_buffer, 1, axis=0)
        self.ref_buffer[0, :] = ref_sample
        
        self.filtered_ref_buffer = np.roll(self.filtered_ref_buffer, 1, axis=0)
        self.filtered_ref_buffer[0, :] = ref_sample
        
        # 计算控制信号
        control_signal = self._compute_control_signal()
        
        # 更新控制信号缓存
        self.ctrl_buffer = np.roll(self.ctrl_buffer, 1, axis=0)
        self.ctrl_buffer[0, :] = control_signal
        
        # 计算次级信号
        secondary_signal = self._compute_secondary_signal()
        
        # 计算误差信号
        error_signal = error_sample - secondary_signal
        
        # 更新权重
        self._update_weights(error_signal)
        
        return control_signal, error_signal
    
    def _compute_control_signal(self) -> np.ndarray:
        """计算控制信号"""
        control_signal = np.zeros(self.ctrl_num)
        
        for ctrl_idx in range(self.ctrl_num):
            # 获取对应这个控制源的所有滤波器权重
            weight_start = ctrl_idx * self.ref_num
            weight_end = (ctrl_idx + 1) * self.ref_num
            weights_ctrl = self.weights[:, weight_start:weight_end]  # (filter_len, ref_num)
            
            # 计算控制信号
            control_signal[ctrl_idx] = np.sum(weights_ctrl * self.ref_buffer)
            
        return control_signal
    
    def _compute_secondary_signal(self) -> np.ndarray:
        """计算次级信号（在误差点的信号）"""
        secondary_signal = np.zeros(self.err_num)
        
        for err_idx in range(self.err_num):
            for ctrl_idx in range(self.ctrl_num):
                # 获取对应的次级路径
                sec_path_idx = ctrl_idx * self.err_num + err_idx
                sec_path_impulse = self.sec_path[:, sec_path_idx]
                
                # 卷积计算
                secondary_signal[err_idx] += np.dot(sec_path_impulse, self.ctrl_buffer[:, ctrl_idx])
        
        return secondary_signal
    
    def _update_weights(self, error_signal: np.ndarray):
        """更新权重"""
        # 计算滤波参考信号
        filtered_ref = np.zeros((self.ctrl_num * self.ref_num, 1))
        
        idx = 0
        for ctrl_idx in range(self.ctrl_num):
            for ref_idx in range(self.ref_num):
                for err_idx in range(self.err_num):
                    sec_path_idx = ctrl_idx * self.err_num + err_idx
                    sec_path_impulse = self.sec_path[:, sec_path_idx]
                    filtered_ref[idx, 0] += np.dot(sec_path_impulse, 
                                                 self.filtered_ref_buffer[:, ref_idx])
                idx += 1
        
        # 更新滤波参考信号缓存
        self.update_buffer = np.roll(self.update_buffer, 1, axis=0)
        self.update_buffer[0, :] = filtered_ref.flatten()
        
        # 更新权重
        for i in range(self.w_sum):
            ctrl_idx = i // self.ref_num
            ref_idx = i % self.ref_num
            
            # 计算梯度
            gradient = np.zeros(self.filter_len)
            for err_idx in range(self.err_num):
                update_idx = ctrl_idx * self.err_num * self.ref_num + ref_idx * self.err_num + err_idx
                gradient += self.update_buffer[:, update_idx] * error_signal[err_idx]
            
            # 权重更新
            self.weights[:, i] -= self.stepsize * gradient
    
    def process_batch(self, ref_signals: np.ndarray, error_signals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        批量处理信号
        
        参数:
        ref_signals: 参考信号 (length, ref_num)
        error_signals: 期望信号 (length, err_num)
        
        返回:
        control_signals: 控制信号 (length, ctrl_num)
        error_out: 输出误差信号 (length, err_num)
        """
        ref_signals = np.atleast_2d(ref_signals)
        error_signals = np.atleast_2d(error_signals)
        
        if ref_signals.ndim == 1:
            ref_signals = ref_signals.reshape(-1, 1)
        if error_signals.ndim == 1:
            error_signals = error_signals.reshape(-1, 1)
            
        length = ref_signals.shape[0]
        
        # 输出缓存
        control_signals = np.zeros((length, self.ctrl_num))
        error_out = np.zeros((length, self.err_num))
        
        # 逐样本处理
        for n in range(length):
            ctrl_sig, err_sig = self.process_sample(ref_signals[n, :], error_signals[n, :])
            control_signals[n, :] = ctrl_sig
            error_out[n, :] = err_sig
            
        return control_signals, error_out
    
    def reset(self):
        """重置算法状态"""
        self.weights.fill(0)
        self._init_buffers()
    
    def get_weights(self) -> np.ndarray:
        """获取当前权重"""
        return self.weights.copy()
    
    def set_stepsize(self, stepsize: float):
        """设置步长"""
        if stepsize <= 0:
            raise ValueError("步长必须大于0")
        self.stepsize = stepsize


# 兼容原函数的包装器
def multi_ref_multi_chan_fxlms(ref: np.ndarray, e: np.ndarray, 
                                 filter_len: int, sec_path: np.ndarray, 
                                 stepsize: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    原MATLAB函数的兼容包装器
    
    参数:
    ref: 参考信号 (length, ref_num)
    e: 误差信号 (length, err_num)
    filter_len: 滤波器长度
    sec_path: 次级路径 (ls, ctrl_num*err_num)  
    stepsize: 步长
    
    返回:
    ww: 最终权重
    ew: 误差信号
    """
    # 输入处理
    ref = np.atleast_2d(ref)
    e = np.atleast_2d(e)
    
    if ref.ndim == 1:
        ref = ref.reshape(-1, 1)
    if e.ndim == 1:
        e = e.reshape(-1, 1)
    
    # 获取参数
    ref_num = ref.shape[1]
    err_num = e.shape[1]
    chn_sum = sec_path.shape[1]
    ctrl_num = chn_sum // err_num
    
    # 创建算法实例
    algorithm = MultChanFxLMS(
        ref_num=ref_num,
        err_num=err_num, 
        ctrl_num=ctrl_num,
        filter_len=filter_len,
        sec_path=sec_path,
        stepsize=stepsize
    )
    
    # 处理信号
    _, error_out = algorithm.process_batch(ref, e)
    
    # 返回结果
    return algorithm.get_weights(), error_out
