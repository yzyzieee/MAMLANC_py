# ANC-MAML Python 项目开发规范 (Development Guide)
# 本文档是针对 ANC + MAML 项目的开发指南，根据当前经过的设计，规范化各个模块的功能、接口和文件结构，便于维护和扩展。

项目目录结构:
anc_maml_project/ 
├── main.py                            # 🔷 主控脚本：协调训练、测试与可视化，，全局参数设置（路径、Fs、滤波器长度等） 
│ 
├── dataloader/ 
│   ├── generate_data.py               # 🔶 路径加载、白噪合成、训练样本生成  
│ 
├── models/ 
│   ├── control_filter.py              # 🔶 控制器结构封装（Phi = 参数向量） 
│   └── maml_filter.py                 # 🔶 封装 MAML 更新逻辑（内循环） 
│ 
├── algorithms/ 
│   └── fxlms.py                       # 🔶 多参考 FxLMS 算法（用于测试对比） 
│ 
├── evaluation/ 
│   ├── mse_plot.py                    # 🔶 滑动均方误差计算 + 图像输出 
│ 
├── utils/ 
│   ├── mat_io.py                      # 🔶 加载 .mat 文件并重采样 
│   └── signal_utils.py                # 🔶 常用信号函数（滤波、normalize、resample） 
│ 
├── checkpoints/                       # ⬜ 模型参数保存位置 
└── AGENT.md                           # 📘 开发指南和环境配置

1. dataloader/generate_data.py - 数据生成
  def generate_anc_training_data(
      path_dir: str,
      train_files: List[str],
      sec_path_file: str,
      N_epcho: int,
      Len_N: int,
      fs: int = 16000,
      broadband_len: int = None
  ) -> Tuple[np.ndarray, np.ndarray]:
      """
      用于通过添加路径响应和噪声，生成两通道Fx和声效d
  
      Returns:
          Fx_data: [2 * Len_N, N_epcho]
          Di_data: [Len_N, N_epcho]
      """

2. models/control_filter.py - 控制滤波器基类
    class ControlFilter:
        def __init__(self, filter_len: int, num_refs: int):
            """Phi初始化为0"""
    
        def predict(self, Fx: np.ndarray) -> float:
            """
            输入Fx: [2*Len_N], 计算 y(n)
            """
    
        def update(self, gradient: np.ndarray):
            """W 更新"""

3. models/maml_filter.py - MAML模型
  class MAMLFilter(ControlFilter):
    def maml_initial(self,
                     Fx: np.ndarray,
                     Di: np.ndarray,
                     mu: float,
                     lamda: float,
                     epslon: float) -> float:
        """
        进行初始化更新，返回第1个时刻误差
        """
   
5. algorithms/fxlms.py - FxLMS 实现
  def multi_ref_multi_chan_fxlms(
    Ref: np.ndarray,
    E: np.ndarray,
    filter_len: int,
    sec_path: np.ndarray,
    stepsize: float
  ) -> Tuple[np.ndarray, np.ndarray]:
      """
      FxLMS baseline 算法实现
  
      Returns:
          Ww: 控制器系数 [Lw, WSum]
          ew: 误差 [Len, ErrNum]
      """

5. main.py - 运行主程序
  # 包括：数据读取 -> 初始化 -> 使用MAMLFilter -> 存储MSE -> 作图

6. evaluation/mse_plot.py - MSE计算与可视化
  def compute_mse(x: np.ndarray, win_len: int = 4096, base_db: float = 94.0) -> np.ndarray:
    """ 计算滑动均方误差，输出单位 dB """

  def plot_mse(mse_curve: np.ndarray, title: str, save_path: str = None):
      """ 绘制 MSE 曲线图 """

7. utils/mat_io.py - MAT文件处理
  def load_and_resample_mat(filepath: str, key: str, fs_target: int) -> np.ndarray:
    """ 加载 .mat 文件中的路径响应并重采样 """

8. utils/signal_utils.py - 信号通用函数
  def normalize(x: np.ndarray) -> np.ndarray:
      """ 归一化信号 """
  
  def bandpass_filter(x: np.ndarray, fs: int, f_low: float, f_high: float) -> np.ndarray:
      """ 简单带通滤波器 """

9. AGENT.md - 环境配置
  conda create -n anc_maml python=3.9
  conda activate anc_maml
  conda install numpy scipy matplotlib scikit-learn tqdm tensorboard
  conda install pytorch torchvision cpuonly -c pytorch








   















