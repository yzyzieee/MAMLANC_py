# 🔶 路径加载、白噪合成、训练样本生成 - 严格模式

import os
from typing import List, Tuple, Optional

import numpy as np
import scipy.io as sio
import scipy.signal as signal

def resample_to_target(x: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Resample signal from fs_orig to fs_target using polyphase filter."""
    if fs_orig == fs_target:
        return x
    return signal.resample_poly(x, fs_target, fs_orig, axis=0)

def generate_anc_training_data(path_dir: str,
                                train_files: List[str],
                                sec_path_file: str,
                                N_epcho: int,
                                Len_N: int,
                                fs: int = 16000,
                                broadband_len: Optional[int] = None,
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ANC training data following the MATLAB reference logic EXACTLY.

    Each training sample is constructed by filtering a broadband excitation
    through a randomly selected primary path and the secondary path. The
    routine assumes a **single** reference microphone and returns stacked
    segments of length ``Len_N``.

    Returns
    -------
    Fx_data : np.ndarray
        ``[Len_N, N_epcho]`` reference signal at the error microphone.
    Di_data : np.ndarray
        ``[Len_N, N_epcho]`` disturbance signal at the error microphone.
    """
    if broadband_len is None:
        broadband_len = int(fs * 3)  # default 3 seconds

    # 严格检查路径目录
    if not os.path.exists(path_dir):
        raise FileNotFoundError(f"Path directory does not exist: {path_dir}")
    
    if not os.path.exists(sec_path_file):
        raise FileNotFoundError(f"Secondary path file does not exist: {sec_path_file}")

    print(f"Loading training data from {len(train_files)} path files...")
    
    # Step 1: Load primary paths (electric→ref and electric→err) - 严格对应MATLAB索引
    all_ref_path, all_err_path = [], []
    for i, fname in enumerate(train_files):
        file_path = os.path.join(path_dir, fname)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Training file does not exist: {file_path}")
            
        print(f"Loading file {i+1}/{len(train_files)}: {fname}")
        
        dat = sio.loadmat(file_path)
        if "G_matrix" not in dat:
            raise KeyError(f"G_matrix not found in {fname}. Available keys: {list(dat.keys())}")
        
        G = dat["G_matrix"]
        
        # 严格检查矩阵维度
        if G.ndim != 2 or G.shape[1] < 4:  # 需要至少4列：ch1→ch2,ch3,ch4,ch5
            raise ValueError(f"G_matrix in {fname} has invalid shape {G.shape}. Expected at least 4 columns.")
        
        # 完全对应MATLAB索引：注意MATLAB用G(:,1)和G(:,3)，Python中是G[:,0]和G[:,2]
        h_ref = resample_to_target(G[:, 0], 48000, fs)  # electric→ref (ch1→ch2，MATLAB的G(:,1))
        h_err = resample_to_target(G[:, 2], 48000, fs)  # electric→err (ch1→ch4，MATLAB的G(:,3))

        all_ref_path.append(h_ref)
        all_err_path.append(h_err)

    print(f"Successfully loaded {len(all_ref_path)} reference paths")

    # Step 2: Load and resample secondary path (ref→err) - 严格模式
    print(f"Loading secondary path from {sec_path_file}")
    
    sec_dat = sio.loadmat(sec_path_file)
    # 找到非系统键（不以__开头）
    sec_keys = [k for k in sec_dat.keys() if not k.startswith('__')]
    if not sec_keys:
        raise KeyError(f"No data keys found in secondary path file {sec_path_file}. Available keys: {list(sec_dat.keys())}")
    
    sec_key = sec_keys[-1]  # 使用最后一个键，对应MATLAB的sec_key{1}
    print(f"Using secondary path key: '{sec_key}'")
    
    S_data = sec_dat[sec_key]
    # 确保取第一列，对应MATLAB的(:,1)
    if S_data.ndim == 1:
        S_48k = S_data
    elif S_data.ndim == 2:
        S_48k = S_data[:, 0]  # 对应MATLAB的(:,1)
    else:
        raise ValueError(f"Secondary path data has invalid dimensions: {S_data.shape}")
        
    S = resample_to_target(S_48k, 48000, fs)
    print(f"Secondary path length: {len(S)}")

    # Step 3: Generate broadband excitation - 使用更窄的频带
    print(f"Generating broadband excitation (length: {broadband_len})")
    
    # 对应MATLAB的 white = randn(N, 1);
    white = np.random.randn(broadband_len)
    
    # 修改为更窄的频带：100Hz-1500Hz
    # 计算归一化频率：100Hz和1500Hz对应的归一化频率
    nyquist = fs / 2  # 8000Hz for fs=16000
    low_freq_norm = 100 / nyquist    # 100Hz / 8000Hz = 0.0125
    high_freq_norm = 1500 / nyquist  # 1500Hz / 8000Hz = 0.1875
    
    print(f"Training noise frequency band: {100}Hz - {1500}Hz")
    print(f"Normalized frequencies: [{low_freq_norm:.4f}, {high_freq_norm:.4f}]")
    
    try:
        broadband_filter = signal.firwin(512, [low_freq_norm, high_freq_norm], 
                                       pass_zero=False, window='hamming')
    except:
        # 如果失败，尝试更简单的设计
        broadband_filter = signal.firwin(512, [low_freq_norm, high_freq_norm], 
                                       pass_zero=False)
    
    # 对应MATLAB的 broadband = filter(broadband_filter, 1, white);
    broadband = signal.lfilter(broadband_filter, [1.0], white)

    print(f"Broadband signal RMS: {np.sqrt(np.mean(broadband**2)):.6f}")

    # Step 4: Assemble training samples - 精确对应MATLAB逻辑
    print(f"Generating {N_epcho} training samples...")
    Fx_data = np.zeros((Len_N, N_epcho))  # 对应MATLAB的 Fx_data = zeros(Len_N, N_epcho);
    Di_data = np.zeros((Len_N, N_epcho))  # 对应MATLAB的 Di_data = zeros(Len_N, N_epcho);

    for jj in range(N_epcho):
        if (jj + 1) % max(1, N_epcho // 10) == 0:
            print(f"Processing sample {jj + 1}/{N_epcho}")
            
        # 对应MATLAB的 idx = randi(length(train_files));
        idx = np.random.randint(len(train_files))
        P_ref = all_ref_path[idx]
        P_err = all_err_path[idx]

        # 滤波处理 - 精确对应MATLAB
        # x_ref = filter(P_ref, 1, broadband);
        x_ref = signal.lfilter(P_ref, [1.0], broadband)
        # xprime = filter(S, 1, x_ref);
        xprime = signal.lfilter(S, [1.0], x_ref)  # 通过次级路径
        # d = filter(P_err, 1, broadband);
        d = signal.lfilter(P_err, [1.0], broadband)

        # 严格检查信号长度
        min_len = min(len(d), len(xprime))
        if min_len < Len_N:
            raise ValueError(f"Filtered signal too short ({min_len} < {Len_N}). "
                           f"Increase broadband_len or check filter lengths.")

        # 对应MATLAB的 idx_cut = randi([Len_N, length(d)]);
        idx_cut = np.random.randint(Len_N, min_len)
        
        # 对应MATLAB的切片操作：d(idx_cut - Len_N + 1:idx_cut)
        # MATLAB是1-based，Python是0-based，所以：
        Fx_data[:, jj] = xprime[idx_cut - Len_N: idx_cut]
        Di_data[:, jj] = d[idx_cut - Len_N: idx_cut]

    print(f"Training data generation completed!")
    print(f"Fx_data shape: {Fx_data.shape}, RMS: {np.sqrt(np.mean(Fx_data**2)):.6f}")
    print(f"Di_data shape: {Di_data.shape}, RMS: {np.sqrt(np.mean(Di_data**2)):.6f}")
    
    # 严格验证数据质量
    if np.any(np.isnan(Fx_data)) or np.any(np.isinf(Fx_data)):
        raise ValueError("Fx_data contains NaN or Inf values")
        
    if np.any(np.isnan(Di_data)) or np.any(np.isinf(Di_data)):
        raise ValueError("Di_data contains NaN or Inf values")
    
    fx_rms = np.sqrt(np.mean(Fx_data**2))
    di_rms = np.sqrt(np.mean(Di_data**2))
    
    if fx_rms < 1e-10:
        raise ValueError(f"Fx_data RMS too small ({fx_rms}), check input signals")
        
    if di_rms < 1e-10:
        raise ValueError(f"Di_data RMS too small ({di_rms}), check input signals")
    
    return Fx_data, Di_data


def load_test_noise(file_path: str, target_fs: int = 16000) -> np.ndarray:
    """Load test noise from WAV file.
    
    Args:
        file_path: Path to the WAV file
        target_fs: Target sampling rate
        
    Returns:
        np.ndarray: Loaded and resampled audio signal
    """
    import soundfile as sf
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test noise file does not exist: {file_path}")
    
    print(f"Loading test noise from: {file_path}")
    
    try:
        # 尝试用soundfile加载
        audio, fs_orig = sf.read(file_path)
        print(f"Original sampling rate: {fs_orig}Hz, Target: {target_fs}Hz")
        
    except ImportError:
        print("soundfile not available, trying scipy.io.wavfile...")
        try:
            from scipy.io import wavfile
            fs_orig, audio = wavfile.read(file_path)
            # 转换为float并归一化
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
        except Exception as e:
            raise RuntimeError(f"Failed to load WAV file with scipy: {e}")
    
    except Exception as e:
        print(f"Error loading with soundfile: {e}")
        # 尝试用scipy作为备选
        try:
            from scipy.io import wavfile
            fs_orig, audio = wavfile.read(file_path)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0
        except Exception as e2:
            raise RuntimeError(f"Failed to load WAV file: {e2}")
    
    # 如果是立体声，取第一个通道
    if audio.ndim > 1:
        audio = audio[:, 0]
        print(f"Multi-channel audio detected, using first channel")
    
    print(f"Loaded audio: {len(audio)} samples, RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    
    # 重采样到目标采样率
    if fs_orig != target_fs:
        audio_resampled = resample_to_target(audio, fs_orig, target_fs)
        print(f"Resampled to {target_fs}Hz: {len(audio_resampled)} samples")
        return audio_resampled
    
    return audio


def generate_task_batch(length: int,
                        num_refs: int,
                        with_secondary: bool = False,
                        num_errs: int = 1,
                        sec_len: int = 128,
                        use_test_noise: bool = False,
                        test_noise_path: Optional[str] = None):
    """Generate a random ANC task for quick experiments.

    This utility is primarily for algorithm validation when no
    measured path data are provided. It synthesizes random reference
    and disturbance signals. When ``with_secondary`` is True, a random
    secondary path matrix is also returned.

    Args:
        length (int): Number of time-domain samples.
        num_refs (int): Number of reference channels.
        with_secondary (bool): Whether to generate a secondary path.
        num_errs (int): Number of error microphones.
        sec_len (int): Length of the secondary path impulse response.
        use_test_noise (bool): Whether to use real test noise file.
        test_noise_path (str): Path to test noise WAV file.

    Returns:
        If ``with_secondary`` is True:
            Tuple[np.ndarray, np.ndarray, np.ndarray]
            → (Ref, E, sec_path)
        Else:
            Tuple[np.ndarray, np.ndarray]
            → (Ref, Di)
    """
    
    # 参数严格检查
    if length <= 0:
        raise ValueError(f"Length must be positive, got {length}")
    if num_refs <= 0:
        raise ValueError(f"num_refs must be positive, got {num_refs}")
    if num_errs <= 0:
        raise ValueError(f"num_errs must be positive, got {num_errs}")
    if sec_len <= 0:
        raise ValueError(f"sec_len must be positive, got {sec_len}")
    
    if use_test_noise and test_noise_path:
        print("🔊 Using real test noise file...")
        try:
            # 加载真实测试噪声
            test_audio = load_test_noise(test_noise_path, target_fs=16000)
            
            # 如果音频太短，重复它
            if len(test_audio) < length:
                repeats = (length // len(test_audio)) + 1
                test_audio = np.tile(test_audio, repeats)
            
            # 截取所需长度
            if len(test_audio) >= length:
                # 随机选择起始位置
                start_idx = np.random.randint(0, len(test_audio) - length + 1)
                test_segment = test_audio[start_idx:start_idx + length]
            else:
                test_segment = test_audio
            
            # ✅ 正确做法：两个信号都来自同一个源，但后续会通过不同路径处理
            Ref = np.zeros((length, num_refs))
            Di = np.zeros((length, num_errs))
            
            # 使用相同的测试噪声作为基础（模拟同一个噪声源）
            for i in range(num_refs):
                Ref[:, i] = test_segment[:length]  # 所有通道使用相同的源信号
            
            for i in range(num_errs):
                Di[:, i] = test_segment[:length]   # 所有通道使用相同的源信号
            
            # ❌ 删除了人工相关性处理，因为：
            # 1. 物理上不合理 - 真实系统中相关性来自路径滤波，不是人工混合
            # 2. 与MATLAB不一致 - MATLAB直接使用原始测试信号
            # 3. 会在后续通过真实路径产生正确的相关性
            
            print(f"✅ Using same source signal for all channels (physically correct)")
            
        except Exception as e:
            print(f"Failed to load test noise: {e}")
            print("Falling back to synthetic noise...")
            use_test_noise = False
    
    if not use_test_noise:
        # 生成合成信号用于算法验证
        print("🔧 Generating synthetic signals...")
        
        # 生成基础白噪声
        base_noise = np.random.randn(length)
        
        # 对于合成数据，也使用相同的源信号
        Ref = np.zeros((length, num_refs))
        Di = np.zeros((length, num_errs))
        
        # 所有通道使用相同的基础信号（模拟同一个噪声源）
        for i in range(num_refs):
            Ref[:, i] = base_noise
        
        for i in range(num_errs):  
            Di[:, i] = base_noise
        
        # 为了避免完全相同，添加极小的独立噪声（模拟测量噪声）
        Ref += np.random.randn(length, num_refs) * 0.01  # 1% measurement noise
        Di += np.random.randn(length, num_errs) * 0.01   # 1% measurement noise
        
        print("✅ Generated synthetic signals with same source + small measurement noise")

    # 标准化信号幅度
    Ref = Ref / (np.std(Ref) + 1e-10) * 0.1
    Di = Di / (np.std(Di) + 1e-10) * 0.1

    if with_secondary:
        # 生成更真实的次级路径：FIR滤波器系数
        sec_path = np.random.randn(sec_len, num_refs * num_errs) * 0.05
        # 确保主要能量在前几个系数中
        sec_path[:min(10, sec_len)] += np.random.randn(min(10, sec_len), num_refs * num_errs) * 0.2
        
        return Ref, Di, sec_path

    return Ref, Di
