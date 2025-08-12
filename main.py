# 简化的MAML-ANC实现 - 严格按照MATLAB逻辑

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal

# ------------------- MAML-ANC核心算法 -------------------
class MAML_ANC_Single:
    """基于MATLAB原版的单通道MAML-ANC实现"""
    
    def __init__(self, len_c):
        self.len_c = len_c
        self.Phi = np.zeros(len_c)

    def maml_update(self, Fx, Di, mu, lamda, epsilon):
        # 精确对应MATLAB的flipud操作
        Fx = np.flipud(Fx)
        Di = np.flipud(Di)
        
        Grad = 0
        Er = 0
        Li = len(self.Phi)
        
        # <-4-> 基于初始控制滤波器计算误差
        e = Di[0] - np.dot(self.Phi, Fx)
        
        # <-5-> 获得一步更新后的控制滤波器
        Wo = self.Phi + mu * e * Fx
        
        # <-6-> 遍历所有时间步
        for jj in range(Li):
            if jj == 0:
                Fd = Fx.copy()
            else:
                Fd = np.concatenate([Fx[jj:], np.zeros(jj)])
            
            e = Di[jj] - np.dot(Wo, Fd)
            Grad += epsilon * (mu / Li) * e * Fd * (lamda ** jj)
            
            if jj == 0:
                Er = e
        
        # <-7-> 更新初始值
        self.Phi = self.Phi + Grad
        return Er

def resample_to_target(x: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """重采样函数"""
    if fs_orig == fs_target:
        return x
    return signal.resample_poly(x, fs_target, fs_orig, axis=0)

def static_filter_test(W, Dis, Rf):
    """
    静态滤波测试 - 对应MATLAB中步长为0的FxLMS测试
    这实际上就是用固定权重进行滤波，不进行自适应更新
    
    Args:
        W: 控制滤波器权重 [filter_len]
        Dis: 干扰信号 [N]  
        Rf: 滤波参考信号 [N]
    
    Returns:
        Er: 误差信号 [N]
    """
    N = len(Dis)
    filter_len = len(W)
    Er = np.zeros(N)
    
    # 参考信号缓存
    ref_buffer = np.zeros(filter_len)
    
    for n in range(N):
        # 更新参考信号缓存
        ref_buffer = np.roll(ref_buffer, 1)
        ref_buffer[0] = Rf[n]
        
        # 计算控制信号
        control_signal = np.dot(W, ref_buffer)
        
        # 计算误差（在MATLAB中，次级路径已经包含在Rf中）
        Er[n] = Dis[n] + control_signal
    
    return Er

# ================= 配置参数 =================
FILTER_LEN = 512
NUM_EPOCHS = 4096 * 5
MU = 0.003
LAMBDA = 0.99
EPSILON = 0.5

# 路径配置 - 严格按照MATLAB
PATH_DIR = 'E:/NTU/Test/HeadphonePathMeasurement/Recording/Primary/PriPathModels_E'
TRAIN_FILES = ['PriPath_Pri_d50_180.mat']  # 只用一个文件，与MATLAB一致
SEC_PATH_FILE = 'E:/NTU/Test/HeadphonePathMeasurement/Recording/Secondary/SecPathModels/SecPath_firmed.mat'
TEST_NOISE_PATH = r'E:\NTU\AIANC\Meta-main\bandpassed_200_700.wav'

print("=" * 60)
print("Simplified MAML-ANC - Following MATLAB Logic Exactly")
print("=" * 60)

# ================= 数据生成（对应MATLAB训练数据生成） =================
print("🔄 Generating training data exactly like MATLAB...")

# 加载路径文件
all_ref_path = []
all_err_path = []

for fname in TRAIN_FILES:
    file_path = os.path.join(PATH_DIR, fname)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training file not found: {file_path}")
    
    dat = sio.loadmat(file_path)
    G = dat["G_matrix"]
    
    # 对应MATLAB: h_ref = resample(G(:,1), fs, 48000);
    h_ref = resample_to_target(G[:, 0], 48000, 16000)
    # 对应MATLAB: h_err = resample(G(:,3), fs, 48000);  
    h_err = resample_to_target(G[:, 2], 48000, 16000)
    
    all_ref_path.append(h_ref)
    all_err_path.append(h_err)

# 加载次级路径
if not os.path.exists(SEC_PATH_FILE):
    raise FileNotFoundError(f"Secondary path file not found: {SEC_PATH_FILE}")

sec_dat = sio.loadmat(SEC_PATH_FILE)
sec_keys = [k for k in sec_dat.keys() if not k.startswith('__')]
sec_key = sec_keys[0]  # 对应MATLAB的fieldnames(sec_dat)的第一个
S_data = sec_dat[sec_key]
S_48k = S_data[:, 0]  # 对应MATLAB的(:,1)
S = resample_to_target(S_48k, 48000, 16000)

print(f"Loaded paths: {len(all_ref_path)} training paths, secondary path length: {len(S)}")

# 生成训练数据 - 对应MATLAB的训练样本生成
print("🔄 Generating training samples...")

# 对应MATLAB: white = randn(N, 1);
N = 16000 * 3  # 3秒
white = np.random.randn(N)

# 对应MATLAB: broadband_filter = fir1(512, [0.015 0.25]);
# 但根据你的要求，改为100Hz-1500Hz
nyquist = 16000 / 2
low_freq = 100 / nyquist
high_freq = 1500 / nyquist
broadband_filter = signal.firwin(512, [low_freq, high_freq], pass_zero=False, window='hamming')

Fx_data = np.zeros((FILTER_LEN, NUM_EPOCHS))
Di_data = np.zeros((FILTER_LEN, NUM_EPOCHS))

print("Generating training samples...")
for jj in range(NUM_EPOCHS):
    if (jj + 1) % (NUM_EPOCHS // 10) == 0:
        print(f"  Progress: {jj + 1}/{NUM_EPOCHS}")
    
    # 对应MATLAB: idx = randi(length(train_files));
    idx = np.random.randint(len(TRAIN_FILES))
    P_ref = all_ref_path[idx]
    P_err = all_err_path[idx]
    
    # 对应MATLAB: broadband = filter(broadband_filter, 1, white);
    broadband = signal.lfilter(broadband_filter, [1.0], white)
    
    # 对应MATLAB信号处理链
    x_ref = signal.lfilter(P_ref, [1.0], broadband)    # 电→ref
    xprime = signal.lfilter(S, [1.0], x_ref)           # ref→err（次级路径）
    d = signal.lfilter(P_err, [1.0], broadband)        # 电→err
    
    # 对应MATLAB: idx_cut = randi([Len_N, length(d)]);
    idx_cut = np.random.randint(FILTER_LEN, len(d))
    
    # 对应MATLAB的切片
    Di_data[:, jj] = d[idx_cut - FILTER_LEN:idx_cut]
    Fx_data[:, jj] = xprime[idx_cut - FILTER_LEN:idx_cut]

print(f"Training data generated: Fx_data {Fx_data.shape}, Di_data {Di_data.shape}")

# ================= MAML训练 =================
print("\n🔄 Training MAML...")

maml_anc = MAML_ANC_Single(FILTER_LEN)
Er_train = []

for jj in range(NUM_EPOCHS):
    Er = maml_anc.maml_update(Fx_data[:, jj], Di_data[:, jj], MU, LAMBDA, EPSILON)
    Er_train.append(Er)
    
    if (jj + 1) % (NUM_EPOCHS // 20) == 0 or jj == 0:
        print(f"Epoch {jj + 1}: Er = {Er:.6f}, Weight max = {np.max(np.abs(maml_anc.Phi)):.6f}")

print(f"MAML training completed!")
print(f"Final weights - Mean: {np.mean(maml_anc.Phi):.8f}, Std: {np.std(maml_anc.Phi):.8f}")
print(f"Weight range: [{np.min(maml_anc.Phi):.6f}, {np.max(maml_anc.Phi):.6f}]")

# ================= 测试数据准备（严格按照MATLAB） =================
print("\n🔄 Preparing test data exactly like MATLAB...")

# 对应MATLAB: [x_input_test, fs_file] = audioread('bandpassed_200_700.wav');
if not os.path.exists(TEST_NOISE_PATH):
    raise FileNotFoundError(f"Test noise file not found: {TEST_NOISE_PATH}")

from scipy.io import wavfile
fs_orig, test_audio_raw = wavfile.read(TEST_NOISE_PATH)

# 转换并归一化
if test_audio_raw.dtype == np.int16:
    Pri_1 = test_audio_raw.astype(np.float32) / 32768.0
else:
    Pri_1 = test_audio_raw.astype(np.float32)

if Pri_1.ndim > 1:
    Pri_1 = Pri_1[:, 0]

# 对应MATLAB: Pri_1 = resample(x_input_test, fs, fs_file);
if fs_orig != 16000:
    Pri_1 = resample_to_target(Pri_1, fs_orig, 16000)

print(f"Test noise loaded: {len(Pri_1)} samples")

# 对应MATLAB: 使用d50_180作为测试路径
test_file = 'PriPath_Pri_d50_180.mat'
test_path = os.path.join(PATH_DIR, test_file)
if not os.path.exists(test_path):
    raise FileNotFoundError(f"Test path file not found: {test_path}")

test_data = sio.loadmat(test_path)
G_test = test_data["G_matrix"]

# 对应MATLAB测试路径处理
P_ref_test = resample_to_target(G_test[:, 0], 48000, 16000)
P_err_test = resample_to_target(G_test[:, 2], 48000, 16000)

# 对应MATLAB测试信号生成
x_ref_test = signal.lfilter(P_ref_test, [1.0], Pri_1)    # 电→ref
Rf_1 = signal.lfilter(S, [1.0], x_ref_test)             # ref→err
Dis_1 = signal.lfilter(P_err_test, [1.0], Pri_1)        # 电→err

print(f"Test signals generated: Rf_1 {len(Rf_1)}, Dis_1 {len(Dis_1)}")

# ================= FxLMS测试（静态滤波，对应MATLAB的muw=0.000） =================
print("\n🔄 Running static filter tests (like MATLAB with muw=0.000)...")

# 对应MATLAB: Wc_zero = zeros(Len_N,1);
Wc_zero = np.zeros(FILTER_LEN)

# 对应MATLAB: [Er_zero,~] = FxLMS(Len_N, Wc_zero, Dis_1, Rf_1, muw);
# 这里muw=0.000，实际上是静态滤波
Er_zero = static_filter_test(Wc_zero, Dis_1, Rf_1)

# 对应MATLAB: [Er_maml,~] = FxLMS(Len_N, Wc, Dis_1, Rf_1, 1*muw);
# Wc是MAML训练得到的权重
Er_maml = static_filter_test(maml_anc.Phi, Dis_1, Rf_1)

print(f"Static filter tests completed!")

# ================= 结果分析 =================
print("\n" + "=" * 60)
print("Results Analysis")
print("=" * 60)

# 计算MSE (对应MATLAB的平均MSE计算)
def compute_mse_db(signal):
    return 10 * np.log10(np.mean(signal**2) + 1e-10)

mse_original = compute_mse_db(Dis_1)
mse_zero = compute_mse_db(Er_zero)
mse_maml = compute_mse_db(Er_maml)

print(f"MSE Results:")
print(f"  Original (ANC off):     {mse_original:.2f} dB")
print(f"  Zero-init (static):     {mse_zero:.2f} dB")
print(f"  MAML-init (static):     {mse_maml:.2f} dB")
print(f"  MAML improvement:       {mse_zero - mse_maml:.2f} dB")

if mse_maml < mse_zero:
    print("🎉 MAML shows improvement over zero initialization!")
else:
    print("⚠️ MAML does not show improvement")

# ================= 可视化 =================
print("\n🔄 Creating visualizations...")

plt.figure(figsize=(15, 10))

# 训练误差
plt.subplot(2, 3, 1)
plt.plot(Er_train)
plt.title('MAML Training Error')
plt.xlabel('Epoch')
plt.ylabel('Training Error')
plt.grid(True)

# MAML权重
plt.subplot(2, 3, 2)
plt.plot(maml_anc.Phi)
plt.title('MAML Weights')
plt.xlabel('Coefficient Index')
plt.ylabel('Weight Value')
plt.grid(True)

# 时域比较
plt.subplot(2, 3, 3)
t = np.arange(len(Dis_1)) / 16000
plt.plot(t, Dis_1, label='Original', alpha=0.7)
plt.plot(t, Er_zero, label='Zero-init', alpha=0.8)
plt.plot(t, Er_maml, label='MAML-init', alpha=0.8)
plt.title('Time Domain Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# MSE对比
plt.subplot(2, 3, 4)
mse_values = [mse_original, mse_zero, mse_maml]
labels = ['Original', 'Zero-init', 'MAML-init']
colors = ['red', 'blue', 'green']
plt.bar(labels, mse_values, color=colors, alpha=0.7)
plt.title('MSE Comparison')
plt.ylabel('MSE (dB)')
plt.grid(True)

# 权重能量
plt.subplot(2, 3, 5)
plt.plot(np.cumsum(maml_anc.Phi**2))
plt.title('Cumulative Weight Energy')
plt.xlabel('Coefficient Index')
plt.ylabel('Cumulative Energy')
plt.grid(True)

# 训练收敛
plt.subplot(2, 3, 6)
if len(Er_train) > 100:
    window_size = len(Er_train) // 50
    moving_avg = np.convolve(Er_train, np.ones(window_size)/window_size, mode='valid')
    plt.plot(moving_avg)
    plt.title('Training Convergence (Moving Average)')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)

plt.tight_layout()
plt.savefig('maml_results_simplified.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Simplified MAML-ANC Completed Successfully!")
print("=" * 60)
