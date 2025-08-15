# 多参考MAML-ANC实现 - 基于MATLAB版本的完整实现

import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy import signal
from scipy.io import wavfile

# ------------------- 多参考MAML-ANC核心算法 -------------------
class MAML_ANC_Multi:
    """多参考多控制源MAML-ANC实现 (对应MATLAB的MAML_Nstep_forget_MultiRef)"""
    
    def __init__(self, L, num_refs):
        """
        Args:
            L: 滤波器长度 (每个通道)
            num_refs: 参考源数量
        """
        self.L = L
        self.num_refs = num_refs
        # 权重为堆叠向量 [L*num_refs × 1]，与MATLAB保持一致
        self.Phi = np.zeros(L * num_refs)
        
    def maml_update(self, Fx_input, Di_input, mu, lamda, epsilon):
        """
        Fx_input: [L, num_refs]  每列一条参考的 filtered-x
        Di_input: [L]
        """
        L = self.L
        K = self.num_refs

        # ✅ 关键：保持 2D，再按“时间轴”翻转
        F = np.flipud(Fx_input)          # [L, K]，只在时间维度上下翻
        D = np.flipud(Di_input)          # [L]

        F_vec = F.flatten('F')           # [L*K]，按列堆叠
        e0 = D[0] - np.dot(self.Phi, F_vec)
        Wo = self.Phi + mu * e0 * F_vec  # 内层一步

        Grad = np.zeros_like(self.Phi)
        Er = 0.0

        for jj in range(L):
            if jj == 0:
                Fd = F
            else:
                # 每列各自向下移位（时间对齐），列不互换
                Fd = np.vstack([F[jj:, :], np.zeros((jj, K))])  # [L, K]

            Fd_vec = Fd.flatten('F')      # [L*K]
            e = D[jj] - np.dot(Wo, Fd_vec)
            Grad += epsilon * (mu / L) * e * Fd_vec * (lamda ** jj)

            if jj == 0:
                Er = e

        self.Phi += Grad
        return Er


def resample_to_target(x: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """重采样函数"""
    if fs_orig == fs_target:
        return x
    return signal.resample_poly(x, fs_target, fs_orig, axis=0)


def FxLMS_MultiRef(Len_N, Wc, Dis, Rf, muw):
    """
    多参考FxLMS算法实现（对应MATLAB的FxLMS_MultiRef）
    
    Args:
        Len_N: 滤波器长度
        Wc: 控制滤波器权重 [Len_N*num_refs]
        Dis: 干扰信号 [N]
        Rf: 多通道filtered参考信号 [N, num_refs]
        muw: 步长
    
    Returns:
        Er: 误差信号 [N]
        W_final: 最终权重
    """
    N = len(Dis)
    num_refs = Rf.shape[1]
    
    # 初始化
    W = Wc.copy()
    Er = np.zeros(N)
    
    # 参考信号缓存 [Len_N, num_refs]
    ref_buffer = np.zeros((Len_N, num_refs))
    
    for n in range(N):
        # 更新缓存
        ref_buffer = np.roll(ref_buffer, 1, axis=0)
        ref_buffer[0, :] = Rf[n, :]
        
        # 堆叠成列向量（与MATLAB一致）
        ref_stacked = ref_buffer.flatten('F')  # 按列优先
        
        # 计算控制信号
        control_signal = np.dot(W, ref_stacked)
        
        # 计算误差
        Er[n] = Dis[n] - control_signal
        
        # 更新权重
        W = W + muw * Er[n] * ref_stacked
    
    return Er, W


# ================= 配置参数 =================
fs = 16000
T = 3
N = fs * T
Len_N = 512
NUM_REFS = 2  # 双参考

# MAML参数
NUM_EPOCHS = 4096 * 5  # 与MATLAB一致
MU = 0.003
LAMBDA = 0.99
EPSILON = 0.5

# 路径配置
PATH_DIR = 'E:/NTU/Test/HeadphonePathMeasurement/Recording/Primary/PriPathModels_E'

# 训练文件列表（与MATLAB一致）
TRAIN_FILES = [
    'PriPath_Pri_d50_180.mat',
]

SEC_PATH_FILE = 'E:/NTU/Test/HeadphonePathMeasurement/Recording/Secondary/SecPathModels/SecPath_firmed.mat'
TEST_NOISE_PATH = r'E:\NTU\AIANC\Meta-main\bandpassed_200_700.wav'

print("=" * 60)
print(f"Multi-Reference MAML-ANC (2 References)")
print(f"Configuration: {NUM_REFS} references, {len(TRAIN_FILES)} training paths")
print("=" * 60)

# ================= 加载训练路径 =================
print("🔄 Loading training paths...")

all_ref_path = []  # 存储所有的参考路径对 [refL, refR]
all_err_path = []  # 存储所有的误差路径

for fname in TRAIN_FILES:
    file_path = os.path.join(PATH_DIR, fname)
    if not os.path.exists(file_path):
        print(f"⚠️ Warning: Training file not found: {file_path}")
        continue
    
    dat = sio.loadmat(file_path)
    G = dat["G_matrix"]  # 4列：LL, LR, RL, RR
    
    # 提取路径（与MATLAB对应）
    # G(:,1) = ch1→ch2 (电→refL)
    # G(:,2) = ch1→ch3 (电→refR) 
    # G(:,3) = ch1→ch4 (电→errL)
    h_refL = resample_to_target(G[:, 0], 48000, fs)  # 第1列
    h_refR = resample_to_target(G[:, 1], 48000, fs)  # 第2列
    h_err = resample_to_target(G[:, 2], 48000, fs)   # 第3列
    
    # 存储为[refL, refR]对
    all_ref_path.append([h_refL, h_refR])
    all_err_path.append(h_err)

print(f"✅ Loaded {len(all_ref_path)} training path sets")

# 加载次级路径
if not os.path.exists(SEC_PATH_FILE):
    raise FileNotFoundError(f"Secondary path file not found: {SEC_PATH_FILE}")

sec_dat = sio.loadmat(SEC_PATH_FILE)
sec_keys = [k for k in sec_dat.keys() if not k.startswith('__')]
sec_key = sec_keys[0]
S_data = sec_dat[sec_key]
S_48k = S_data[:, 0]  # 使用第1列
S = resample_to_target(S_48k, 48000, fs)

print(f"✅ Secondary path loaded, length: {len(S)}")

# ================= 生成训练数据 =================
print(f"\n🔄 Generating training samples for {NUM_REFS} references...")

# 生成白噪声
white = np.random.randn(N)

# 宽带滤波器（对应MATLAB的broadband_filter）
nyquist = fs / 2
broadband_filter = signal.firwin(512, [0.015, 0.25], pass_zero=False, window='hamming')
# 0.015 * 8000 = 120Hz, 0.25 * 8000 = 2000Hz

# 准备训练数据存储
Fx_data = np.zeros((Len_N * NUM_REFS, NUM_EPOCHS))  # [2*Len_N, N_epochs]
Di_data = np.zeros((Len_N, NUM_EPOCHS))

print("Generating training samples...")
for jj in range(NUM_EPOCHS):
    if (jj + 1) % (NUM_EPOCHS // 10) == 0:
        print(f"  Progress: {jj + 1}/{NUM_EPOCHS}")
    
    # 随机选择路径（对应MATLAB: idx = randi(length(train_files))）
    idx = np.random.randint(len(all_ref_path))
    P_ref = all_ref_path[idx]  # [refL, refR]
    P_err = all_err_path[idx]
    
    # 分离refL和refR
    P_ref_L = P_ref[0]
    P_ref_R = P_ref[1]
    
    # 生成宽带噪声
    broadband = signal.lfilter(broadband_filter, [1.0], white)
    
    # 控制输入路径（参考信号）
    x_ref_L = signal.lfilter(P_ref_L, [1.0], broadband)  # 电→refL
    x_ref_R = signal.lfilter(P_ref_R, [1.0], broadband)  # 电→refR
    xprime_L = signal.lfilter(S, [1.0], x_ref_L)         # refL→err
    xprime_R = signal.lfilter(S, [1.0], x_ref_R)         # refR→err
    
    # 误差信号路径
    d = signal.lfilter(P_err, [1.0], broadband)          # 电→err
    
    # 随机裁剪（对应MATLAB: idx_cut = randi([Len_N, length(d)])）
    idx_cut = np.random.randint(Len_N, len(d))
    
    # 存储数据
    Di_data[:, jj] = d[idx_cut - Len_N:idx_cut]
    
    # 拼接2通道参考信号（垂直拼接）
    x1 = xprime_L[idx_cut - Len_N:idx_cut]
    x2 = xprime_R[idx_cut - Len_N:idx_cut]
    Fx_data[:, jj] = np.concatenate([x1, x2])  # [2*Len_N]

print(f"✅ Training data shape: Fx_data {Fx_data.shape}, Di_data {Di_data.shape}")

# ================= MAML训练 =================
print(f"\n🔄 Training Multi-Reference MAML with {NUM_REFS} references...")

maml_anc = MAML_ANC_Multi(Len_N, NUM_REFS)
Er_train = []

for jj in range(NUM_EPOCHS):
    # 准备输入数据
    Fx_input_stacked = Fx_data[:, jj]  # [2*Len_N]
    Fx_input = Fx_input_stacked.reshape(NUM_REFS, Len_N).T  # [Len_N, 2]
    Di_input = Di_data[:, jj]  # [Len_N]
    
    Er = maml_anc.maml_update(Fx_input, Di_input, MU, LAMBDA, EPSILON)
    Er_train.append(Er)
    
    if (jj + 1) % (NUM_EPOCHS // 20) == 0 or jj == 0:
        weight_energy = np.sum(maml_anc.Phi**2)
        print(f"Epoch {jj + 1}: Er = {Er:.6f}, Weight energy = {weight_energy:.6f}")

# 获取最终权重
Wc = maml_anc.Phi  # [2*Len_N]

print(f"\n✅ Multi-Reference MAML training completed!")
print(f"Final weights shape: {Wc.shape}")
print(f"Weight statistics:")
print(f"  RefL weights - Mean: {np.mean(Wc[:Len_N]):.6f}, Energy: {np.sum(Wc[:Len_N]**2):.6f}")
print(f"  RefR weights - Mean: {np.mean(Wc[Len_N:]):.6f}, Energy: {np.sum(Wc[Len_N:]**2):.6f}")

# ================= 测试数据准备 =================
print("\n🔄 Preparing test data...")

# 加载测试噪声
if not os.path.exists(TEST_NOISE_PATH):
    raise FileNotFoundError(f"Test noise file not found: {TEST_NOISE_PATH}")

fs_orig, test_audio_raw = wavfile.read(TEST_NOISE_PATH)

# 转换并归一化
if test_audio_raw.dtype == np.int16:
    Pri_1 = test_audio_raw.astype(np.float32) / 32768.0
else:
    Pri_1 = test_audio_raw.astype(np.float32)

if Pri_1.ndim > 1:
    Pri_1 = Pri_1[:, 0]

# 重采样到16kHz
if fs_orig != fs:
    Pri_1 = resample_to_target(Pri_1, fs_orig, fs)

print(f"✅ Test noise loaded: {len(Pri_1)} samples")

# 使用d20_90作为测试路径（对应MATLAB）
test_file = 'PriPath_Pri_d50_180.mat'
test_path = os.path.join(PATH_DIR, test_file)

if not os.path.exists(test_path):
    print(f"⚠️ Test file not found: {test_path}, using first training file instead")
    test_path = os.path.join(PATH_DIR, TRAIN_FILES[0])

test_data = sio.loadmat(test_path)
G_test = test_data["G_matrix"]

# 构造两个参考路径
P_refL_test = resample_to_target(G_test[:, 0], 48000, fs)  # ch1→ch2
P_refR_test = resample_to_target(G_test[:, 1], 48000, fs)  # ch1→ch3
P_err_test = resample_to_target(G_test[:, 2], 48000, fs)   # ch1→ch4

# 参考通道分别通过各自路径
x_refL = signal.lfilter(P_refL_test, [1.0], Pri_1)  # 电→refL
x_refR = signal.lfilter(P_refR_test, [1.0], Pri_1)  # 电→refR

# 通过次级路径S
Rf_L = signal.lfilter(S, [1.0], x_refL)
Rf_R = signal.lfilter(S, [1.0], x_refR)
Rf_test = np.column_stack([Rf_L, Rf_R])  # [N, 2] 双通道参考

# 构造误差信号
Dis_1 = signal.lfilter(P_err_test, [1.0], Pri_1)

print(f"✅ Test signals generated: Rf_test shape {Rf_test.shape}, Dis_1 length {len(Dis_1)}")

# ================= FxLMS测试 =================
print("\n📊 Running FxLMS tests...")

# 1. 零初始化
Wc_zero = np.zeros(2 * Len_N)
muw = 0.00001
print("  Testing zero-init FxLMS...")
Er_zero, _ = FxLMS_MultiRef(Len_N, Wc_zero, Dis_1, Rf_test, muw)

# 2. MAML初始化
print("  Testing MAML-init FxLMS...")
Er_maml, _ = FxLMS_MultiRef(Len_N, Wc, Dis_1, Rf_test, 5*muw)

# 3. Normalize初始化（与MATLAB对应）
fan_in = Len_N
fan_out = Len_N * 2
limit = np.sqrt(6) / 512
Wc_norm = (2 * np.random.rand(2 * Len_N) - 1) * limit
print("  Testing Normalize-init FxLMS...")
Er_norm, _ = FxLMS_MultiRef(Len_N, Wc_norm, Dis_1, Rf_test, muw)

print("✅ FxLMS tests completed!")

# ================= 结果分析 =================
print("\n" + "=" * 60)
print("Multi-Reference Results Analysis")
print("=" * 60)

def compute_mse_db(signal, L0=94):
    """计算MSE（dB SPL）"""
    return 10 * np.log10(np.mean(signal**2) + 1e-10) + L0

# 计算平均MSE
L0 = 94  # 参考SPL基准
avg_off = compute_mse_db(Dis_1, L0)
avg_zero = compute_mse_db(Er_zero, L0)
avg_norm = compute_mse_db(Er_norm, L0)
avg_maml = compute_mse_db(Er_maml, L0)

print("📊 Average MSE (dB SPL) on test position:")
print(f"  ANC off:           {avg_off:.2f} dB")
print(f"  Zero-init:         {avg_zero:.2f} dB")
print(f"  Normalize-init:    {avg_norm:.2f} dB")
print(f"  MAML-init:         {avg_maml:.2f} dB")
print(f"  MAML improvement:  {avg_zero - avg_maml:.2f} dB")

# ================= 可视化 =================
print("\n🔄 Creating visualizations...")

plt.figure(figsize=(18, 12))

# 1. 训练误差曲线
plt.subplot(3, 4, 1)
plt.plot(Er_train)
plt.title('Multi-Ref MAML Training Error')
plt.xlabel('Epoch')
plt.ylabel('Training Error')
plt.grid(True)

# 2. MAML权重（分通道显示）
plt.subplot(3, 4, 2)
plt.plot(Wc[:Len_N], label='RefL weights', alpha=0.8)
plt.plot(Wc[Len_N:], label='RefR weights', alpha=0.8)
plt.title('MAML Weights (2 References)')
plt.xlabel('Coefficient Index')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)

# 3. 时域信号比较
plt.subplot(3, 4, 3)
t = np.arange(min(10000, len(Dis_1))) / fs
plt.plot(t, Dis_1[:len(t)], 'k', label='ANC off', alpha=0.7)
plt.plot(t, Er_zero[:len(t)], 'b--', label='Zero-init', alpha=0.8)
plt.plot(t, Er_norm[:len(t)], 'g', label='Norm-init', alpha=0.8)
plt.plot(t, Er_maml[:len(t)], 'r', label='MAML-init', alpha=0.8)
plt.title('Time Domain Comparison')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 4. MSE对比柱状图
plt.subplot(3, 4, 4)
categories = ['ANC off', 'Zero', 'Normalize', 'MAML']
mse_values = [avg_off, avg_zero, avg_norm, avg_maml]
colors = ['black', 'blue', 'green', 'red']
bars = plt.bar(categories, mse_values, color=colors, alpha=0.7)
plt.title('Average MSE Comparison')
plt.ylabel('MSE (dB SPL)')
plt.ylim([40, 100])
plt.grid(True, alpha=0.3)
for bar, val in zip(bars, mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{val:.1f}', ha='center', va='bottom')

# 5. 滑动MSE曲线
plt.subplot(3, 4, 5)
win_len = 4096

# 使用滑动平均计算MSE
from scipy.ndimage import uniform_filter1d
mse_off = 10*np.log10(uniform_filter1d(Dis_1**2, win_len, mode='constant') + 1e-10) + L0
mse_zero = 10*np.log10(uniform_filter1d(Er_zero**2, win_len, mode='constant') + 1e-10) + L0
mse_maml = 10*np.log10(uniform_filter1d(Er_maml**2, win_len, mode='constant') + 1e-10) + L0

t_mse = np.arange(len(Dis_1)) / fs
plt.plot(t_mse, mse_off, 'k', linewidth=2, label='ANC off')
plt.plot(t_mse, mse_zero, 'b--', linewidth=2, label='Zero-init')
plt.plot(t_mse, mse_maml, 'r', linewidth=2, label='MAML-init')
plt.title('MSE Evolution (dB SPL)')
plt.xlabel('Time (s)')
plt.ylabel('MSE (dB)')
plt.ylim([45, 95])
plt.legend()
plt.grid(True)

# 6. 频谱分析
plt.subplot(3, 4, 6)
# 提取稳态部分（后半段）
E_tail_off = Dis_1[len(Dis_1)//2:]
E_tail_zero = Er_zero[len(Er_zero)//2:]
E_tail_maml = Er_maml[len(Er_maml)//2:]

# Welch频谱估计
nfft = 2048
f, Pxx_off = signal.welch(E_tail_off, fs=fs, nperseg=nfft)
_, Pxx_zero = signal.welch(E_tail_zero, fs=fs, nperseg=nfft)
_, Pxx_maml = signal.welch(E_tail_maml, fs=fs, nperseg=nfft)

# 转为SPL
sensitivity = 36
L0_spec = 20*np.log10(5e4*11.9/sensitivity) + 40
SPL_off = L0_spec + 10*np.log10(Pxx_off)
SPL_zero = L0_spec + 10*np.log10(Pxx_zero)
SPL_maml = L0_spec + 10*np.log10(Pxx_maml)

plt.plot(f, SPL_off, 'k', linewidth=1.8, label='ANC off')
plt.plot(f, SPL_zero, 'b--', linewidth=1.8, label='Zero-init')
plt.plot(f, SPL_maml, 'r', linewidth=1.8, label='MAML-init')
plt.xlabel('Frequency (Hz)')
plt.ylabel('SPL (dB)')
plt.xlim([20, 1000])
plt.ylim([-30, 100])
plt.legend()
plt.grid(True)

# 7. 权重能量分布
plt.subplot(3, 4, 7)
energy_L = np.sum(Wc[:Len_N]**2)
energy_R = np.sum(Wc[Len_N:]**2)
plt.bar(['RefL', 'RefR'], [energy_L, energy_R], alpha=0.7)
plt.title('Weight Energy Distribution')
plt.ylabel('Energy')
plt.grid(True, alpha=0.3)

# 8. 权重相关性
plt.subplot(3, 4, 8)
corr = np.corrcoef(Wc[:Len_N], Wc[Len_N:])[0, 1]
plt.text(0.5, 0.5, f'Correlation: {corr:.3f}', 
         fontsize=14, ha='center', va='center')
plt.title('RefL-RefR Weight Correlation')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.axis('off')

# 9. 训练收敛（滑动平均）
plt.subplot(3, 4, 9)
if len(Er_train) > 100:
    window_train = max(50, len(Er_train) // 50)
    moving_avg = np.convolve(Er_train, np.ones(window_train)/window_train, mode='valid')
    plt.plot(moving_avg, linewidth=2)
    plt.title('Training Convergence (Moving Avg)')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)

# 10. 改进总结
plt.subplot(3, 4, 10)
improvements = [
    avg_zero - avg_maml,
    avg_norm - avg_maml,
]
plt.bar(['vs Zero-init', 'vs Norm-init'], improvements, alpha=0.7, color=['blue', 'green'])
plt.title('MAML Improvements (dB)')
plt.ylabel('Improvement (dB)')
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
plt.grid(True, alpha=0.3)
for i, val in enumerate(improvements):
    plt.text(i, val + (0.1 if val >= 0 else -0.3),
             f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top')

# 11. 权重频率响应
plt.subplot(3, 4, 11)
w_L, h_L = signal.freqz(Wc[:Len_N], worN=512, fs=fs)
w_R, h_R = signal.freqz(Wc[Len_N:], worN=512, fs=fs)
plt.plot(w_L, 20*np.log10(np.abs(h_L)), label='RefL filter')
plt.plot(w_R, 20*np.log10(np.abs(h_R)), label='RefR filter')
plt.title('Filter Frequency Response')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.xlim([0, 2000])
plt.legend()
plt.grid(True)

# 12. 收敛速度比较
plt.subplot(3, 4, 12)
# 计算累积误差能量
cumsum_zero = np.cumsum(Er_zero**2)
cumsum_maml = np.cumsum(Er_maml**2)
t_cum = np.arange(len(Er_zero)) / fs
plt.plot(t_cum, cumsum_zero/cumsum_zero[-1], 'b--', label='Zero-init', linewidth=2)
plt.plot(t_cum, cumsum_maml/cumsum_maml[-1], 'r', label='MAML-init', linewidth=2)
plt.title('Normalized Cumulative Error Energy')
plt.xlabel('Time (s)')
plt.ylabel('Normalized Energy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('multi_ref_maml_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ================= 额外测试：使用FxLMS训练单个位置 =================
print("\n" + "=" * 60)
print("Additional Test: Single-task FxLMS Training")
print("=" * 60)

# 选择一个训练位置（如d20_180）进行FxLMS训练
single_task_file = 'PriPath_Pri_d20_180.mat'
single_task_path = os.path.join(PATH_DIR, single_task_file)

if os.path.exists(single_task_path):
    print(f"🔄 Training FxLMS on single position: {single_task_file}")
    
    # 加载单任务路径
    d20_data = sio.loadmat(single_task_path)
    G_d20 = d20_data["G_matrix"]
    
    P_refL_d20 = resample_to_target(G_d20[:, 0], 48000, fs)
    P_refR_d20 = resample_to_target(G_d20[:, 1], 48000, fs)
    P_err_d20 = resample_to_target(G_d20[:, 2], 48000, fs)
    
    # 生成训练信号（使用相同的宽带噪声）
    x_train = signal.lfilter(broadband_filter, [1.0], np.random.randn(N))
    
    x_refL_train = signal.lfilter(P_refL_d20, [1.0], x_train)
    x_refR_train = signal.lfilter(P_refR_d20, [1.0], x_train)
    Rf_L_train = signal.lfilter(S, [1.0], x_refL_train)
    Rf_R_train = signal.lfilter(S, [1.0], x_refR_train)
    Rf_train = np.column_stack([Rf_L_train, Rf_R_train])
    
    Dis_train = signal.lfilter(P_err_d20, [1.0], x_train)
    
    # 训练FxLMS
    Wc_init_d20 = np.zeros(2 * Len_N)
    muw_train = 0.002
    print("  Training single-task FxLMS...")
    Er_d20_train, W_d20 = FxLMS_MultiRef(Len_N, Wc_init_d20, Dis_train, Rf_train, muw_train)
    
    # 在测试位置应用
    print("  Testing single-task FxLMS on test position...")
    muw_test = 0.001
    Er_d20, _ = FxLMS_MultiRef(Len_N, W_d20, Dis_1, Rf_test, muw_test)
    
    # 计算性能
    avg_d20 = compute_mse_db(Er_d20, L0)
    print(f"\n📊 Single-task FxLMS Results:")
    print(f"  Single-task init:  {avg_d20:.2f} dB")
    print(f"  vs Zero-init:      {avg_zero - avg_d20:.2f} dB improvement")
    print(f"  vs MAML-init:      {avg_d20 - avg_maml:.2f} dB worse than MAML")
    
    # 添加到最终比较图
    plt.figure(figsize=(12, 8))
    
    # 时域比较
    plt.subplot(2, 2, 1)
    t_plot = np.arange(min(20000, len(Dis_1))) / fs
    plt.plot(t_plot, Dis_1[:len(t_plot)], 'k', label='ANC off', alpha=0.6, linewidth=1)
    plt.plot(t_plot, Er_zero[:len(t_plot)], 'b--', label='Zero-init', alpha=0.7, linewidth=1)
    plt.plot(t_plot, Er_d20[:len(t_plot)], 'orange', label='Single-task', alpha=0.8, linewidth=1.5)
    plt.plot(t_plot, Er_maml[:len(t_plot)], 'r', label='MAML-init', alpha=0.8, linewidth=1.5)
    plt.title('Final Comparison: Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    
    # MSE比较
    plt.subplot(2, 2, 2)
    categories_final = ['ANC off', 'Zero', 'Single-task', 'MAML']
    mse_values_final = [avg_off, avg_zero, avg_d20, avg_maml]
    colors_final = ['black', 'blue', 'orange', 'red']
    bars = plt.bar(categories_final, mse_values_final, color=colors_final, alpha=0.7)
    plt.title('Final MSE Comparison (dB SPL)')
    plt.ylabel('MSE (dB)')
    plt.ylim([40, 100])
    plt.grid(True, alpha=0.3)
    for bar, val in zip(bars, mse_values_final):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.1f}', ha='center', va='bottom')
    
    # 滑动MSE曲线
    plt.subplot(2, 2, 3)
    mse_d20 = 10*np.log10(uniform_filter1d(Er_d20**2, win_len, mode='constant') + 1e-10) + L0
    
    plt.plot(t_mse, mse_off, 'k', linewidth=2, label='ANC off')
    plt.plot(t_mse, mse_zero, 'b--', linewidth=2, label='Zero-init')
    plt.plot(t_mse, mse_d20, 'orange', linewidth=2, label='Single-task')
    plt.plot(t_mse, mse_maml, 'r', linewidth=2, label='MAML-init')
    plt.title('MSE Evolution with Single-task')
    plt.xlabel('Time (s)')
    plt.ylabel('MSE (dB SPL)')
    plt.ylim([45, 95])
    plt.legend(loc='best')
    plt.grid(True)
    
    # 频谱比较
    plt.subplot(2, 2, 4)
    E_tail_d20 = Er_d20[len(Er_d20)//2:]
    _, Pxx_d20 = signal.welch(E_tail_d20, fs=fs, nperseg=nfft)
    SPL_d20 = L0_spec + 10*np.log10(Pxx_d20)
    
    plt.plot(f, SPL_off, 'k', linewidth=1.8, label='ANC off')
    plt.plot(f, SPL_zero, 'b--', linewidth=1.8, label='Zero-init')
    plt.plot(f, SPL_d20, 'orange', linewidth=1.8, label='Single-task')
    plt.plot(f, SPL_maml, 'r', linewidth=1.8, label='MAML-init')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('SPL (dB)')
    plt.xlim([20, 1000])
    plt.ylim([-30, 100])
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.suptitle('Multi-Reference MAML vs Single-task Learning', fontsize=14)
    plt.tight_layout()
    plt.savefig('maml_vs_single_task.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    print(f"⚠️ Single-task file not found: {single_task_path}")

# ================= 最终总结 =================
print("\n" + "=" * 60)
print("FINAL SUMMARY - Multi-Reference MAML-ANC")
print("=" * 60)
print(f"\n📊 Configuration:")
print(f"  - References: {NUM_REFS}")
print(f"  - Filter length: {Len_N}")
print(f"  - Training paths: {len(TRAIN_FILES)}")
print(f"  - Training epochs: {NUM_EPOCHS}")
print(f"  - MAML parameters: μ={MU}, λ={LAMBDA}, ε={EPSILON}")

print(f"\n📊 Performance Summary (dB SPL):")
print(f"  {'Method':<20} {'MSE':<10} {'vs Zero':<12} {'vs MAML':<12}")
print(f"  {'-'*54}")
print(f"  {'ANC off':<20} {avg_off:>8.2f}")
print(f"  {'Zero-init':<20} {avg_zero:>8.2f} {'':<12} {avg_zero-avg_maml:>10.2f}")
print(f"  {'Normalize-init':<20} {avg_norm:>8.2f} {avg_zero-avg_norm:>10.2f} {avg_norm-avg_maml:>10.2f}")
if 'avg_d20' in locals():
    print(f"  {'Single-task':<20} {avg_d20:>8.2f} {avg_zero-avg_d20:>10.2f} {avg_d20-avg_maml:>10.2f}")
print(f"  {'MAML-init':<20} {avg_maml:>8.2f} {avg_zero-avg_maml:>10.2f} {'':<12} ✅")

print(f"\n📊 Key Findings:")
print(f"  - MAML provides {avg_zero-avg_maml:.2f} dB improvement over zero-init")
print(f"  - MAML provides {avg_norm-avg_maml:.2f} dB improvement over normalize-init")
if 'avg_d20' in locals():
    print(f"  - MAML provides {avg_d20-avg_maml:.2f} dB improvement over single-task learning")
print(f"  - Weight energy: RefL={np.sum(Wc[:Len_N]**2):.4f}, RefR={np.sum(Wc[Len_N:]**2):.4f}")
print(f"  - Weight correlation: {np.corrcoef(Wc[:Len_N], Wc[Len_N:])[0,1]:.3f}")

print("\n✅ Multi-Reference MAML-ANC Implementation Completed Successfully!")
print("=" * 60)