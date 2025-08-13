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
        Er[n] = Dis[n] - control_signal
    
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

# ================= FxLMS测试（静态+动态） =================
print("\n🔄 Running FxLMS tests...")

# ================= 1. 静态滤波测试（对应MATLAB的muw=0.000） =================
print("📊 Static filter tests (like MATLAB with muw=0.000)...")

# 对应MATLAB: Wc_zero = zeros(Len_N,1);
Wc_zero = np.zeros(FILTER_LEN)

# 对应MATLAB: [Er_zero,~] = FxLMS(Len_N, Wc_zero, Dis_1, Rf_1, muw);
Er_zero_static = static_filter_test(Wc_zero, Dis_1, Rf_1)

# 对应MATLAB: [Er_maml,~] = FxLMS(Len_N, Wc, Dis_1, Rf_1, 1*muw);
Er_maml_static = static_filter_test(maml_anc.Phi, Dis_1, Rf_1)

print(f"Static filter tests completed!")

# ================= 2. 动态自适应FxLMS测试 =================
print("🔄 Dynamic adaptive FxLMS tests...")

# 首先导入FxLMS类
from algorithms.MultChanFxLMS import MultChanFxLMS

# 准备次级路径（单通道情况）
sec_path_for_fxlms = S.reshape(-1, 1)  # [filter_len, 1]

print(f"Preparing FxLMS with secondary path shape: {sec_path_for_fxlms.shape}")
print(f"Secondary path energy: {np.sum(sec_path_for_fxlms**2):.6f}")

# 2.1 零初始化的自适应FxLMS
print("🔄 Zero-initialized adaptive FxLMS...")
fxlms_zero = MultChanFxLMS(
    ref_num=1,
    err_num=1, 
    ctrl_num=1,
    filter_len=FILTER_LEN,
    sec_path=sec_path_for_fxlms,
    stepsize=0.000001,  # 小步长确保稳定
)

# 准备FxLMS的输入格式
x_ref_fxlms = x_ref_test.reshape(-1, 1)      # [N, 1] 参考信号  
Dis_1_fxlms = Dis_1.reshape(-1, 1)    # [N, 1] 干扰信号

print(f"FxLMS input shapes: x_ref={x_ref_fxlms.shape}, Dis_1={Dis_1_fxlms.shape}")

# 运行零初始化FxLMS
try:
    _, Er_zero_adaptive = fxlms_zero.process_batch(x_ref_fxlms, Dis_1_fxlms)
    Er_zero_adaptive = Er_zero_adaptive[:, 0]  # 取出单通道结果
    print(f"✅ Zero-init adaptive FxLMS completed, error shape: {Er_zero_adaptive.shape}")
    
    # 获取最终权重
    W_zero_final = fxlms_zero.get_weights()[:, 0]
    print(f"Zero-init final weight energy: {np.sum(W_zero_final**2):.6f}")
    
except Exception as e:
    print(f"❌ CRITICAL ERROR in zero-init adaptive FxLMS: {e}")
    raise RuntimeError(f"Zero-init adaptive FxLMS failed: {e}")

# 2.2 MAML初始化的自适应FxLMS
print("🔄 MAML-initialized adaptive FxLMS...")
fxlms_maml = MultChanFxLMS(
    ref_num=1,
    err_num=1, 
    ctrl_num=1,
    filter_len=FILTER_LEN,
    sec_path=sec_path_for_fxlms,
    stepsize=0.000001,  # 相同步长
)

# 设置MAML初始权重
print(f"Setting MAML weights to FxLMS...")
print(f"MAML weight shape: {maml_anc.Phi.shape}")
print(f"FxLMS weight shape: {fxlms_maml.weights.shape}")

# 验证并设置权重
if fxlms_maml.weights.shape[1] >= 1:
    fxlms_maml.weights[:, 0] = maml_anc.Phi.copy()
    
    # 验证设置是否成功
    if np.allclose(fxlms_maml.weights[:, 0], maml_anc.Phi, atol=1e-10):
        print("✅ MAML weights successfully set to FxLMS")
        print(f"Initial weight energy: {np.sum(fxlms_maml.weights[:, 0]**2):.6f}")
    else:
        raise RuntimeError("MAML weight setting verification failed")
else:
    raise RuntimeError(f"FxLMS weights shape incompatible: {fxlms_maml.weights.shape}")

# 运行MAML初始化FxLMS
try:
    _, Er_maml_adaptive = fxlms_maml.process_batch(x_ref_fxlms, Dis_1_fxlms)
    Er_maml_adaptive = Er_maml_adaptive[:, 0]  # 取出单通道结果
    print(f"✅ MAML-init adaptive FxLMS completed, error shape: {Er_maml_adaptive.shape}")
    
    # 获取最终权重
    W_maml_final = fxlms_maml.get_weights()[:, 0]
    print(f"MAML-init final weight energy: {np.sum(W_maml_final**2):.6f}")
    
except Exception as e:
    print(f"❌ CRITICAL ERROR in MAML-init adaptive FxLMS: {e}")
    raise RuntimeError(f"MAML-init adaptive FxLMS failed: {e}")

print(f"Dynamic FxLMS tests completed!")

# ================= 结果分析 =================
print("\n" + "=" * 60)
print("Results Analysis")
print("=" * 60)

# 计算MSE (对应MATLAB的平均MSE计算)
def compute_mse_db(signal):
    return 10 * np.log10(np.mean(signal**2) + 1e-10)

# 静态测试结果
print("📊 Static Filter Test Results (like MATLAB):")
mse_original = compute_mse_db(Dis_1)
mse_zero_static = compute_mse_db(Er_zero_static)
mse_maml_static = compute_mse_db(Er_maml_static)

print(f"  Original (ANC off):     {mse_original:.2f} dB")
print(f"  Zero-init (static):     {mse_zero_static:.2f} dB")
print(f"  MAML-init (static):     {mse_maml_static:.2f} dB")
print(f"  MAML static improvement: {mse_zero_static - mse_maml_static:.2f} dB")

if mse_maml_static < mse_zero_static:
    print("🎉 MAML shows static improvement over zero initialization!")
else:
    print("⚠️ MAML does not show static improvement")

# 动态测试结果
print("\n🔄 Dynamic Adaptive FxLMS Results:")
mse_zero_adaptive = compute_mse_db(Er_zero_adaptive)
mse_maml_adaptive = compute_mse_db(Er_maml_adaptive)

print(f"  Zero-init (adaptive):   {mse_zero_adaptive:.2f} dB")
print(f"  MAML-init (adaptive):   {mse_maml_adaptive:.2f} dB")
print(f"  MAML adaptive improvement: {mse_zero_adaptive - mse_maml_adaptive:.2f} dB")

if mse_maml_adaptive < mse_zero_adaptive:
    print("🎉 MAML shows adaptive improvement over zero initialization!")
else:
    print("⚠️ MAML does not show adaptive improvement")

# 收敛速度分析
print("\n📈 Convergence Analysis:")
# 计算前10%和后10%的MSE来评估收敛速度
n_samples = len(Er_zero_adaptive)
early_samples = n_samples // 10  # 前10%
late_start = int(n_samples * 0.9)  # 后10%

# 零初始化收敛分析
zero_early_mse = compute_mse_db(Er_zero_adaptive[:early_samples])
zero_late_mse = compute_mse_db(Er_zero_adaptive[late_start:])
zero_improvement = zero_early_mse - zero_late_mse

# MAML初始化收敛分析  
maml_early_mse = compute_mse_db(Er_maml_adaptive[:early_samples])
maml_late_mse = compute_mse_db(Er_maml_adaptive[late_start:])
maml_improvement = maml_early_mse - maml_late_mse

print(f"Zero-init: Early={zero_early_mse:.2f} dB, Late={zero_late_mse:.2f} dB, Improvement={zero_improvement:.2f} dB")
print(f"MAML-init: Early={maml_early_mse:.2f} dB, Late={maml_late_mse:.2f} dB, Improvement={maml_improvement:.2f} dB")
print(f"MAML early advantage: {zero_early_mse - maml_early_mse:.2f} dB")
print(f"MAML final advantage: {zero_late_mse - maml_late_mse:.2f} dB")

# 权重变化分析
print("\n🏋️ Weight Evolution Analysis:")
weight_change_zero = np.sum((W_zero_final - np.zeros(FILTER_LEN))**2)
weight_change_maml = np.sum((W_maml_final - maml_anc.Phi)**2)

print(f"Zero-init weight change: {weight_change_zero:.6f}")
print(f"MAML-init weight change: {weight_change_maml:.6f}")
print(f"MAML started with energy: {np.sum(maml_anc.Phi**2):.6f}")
print(f"MAML ended with energy:   {np.sum(W_maml_final**2):.6f}")

# ================= 可视化 =================
print("\n🔄 Creating visualizations...")

plt.figure(figsize=(18, 12))

# 训练误差
plt.subplot(3, 4, 1)
plt.plot(Er_train)
plt.title('MAML Training Error')
plt.xlabel('Epoch')
plt.ylabel('Training Error')
plt.grid(True)

# MAML权重
plt.subplot(3, 4, 2)
plt.plot(maml_anc.Phi)
plt.title('MAML Weights')
plt.xlabel('Coefficient Index')
plt.ylabel('Weight Value')
plt.grid(True)

# 静态测试时域比较
plt.subplot(3, 4, 3)
t = np.arange(min(10000, len(Dis_1))) / 16000  # 只显示前10000个样本
plt.plot(t, Dis_1[:len(t)], label='Original', alpha=0.7, linewidth=1)
plt.plot(t, Er_zero_static[:len(t)], label='Zero-init (Static)', alpha=0.8, linewidth=1)
plt.plot(t, Er_maml_static[:len(t)], label='MAML-init (Static)', alpha=0.8, linewidth=1)
plt.title('Static Test Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 动态测试时域比较
plt.subplot(3, 4, 4)
plt.plot(t, Dis_1[:len(t)], label='Original', alpha=0.7, linewidth=1)
plt.plot(t, Er_zero_adaptive[:len(t)], label='Zero-init (Adaptive)', alpha=0.8, linewidth=1)
plt.plot(t, Er_maml_adaptive[:len(t)], label='MAML-init (Adaptive)', alpha=0.8, linewidth=1)
plt.title('Adaptive Test Time Domain')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)

# 静态MSE对比
plt.subplot(3, 4, 5)
static_mse_values = [mse_original, mse_zero_static, mse_maml_static]
static_labels = ['Original', 'Zero-static', 'MAML-static']
colors = ['red', 'blue', 'green']
bars1 = plt.bar(static_labels, static_mse_values, color=colors, alpha=0.7)
plt.title('Static MSE Comparison')
plt.ylabel('MSE (dB)')
plt.grid(True, alpha=0.3)
# 添加数值标签
for bar, val in zip(bars1, static_mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}', ha='center', va='bottom')

# 动态MSE对比
plt.subplot(3, 4, 6)
adaptive_mse_values = [mse_original, mse_zero_adaptive, mse_maml_adaptive]
adaptive_labels = ['Original', 'Zero-adaptive', 'MAML-adaptive']
bars2 = plt.bar(adaptive_labels, adaptive_mse_values, color=colors, alpha=0.7)
plt.title('Adaptive MSE Comparison')
plt.ylabel('MSE (dB)')
plt.grid(True, alpha=0.3)
# 添加数值标签
for bar, val in zip(bars2, adaptive_mse_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{val:.1f}', ha='center', va='bottom')

# 收敛曲线对比（滑动平均MSE）
plt.subplot(3, 4, 7)
window_size = max(100, len(Er_zero_adaptive) // 50)
t_conv = np.arange(len(Er_zero_adaptive)) / 16000

# 计算滑动MSE
zero_mse_curve = []
maml_mse_curve = []
for i in range(len(Er_zero_adaptive)):
    start_idx = max(0, i - window_size + 1)
    zero_mse_curve.append(compute_mse_db(Er_zero_adaptive[start_idx:i+1]))
    maml_mse_curve.append(compute_mse_db(Er_maml_adaptive[start_idx:i+1]))

plt.plot(t_conv, zero_mse_curve, label='Zero-init', linewidth=2)
plt.plot(t_conv, maml_mse_curve, label='MAML-init', linewidth=2)
plt.title('Convergence Curves (Moving MSE)')
plt.xlabel('Time (s)')
plt.ylabel('MSE (dB)')
plt.legend()
plt.grid(True)

# 权重演化对比
plt.subplot(3, 4, 8)
plt.plot(np.zeros(FILTER_LEN), label='Zero initial', linewidth=2, alpha=0.8)
plt.plot(maml_anc.Phi, label='MAML initial', linewidth=2, alpha=0.8)
plt.plot(W_zero_final, label='Zero final', linewidth=2, alpha=0.8)
plt.plot(W_maml_final, label='MAML final', linewidth=2, alpha=0.8)
plt.title('Weight Evolution')
plt.xlabel('Coefficient Index')
plt.ylabel('Weight Value')
plt.legend()
plt.grid(True)

# 权重能量演化
plt.subplot(3, 4, 9)
plt.plot(np.cumsum(maml_anc.Phi**2), label='MAML initial', linewidth=2)
plt.plot(np.cumsum(W_zero_final**2), label='Zero final', linewidth=2)  
plt.plot(np.cumsum(W_maml_final**2), label='MAML final', linewidth=2)
plt.title('Cumulative Weight Energy')
plt.xlabel('Coefficient Index')
plt.ylabel('Cumulative Energy')
plt.legend()
plt.grid(True)

# 训练收敛（滑动平均）
plt.subplot(3, 4, 10)
if len(Er_train) > 100:
    window_size_train = max(50, len(Er_train) // 50)
    moving_avg = np.convolve(Er_train, np.ones(window_size_train)/window_size_train, mode='valid')
    plt.plot(moving_avg, linewidth=2)
    plt.title('MAML Training Convergence')
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.grid(True)

# 性能提升总结
plt.subplot(3, 4, 11)
improvements = [
    mse_zero_static - mse_maml_static,    # 静态改进
    mse_zero_adaptive - mse_maml_adaptive, # 动态改进
    zero_early_mse - maml_early_mse,      # 早期优势
    zero_late_mse - maml_late_mse         # 最终优势
]
improvement_labels = ['Static', 'Adaptive', 'Early', 'Final']
colors_imp = ['lightblue', 'lightgreen', 'orange', 'pink']
bars3 = plt.bar(improvement_labels, improvements, color=colors_imp, alpha=0.8)
plt.title('MAML Improvements (dB)')
plt.ylabel('Improvement (dB)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
# 添加数值标签
for bar, val in zip(bars3, improvements):
    plt.text(bar.get_x() + bar.get_width()/2, 
             bar.get_height() + (0.1 if val >= 0 else -0.3), 
             f'{val:.2f}', ha='center', va='bottom' if val >= 0 else 'top')

# 频域分析（功率谱比较）
plt.subplot(3, 4, 12)
from scipy import signal as scipy_signal
f, Pxx_orig = scipy_signal.welch(Dis_1, fs=16000, nperseg=1024)
_, Pxx_zero = scipy_signal.welch(Er_zero_adaptive, fs=16000, nperseg=1024)
_, Pxx_maml = scipy_signal.welch(Er_maml_adaptive, fs=16000, nperseg=1024)

plt.semilogy(f, Pxx_orig, label='Original', alpha=0.8)
plt.semilogy(f, Pxx_zero, label='Zero-init', alpha=0.8)
plt.semilogy(f, Pxx_maml, label='MAML-init', alpha=0.8)
plt.title('Power Spectral Density')
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD')
plt.xlim([0, 2000])  # 显示0-2kHz
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('maml_results_complete.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("Simplified MAML-ANC Completed Successfully!")
print("=" * 60)
