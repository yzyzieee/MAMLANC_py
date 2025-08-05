# 🔶 路径加载、白噪合成、训练样本生成

import os
import numpy as np
import scipy.io as sio
import scipy.signal as signal
from typing import List, Tuple

def resample_to_target(x: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Resample signal from fs_orig to fs_target using polyphase filter."""
    return signal.resample_poly(x, fs_target, fs_orig, axis=0)

def generate_anc_training_data(path_dir: str,
                                train_files: List[str],
                                sec_path_file: str,
                                N_epcho: int,
                                Len_N: int,
                                fs: int = 16000,
                                broadband_len: int = None
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate ANC training data using real measured paths and synthetic broadband noise.

    Returns:
        Fx_data: [2*Len_N, N_epcho] double-channel reference input
        Di_data: [Len_N, N_epcho] error mic signal
    """
    if broadband_len is None:
        broadband_len = int(fs * 3)  # default 3 seconds

    # Step 1: Load primary paths
    all_ref_path, all_err_path = [], []
    for fname in train_files:
        dat = sio.loadmat(os.path.join(path_dir, fname))
        G = dat['G_matrix']  # shape: [?, 4]

        h_refL = resample_to_target(G[:, 0], 48000, fs)  # ch1→ch2
        h_refR = resample_to_target(G[:, 1], 48000, fs)  # ch1→ch3
        h_err  = resample_to_target(G[:, 2], 48000, fs)  # ch1→ch4

        all_ref_path.append(np.stack([h_refL, h_refR], axis=1))  # shape: [T, 2]
        all_err_path.append(h_err)  # shape: [T]

    # Step 2: Load secondary path
    sec_dat = sio.loadmat(sec_path_file)
    sec_key = list(sec_dat.keys())[-1]
    S_48k = sec_dat[sec_key][:, 0]  # [T] from ref → err
    S = resample_to_target(S_48k, 48000, fs)

    # Step 3: Generate broadband excitation signal
    white = np.random.randn(broadband_len)
    broadband_filter = signal.firwin(513, [0.015, 0.25], pass_zero=False)  # 120Hz–2kHz
    broadband = signal.lfilter(broadband_filter, [1.0], white)

    # Step 4: Generate training samples
    Fx_data = np.zeros((2 * Len_N, N_epcho))
    Di_data = np.zeros((Len_N, N_epcho))

    for jj in range(N_epcho):
        idx = np.random.randint(len(train_files))
        P_ref = all_ref_path[idx]  # shape: [T, 2]
        P_err = all_err_path[idx]  # shape: [T]

        P_ref_L, P_ref_R = P_ref[:, 0], P_ref[:, 1]

        x_ref_L = signal.lfilter(P_ref_L, [1.0], broadband)
        x_ref_R = signal.lfilter(P_ref_R, [1.0], broadband)

        xprime_L = signal.lfilter(S, [1.0], x_ref_L)
        xprime_R = signal.lfilter(S, [1.0], x_ref_R)

        d = signal.lfilter(P_err, [1.0], broadband)

        idx_cut = np.random.randint(Len_N, len(d))
        Di_data[:, jj] = d[idx_cut - Len_N: idx_cut]

        x1 = xprime_L[idx_cut - Len_N: idx_cut]
        x2 = xprime_R[idx_cut - Len_N: idx_cut]
        Fx_data[:, jj] = np.concatenate([x1, x2], axis=0)

    return Fx_data, Di_data