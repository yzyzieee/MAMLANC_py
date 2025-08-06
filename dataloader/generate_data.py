# 🔶 路径加载、白噪合成、训练样本生成

import os
from typing import List, Tuple, Optional

import numpy as np
import scipy.io as sio
import scipy.signal as signal

def resample_to_target(x: np.ndarray, fs_orig: int, fs_target: int) -> np.ndarray:
    """Resample signal from fs_orig to fs_target using polyphase filter."""
    return signal.resample_poly(x, fs_target, fs_orig, axis=0)

def generate_anc_training_data(path_dir: str,
                                train_files: List[str],
                                sec_path_file: str,
                                N_epcho: int,
                                Len_N: int,
                                fs: int = 16000,
                                broadband_len: Optional[int] = None,
                                ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate ANC training data following the MATLAB reference logic.

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

    # Step 1: Load primary paths (electric→ref and electric→err)
    all_ref_path, all_err_path = [], []
    for fname in train_files:
        dat = sio.loadmat(os.path.join(path_dir, fname))
        G = dat["G_matrix"]  # columns: ch1→ch2, ch1→ch3, ch1→ch4, ch1→ch5

        h_ref = resample_to_target(G[:, 0], 48000, fs)  # electric→ref (ch2)
        h_err = resample_to_target(G[:, 2], 48000, fs)  # electric→err (ch4)

        all_ref_path.append(h_ref)
        all_err_path.append(h_err)

    # Step 2: Load and resample secondary path (ref→err)
    sec_dat = sio.loadmat(sec_path_file)
    sec_key = list(sec_dat.keys())[-1]
    S_48k = sec_dat[sec_key][:, 0]
    S = resample_to_target(S_48k, 48000, fs)

    # Step 3: Generate broadband excitation
    white = np.random.randn(broadband_len)
    broadband_filter = signal.firwin(513, [0.015, 0.25], pass_zero=False)
    broadband = signal.lfilter(broadband_filter, [1.0], white)

    # Step 4: Assemble training samples
    Fx_data = np.zeros((Len_N, N_epcho))
    Di_data = np.zeros((Len_N, N_epcho))

    for jj in range(N_epcho):
        idx = np.random.randint(len(train_files))
        P_ref = all_ref_path[idx]
        P_err = all_err_path[idx]

        x_ref = signal.lfilter(P_ref, [1.0], broadband)
        xprime = signal.lfilter(S, [1.0], x_ref)
        d = signal.lfilter(P_err, [1.0], broadband)

        idx_cut = np.random.randint(Len_N, len(d))
        Fx_data[:, jj] = xprime[idx_cut - Len_N: idx_cut]
        Di_data[:, jj] = d[idx_cut - Len_N: idx_cut]

    return Fx_data, Di_data


def generate_task_batch(length: int,
                        num_refs: int,
                        with_secondary: bool = False,
                        num_errs: int = 1,
                        sec_len: int = 128):
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

    Returns:
        If ``with_secondary`` is True:
            Tuple[np.ndarray, np.ndarray, np.ndarray]
            → (Ref, E, sec_path)
        Else:
            Tuple[np.ndarray, np.ndarray]
            → (Ref, Di)
    """

    Ref = np.random.randn(length, num_refs)
    Di = np.random.randn(length, num_errs)

    if with_secondary:
        sec_path = np.random.randn(sec_len, num_refs * num_errs)
        return Ref, Di, sec_path

    return Ref, Di
