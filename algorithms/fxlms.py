# 🔶 多参考 FxLMS 算法

import numpy as np

def my_reshape(W, K, M):
    """
    Reshape filter matrix W of shape [Lw, K*M] to [K*Lw, M].
    """
    Lw = W.shape[0]
    reshaped_W = np.zeros((K * Lw, M))
    for i in range(K):
        reshaped_W[i * Lw:(i + 1) * Lw, :] = W[:, i * M:(i + 1) * M]
    return reshaped_W


def multi_ref_multi_chan_fxlms(Ref,
                               E,
                               filter_len,
                               sec_path,
                               stepsize,
                               init_W=None):
    """Multi-reference multi-channel FxLMS algorithm.

    Parameters
    ----------
    Ref:
        ``[Len × RefNum]`` reference signal.
    E:
        ``[Len × ErrNum]`` primary disturbance signal.
    filter_len:
        Length of the control filter ``Lw``.
    sec_path:
        ``[Ls × ChnSum]`` secondary path impulse responses.
    stepsize:
        Adaptation step size.
    init_W:
        Optional initial weight matrix ``[Lw × WSum]``.  When ``None`` the
        weights are initialised to zeros.
    """
    Len = E.shape[0]
    RefNum = Ref.shape[1]
    ChnSum = sec_path.shape[1]
    ErrNum = E.shape[1]
    CtrlNum = ChnSum // ErrNum  # number of secondary sources
    WSum = CtrlNum * RefNum     # number of total filters

    Ls = sec_path.shape[0]
    Lw = filter_len

    W = np.zeros((Lw, WSum)) if init_W is None else init_W.copy()
    y = np.zeros((CtrlNum, Len))
    s = np.zeros((ErrNum, Len))
    e = np.zeros((ErrNum, Len))

    d = E.T  # shape: [ErrNum × Len]

    Ref_buffer = np.zeros((Lw, RefNum))
    Ctrl_buffer = np.zeros((Lw, RefNum))
    FilterRef_buffer = np.zeros((Ls, RefNum))
    y_buffer = np.zeros((Ls, CtrlNum))
    Update_buffer = np.zeros((Lw, ChnSum * RefNum))

    for n in range(Len):
        Ref_buffer = np.vstack((Ref[n, :], Ref_buffer[:-1, :]))
        Ctrl_buffer = np.vstack((Ref[n, :], Ctrl_buffer[:-1, :]))
        FilterRef_buffer = np.vstack((Ref[n, :], FilterRef_buffer[:-1, :]))

        W_temp = my_reshape(W, RefNum, CtrlNum)  # [RefNum * Lw, CtrlNum]
        y[:, n] = np.dot(W_temp.T, Ctrl_buffer.flatten())

        y_buffer = np.vstack((y[:, n], y_buffer[:-1, :]))
        S_temp = my_reshape(sec_path, CtrlNum, ErrNum)  # [CtrlNum * Ls, ErrNum]
        s[:, n] = np.dot(S_temp.T, y_buffer.flatten())

        e[:, n] = d[:, n] + s[:, n]

        FilterRef = np.dot(sec_path.T, FilterRef_buffer).reshape(-1)
        Update_buffer = np.vstack((FilterRef, Update_buffer[:-1, :]))

        for i in range(WSum):
            X = Update_buffer[:, ErrNum * i:ErrNum * (i + 1)]
            W[:, i] = W[:, i] - stepsize * np.dot(X, e[:, n])

    return W, e.T  # W: [Lw × WSum], e: [Len × ErrNum]

