"""Multi-reference FxLMS and NFxLMS algorithms."""

from typing import Optional, Tuple

import numpy as np


def my_reshape(W: np.ndarray, K: int, M: int) -> np.ndarray:
    """Reshape stacked filters.

    Parameters
    ----------
    W:
        Filter matrix ``[Lw, K*M]`` where each group of ``M`` columns
        corresponds to one reference signal.
    K:
        Number of reference signals.
    M:
        Number of secondary sources.

    Returns
    -------
    np.ndarray
        Reshaped matrix of shape ``[K*Lw, M]`` where each column contains
        the ``K`` stacked filters for a given secondary source.
    """

    Lw = W.shape[0]
    reshaped_W = np.zeros((K * Lw, M))
    for i in range(K):
        reshaped_W[i * Lw:(i + 1) * Lw, :] = W[:, i * M:(i + 1) * M]
    return reshaped_W


def multi_ref_multi_chan_fxlms(
    Ref: np.ndarray,
    E: np.ndarray,
    filter_len: int,
    sec_path: np.ndarray,
    stepsize: float,
    delta: Optional[float] = None,
    init_W: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Multi-reference, multi-channel FxLMS/NFxLMS algorithm.

    Parameters
    ----------
    Ref : np.ndarray
        Reference signals with shape ``[Len, RefNum]`` (rows are time).
    E : np.ndarray
        Primary disturbance signals ``[Len, ErrNum]``.
    filter_len : int
        Length ``Lw`` of each control filter.
    sec_path : np.ndarray
        Secondary-path impulse responses ``[Ls, ChnSum]`` where
        ``ChnSum = CtrlNum * ErrNum``.
    stepsize : float
        Adaptation step size ``mu``.
    delta : float, optional
        Regularisation term for normalised FxLMS.  When provided the update
        step is scaled by ``1 / (||X||^2 + delta)``.  If ``None`` the plain
        FxLMS algorithm is used.
    init_W : np.ndarray, optional
        Initial weights ``[Lw, WSum]``.  Defaults to zeros.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        ``Ww``: final control filters ``[Lw, WSum]`` and ``ew``: error
        signals ``[Len, ErrNum]`` (rows are time).
    """

    Len = E.shape[0]                # number of time samples
    RefNum = Ref.shape[1]           # number of reference signals
    ChnSum = sec_path.shape[1]      # total number of secondary paths
    ErrNum = E.shape[1]             # number of error sensors
    CtrlNum = ChnSum // ErrNum      # number of secondary sources
    WSum = CtrlNum * RefNum         # total number of adaptive filters

    Ls = sec_path.shape[0]          # secondary-path filter length
    Lw = filter_len                 # control filter length

    # Adaptive filter coefficients [Lw, WSum]
    W = np.zeros((Lw, WSum)) if init_W is None else init_W.copy()
    # Secondary source outputs [CtrlNum, Len]
    y = np.zeros((CtrlNum, Len))
    # Control signals at error sensors [ErrNum, Len]
    s = np.zeros((ErrNum, Len))
    # Resulting error signals [ErrNum, Len]
    e = np.zeros((ErrNum, Len))

    d = E.T  # primary disturbances as [ErrNum, Len]

    # Buffers storing the last ``Lw`` or ``Ls`` samples
    Ref_buffer = np.zeros((Lw, RefNum))
    Ctrl_buffer = np.zeros((Lw, RefNum))
    FilterRef_buffer = np.zeros((Ls, RefNum))
    y_buffer = np.zeros((Ls, CtrlNum))
    Update_buffer = np.zeros((Lw, ChnSum * RefNum))

    for n in range(Len):
        # Update input buffers (newest sample on top)
        Ref_buffer = np.vstack((Ref[n, :], Ref_buffer[:-1, :]))
        Ctrl_buffer = np.vstack((Ref[n, :], Ctrl_buffer[:-1, :]))
        FilterRef_buffer = np.vstack((Ref[n, :], FilterRef_buffer[:-1, :]))

        # Secondary source output calculation
        W_temp = my_reshape(W, RefNum, CtrlNum)  # [RefNum*Lw, CtrlNum]
        y[:, n] = np.dot(W_temp.T, Ctrl_buffer.reshape(-1))

        # Propagate through secondary path to error sensors
        y_buffer = np.vstack((y[:, n], y_buffer[:-1, :]))
        S_temp = my_reshape(sec_path, CtrlNum, ErrNum)  # [CtrlNum*Ls, ErrNum]
        s[:, n] = np.dot(S_temp.T, y_buffer.reshape(-1))

        # Sum with primary disturbance to get error
        e[:, n] = d[:, n] + s[:, n]

        # Compute filtered reference for weight update
        FilterRef = np.dot(sec_path.T, FilterRef_buffer).reshape(-1)
        Update_buffer = np.vstack((FilterRef, Update_buffer[:-1, :]))

        for i in range(WSum):
            X = Update_buffer[:, ErrNum * i:ErrNum * (i + 1)]
            grad = np.dot(X, e[:, n])  # gradient estimate [Lw]
            if delta is None:
                W[:, i] -= stepsize * grad
            else:
                norm = np.sum(X ** 2) + delta
                W[:, i] -= stepsize / norm * grad

    return W, e.T  # W: [Lw, WSum], e: [Len, ErrNum]

