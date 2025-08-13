# ◼️ 常用信号函数（滤波、normalize、resample）

"""Lightweight signal processing helpers used across the project."""

from __future__ import annotations

import os
from typing import Iterable, Optional


import numpy as np
from scipy import signal


def normalize(x: np.ndarray) -> np.ndarray:
    """Normalise the signal to unit peak."""

    x = np.asarray(x, dtype=float)
    peak = np.max(np.abs(x)) + 1e-12
    return x / peak


def bandpass_filter(x: np.ndarray, fs: int, f_low: float, f_high: float, order: int = 4) -> np.ndarray:
    """Apply a Butterworth band-pass filter to ``x``."""

    b, a = signal.butter(order, [f_low / (fs / 2), f_high / (fs / 2)], btype="band")
    return signal.lfilter(b, a, x)


__all__ = ["normalize", "bandpass_filter"]

def compute_mse(
    err_signal: Iterable[float],
    fs: int = 1,
    plot_path: Optional[str] = None,
    show: bool = False,
) -> float:
    """Compute the mean squared error of ``err_signal`` in dB.

    Parameters
    ----------
    err_signal:
        Sequence containing the error samples.
    fs:
        Sampling frequency of the signal.  Used only for the time axis when
        plotting.
    plot_path:
        If provided, a plot of the instantaneous squared error in dB will be
        saved to this path.
    show:
        Display the plot in an interactive window.  This is ``False`` by
        default to keep automated runs non-blocking.

    Returns
    -------
    float
        The average MSE expressed in dB.
    """

    err = np.asarray(err_signal, dtype=float)
    mse_curve = err ** 2
    mse_db = 10 * np.log10(np.mean(mse_curve) + 1e-12)

    if plot_path or show:
        time_axis = np.arange(len(err)) / fs
        plt.figure()
        plt.plot(time_axis, 10 * np.log10(mse_curve + 1e-12))
        plt.xlabel("Time [s]")
        plt.ylabel("Squared Error (dB)")
        plt.title("MSE Curve")
        plt.grid(True)

        if plot_path:
            directory = os.path.dirname(plot_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            plt.savefig(plot_path)

        if show:
            plt.show()
        else:
            plt.close()

    return float(mse_db)
