 # 🔶 常用信号函数（滤波、normalize、resample）
"""Signal processing utility functions.

Currently the project only requires a helper to evaluate and visualise the
mean squared error (MSE) of an error signal.  The helper implemented below is
tailored to audio processing conventions by expressing the squared error in
decibels (dB) and optionally plotting the curve over time.
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import matplotlib.pyplot as plt


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


