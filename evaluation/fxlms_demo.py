"""Demonstration of the multi-channel FxLMS algorithm.

This script runs a simple ANC simulation with two reference signals, two
secondary sources and two error microphones, using the
:func:`algorithms.fxlms.multi_ref_multi_chan_fxlms` implementation.  The
error convergence curve is saved as a figure for inspection.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from algorithms.fxlms import multi_ref_multi_chan_fxlms


def demo_2ref_2sec_2err(
    num_samples: int = 2000,
    filter_len: int = 8,
    stepsize: float = 0.1,
    seed: int | None = 0,
    save_path: str | None = "fxlms_convergence.png",
):
    """Run a 2×2×2 FxLMS ANC simulation and plot the convergence curve.

    The simulation uses fixed impulse responses:

    - Primary path ``[0, 0, 0, 1, 0.5, 0.2]`` from each reference to each
      error microphone.
    - Secondary path ``[0, 1, 0]`` from each secondary source to each error
      microphone (shorter delay than the primary path).

    Parameters
    ----------
    num_samples:
        Number of time-domain samples ``Len``.
    filter_len:
        Length ``Lw`` of each adaptive control filter.
    stepsize:
        Adaptation step size ``mu``.
    seed:
        Optional random seed for reproducible results.
    save_path:
        Destination for the convergence plot.  If ``None`` the figure is not
        written to disk.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``W`` the final control filters ``[Lw, WSum]``, ``e`` the error signals
        ``[Len, ErrNum]`` and ``err_curve`` the mean-square error per sample
        ``[Len]``.
    """

    rng = np.random.default_rng(seed)
    Ref = rng.standard_normal((num_samples, 2))

    # Primary-path responses from each reference to each error microphone
    primary_ir = np.array([0.0, 0.0, 0.0, 1.0, 0.5, 0.2])
    primary = np.tile(primary_ir[:, None], 4)
    E = np.zeros((num_samples, 2))
    for err in range(2):
        for ref in range(2):
            h = primary[:, err * 2 + ref]
            E[:, err] += np.convolve(Ref[:, ref], h, mode="same")

    # Secondary-path responses: two secondary sources -> two error mics
    sec_ir = np.array([0.0, 1.0, 0.0])
    sec_path = np.tile(sec_ir[:, None], 4)

    W, e = multi_ref_multi_chan_fxlms(Ref, E, filter_len, sec_path, stepsize)

    # Mean-square error across the two error microphones
    err_curve = np.mean(e**2, axis=1)

    plt.figure()
    plt.plot(err_curve)
    plt.xlabel("Samples")
    plt.ylabel("Mean square error")
    plt.title("2ref-2sec-2err FxLMS convergence")
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.close()

    return W, e, err_curve


if __name__ == "__main__":  # pragma: no cover - manual demo
    demo_2ref_2sec_2err()
