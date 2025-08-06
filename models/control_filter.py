# ◼️ 控制器结构封装（Phi = 参数向量）

"""Basic adaptive control filter classes used by the project."""

from __future__ import annotations

import numpy as np


class ControlFilter:
    """Base adaptive filter structure.

    Parameters
    ----------
    filter_len:
        Length of the FIR control filter for a single reference channel.
    num_refs:
        Number of reference channels.  The internal weight vector ``Phi`` is
        stored as a column vector with ``filter_len * num_refs`` elements.
    """

    def __init__(self, filter_len: int, num_refs: int) -> None:
        self.filter_len = filter_len
        self.num_refs = num_refs
        self.Phi = np.zeros((filter_len * num_refs, 1))  # column vector

    # ------------------------------------------------------------------
    # Interfaces expected by the project specification
    # ------------------------------------------------------------------
    def predict(self, Fx_concat: np.ndarray) -> float:
        """Compute the filter output for the given stacked reference signal."""

        return float(np.dot(self.Phi.T, Fx_concat))

    def update(self, gradient: np.ndarray) -> None:
        """Update the internal weights using the provided gradient."""

        self.Phi += gradient


__all__ = ["ControlFilter"]

