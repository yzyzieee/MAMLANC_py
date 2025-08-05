# 🔶 控制器结构封装（Phi = 参数向量）

import numpy as np

class ControlFilter:
    """
    Base adaptive filter structure.
    Holds the control filter weights (flattened across reference channels).
    """
    def __init__(self, filter_len: int, num_refs: int):
        """
        Initialize the filter.

        Args:
            filter_len (int): Length of control filter for one reference channel.
            num_refs (int): Number of reference channels.
        """
        self.filter_len = filter_len
        self.num_refs = num_refs
        self.Phi = np.zeros((filter_len * num_refs, 1))  # Column vector

    def apply(self, Fx_concat):
        """
        Apply the filter to the concatenated reference signal.

        Args:
            Fx_concat (np.ndarray): [filter_len * num_refs × 1]

        Returns:
            float: Output signal (scalar)
        """
        return float(np.dot(self.Phi.T, Fx_concat))