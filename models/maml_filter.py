# 🔶 封装 MAML 更新逻辑（内循环）
import numpy as np

class MAMLFilter(ControlFilter):
    """
    MAML-based filter initialization for ANC with multi-channel references.
    """
    def maml_initial(self, Fx, Di, mu, lamda, epslon):
        """
        Perform MAML-style initialization update.

        Args:
            Fx (np.ndarray): [Len_N × num_refs], time-domain reference signal.
            Di (np.ndarray): [Len_N × 1], desired signal (interference).
            mu (float): step size.
            lamda (float): forgetting factor.
            epslon (float): scaling factor for meta-gradient.

        Returns:
            float: Initial error (for tracking purposes).
        """
        Len_N, num_refs = Fx.shape

        # Flip time axis for both inputs (reverse time)
        Fx_flip = np.flipud(Fx)
        Di_flip = np.flipud(Di)

        # Concatenate reversed references into column vector
        Fx_concat = Fx_flip.flatten(order='F').reshape(-1, 1)

        # 1-step error and fast update
        e1 = Di_flip[0] - self.apply(Fx_concat)
        Wo = self.Phi + mu * e1 * Fx_concat  # fast-updated weights

        # Gradient accumulation
        Grad = np.zeros_like(self.Phi)
        for jj in range(Len_N):
            Fd_stack = []
            for ch in range(num_refs):
                Fd_j = np.concatenate([
                    Fx_flip[jj:, ch],
                    np.zeros(jj)
                ])
                Fd_stack.append(Fd_j)
            Fd_stack = np.concatenate(Fd_stack).reshape(-1, 1)

            e_j = Di_flip[jj] - float(np.dot(Wo.T, Fd_stack))
            Grad += epslon * (mu / Len_N) * e_j * Fd_stack * (lamda ** jj)

            if jj == 0:
                Er = e_j  # record first error

        # Meta-update
        self.Phi += Grad

        return float(Er)