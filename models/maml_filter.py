"""MAML-style initialization for multi-reference ANC."""

import numpy as np

from .control_filter import ControlFilter


class MAMLFilter(ControlFilter):
    """Meta-learning based control filter for ANC."""

    def maml_initial(self, Fx, Di, mu, lamda, epsilon):
        """Perform one MAML initialization update.

        Args:
            Fx (np.ndarray): [Len_N × num_refs] time-domain reference signal.
            Di (np.ndarray): [Len_N × 1] desired signal (disturbance).
            mu (float): step size for the inner update.
            lamda (float): forgetting factor.
            epsilon (float): scaling factor for the meta-gradient.

        Returns:
            float: Residual error after the fast adaptation step.
        """

        Len_N, num_refs = Fx.shape

        # Reverse time axis for references and disturbance
        Fx_flip = np.flipud(Fx)
        Di_flip = np.flipud(Di)

        # Stack reversed references into a single column vector
        Fx_concat = Fx_flip.flatten(order="F").reshape(-1, 1)

        # One-step fast adaptation
        e1 = Di_flip[0] - self.predict(Fx_concat)
        Wo = self.Phi + mu * e1 * Fx_concat

        # Accumulate meta-gradient
        Grad = np.zeros_like(self.Phi)
        for jj in range(Len_N):
            Fd_stack = []
            for ch in range(num_refs):
                Fd_j = np.concatenate((Fx_flip[jj:, ch], np.zeros(jj)))
                Fd_stack.append(Fd_j)
            Fd_stack = np.concatenate(Fd_stack).reshape(-1, 1)

            e_j = Di_flip[jj] - float(np.dot(Wo.T, Fd_stack))
            Grad += epsilon * (mu / Len_N) * e_j * Fd_stack * (lamda ** jj)

            if jj == 0:
                Er = e_j

        # Meta update of initialization weights
        self.update(Grad)

        return float(Er)
