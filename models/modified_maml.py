"""PyTorch implementation of a modified MAML algorithm for multi-channel ANC."""

import torch
from torch import nn
from torch.autograd import Variable


def build_delay_line(len_c: int, Fx: torch.Tensor) -> torch.Tensor:
    """Construct a delay-line representation of the filtered references.

    Args:
        len_c (int): Length of the control filter or delay line.
        Fx (torch.Tensor): Filtered reference of shape
            ``[num_ref, num_sec, num_err, len_c]``.

    Returns:
        torch.Tensor: Delay-line input of shape
            ``[len_c, num_ref, num_sec, num_err, len_c]``.
    """
    num_ref, num_sec, num_err, _ = Fx.shape
    Fx_extend = torch.zeros((len_c, num_ref, num_sec, num_err, len_c),
                            dtype=torch.float32)
    Fx_delay = torch.zeros((num_ref, num_sec, num_err, len_c),
                           dtype=torch.float32)
    for i in range(len_c):
        Fx_delay = torch.roll(Fx_delay, shifts=1, dims=3)
        Fx_delay[:, :, :, 0] = Fx[:, :, :, i]
        Fx_extend[i] = Fx_delay
    return Fx_extend


class ModifiedMAML(nn.Module):
    """Modified MAML meta-learner for a multi-channel ANC system."""

    def __init__(self, num_ref: int, num_sec: int, len_c: int,
                 lr: float, gamma: float, device: str = "cpu"):
        super().__init__()
        self.initial_weights = nn.Parameter(
            torch.zeros((num_ref, num_sec, len_c), dtype=torch.float32)
        )
        self.len_c = len_c
        self.lr = lr
        self.device = device
        self.gamma = gamma
        self.gamma_vector = self._construct_gamma_vector()

    def _construct_gamma_vector(self) -> torch.Tensor:
        """Create a vector of forgetting factors."""
        gam_vector = torch.zeros(self.len_c, dtype=torch.float32)
        for i in range(self.len_c):
            gam_vector[i] = self.gamma ** (self.len_c - 1 - i)
        return gam_vector.to(self.device)

    def first_grad(self, initial_weights: torch.Tensor,
                   Fx: torch.Tensor, Dis: torch.Tensor) -> torch.Tensor:
        """Compute the gradient from the first-step adaptation."""
        weights_a = Variable(initial_weights.detach(), requires_grad=True).to(self.device)
        anti_noise_ele = torch.einsum('rsen,rsn->rse', Fx, weights_a)
        anti_noise = torch.einsum('rse->e', anti_noise_ele)
        error = Dis[:, -1] - anti_noise
        loss_1 = torch.einsum('i,i->', error, error)
        loss_1.backward()
        return weights_a.grad.detach()

    def adaptive_filtering(self, weights: torch.Tensor,
                           Fx: torch.Tensor) -> torch.Tensor:
        """Generate anti-noise for the entire sequence."""
        anti_noise_ele = torch.einsum('...rsen,...rsn->...rse', Fx, weights)
        anti_noise = torch.einsum('...rse->...e', anti_noise_ele)
        return anti_noise

    def forward(self, Fx: torch.Tensor, Dis: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run one modified MAML iteration.

        This method now constructs the delay-line representation internally
        using :func:`build_delay_line`, ensuring the meta-learner always uses
        the helper utility.

        Args:
            Fx (torch.Tensor): Filtered references ``[R, S, E, Len]``.
            Dis (torch.Tensor): Disturbance matrix ``[E, Len]``.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Anti-noise matrix ``[T, E]`` and
            the forgetting factor vector ``[T]`` where ``T == len_c``.
        """
        Fx_extend = build_delay_line(self.len_c, Fx)
        weights_grad = self.first_grad(self.initial_weights, Fx, Dis)
        control_weights = self.initial_weights - 0.5 * self.lr * weights_grad
        anti_noise_matrix = self.adaptive_filtering(control_weights, Fx_extend)
        return anti_noise_matrix, self.gamma_vector


def loss_function_maml(anti_noise_matrix: torch.Tensor,
                        Dis: torch.Tensor,
                        gamma_vector: torch.Tensor) -> torch.Tensor:
    """Squared error loss with exponential forgetting for modified MAML.

    Args:
        anti_noise_matrix (torch.Tensor): Anti-noise ``[T, E]``.
        Dis (torch.Tensor): Disturbance ``[E, T]``.
        gamma_vector (torch.Tensor): Forgetting factors ``[T]``.

    Returns:
        torch.Tensor: Weighted squared-error loss.
    """
    error_vector = Dis - torch.transpose(anti_noise_matrix, 1, 0)
    loss = torch.einsum('t,t->',
                        torch.einsum('st,st->t', error_vector, error_vector),
                        gamma_vector)
    return loss


if __name__ == "__main__":
    model = ModifiedMAML(num_ref=4, num_sec=4, len_c=512, lr=1e-5, gamma=0.9)
    Fx = torch.randn(4, 4, 1, 512)
    Dis = torch.randn(1, 512)
    anti_noise, gamma_vec = model(Fx, Dis)
    print("anti-noise shape:", anti_noise.shape)
    print("gamma vector shape:", gamma_vec.shape)
