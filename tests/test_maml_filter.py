import numpy as np

from models.maml_filter import MAMLFilter


def test_maml_initial_gradient_and_error():
    Fx = np.array([[1.0], [2.0]])
    Di = np.array([[0.5], [-1.5]])
    filt = MAMLFilter(filter_len=2, num_refs=1)
    mu, lamda, eps = 0.1, 0.9, 0.5
    err = filt.maml_initial(Fx, Di, mu, lamda, eps)

    # Expected calculations
    f1, f2 = 1.0, 2.0
    d1, d2 = 0.5, -1.5
    e1 = d2
    Wo = mu * e1 * np.array([[f2], [f1]])
    e_j0 = d2 - float(np.dot(Wo.T, np.array([[f2], [f1]])))
    e_j1 = d1 - float(np.dot(Wo.T, np.array([[f1], [0.0]])))
    grad = (
        eps * (mu / 2) * e_j0 * np.array([[f2], [f1]]) +
        eps * (mu / 2) * lamda * e_j1 * np.array([[f1], [0.0]])
    )
    expected_Phi = grad
    expected_err = e_j0

    assert np.allclose(filt.Phi, expected_Phi)
    assert np.isclose(err, expected_err)
