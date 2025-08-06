import numpy as np
import scipy.io as sio

from dataloader.generate_data import generate_anc_training_data, generate_task_batch


def test_generate_anc_training_data_shapes_values(tmp_path):
    np.random.seed(0)
    path_dir = tmp_path / 'paths'
    path_dir.mkdir()
    # Create two simple primary path files
    G = np.zeros((10, 4))
    G[:, 0] = np.array([1, 0.5, 0, 0, 0, 0, 0, 0, 0, 0])
    G[:, 2] = np.array([0.3, 0.2, 0, 0, 0, 0, 0, 0, 0, 0])
    for i in range(2):
        sio.savemat(path_dir / f'G_{i}.mat', {'G_matrix': G})

    # Secondary path file
    sec_path = np.array([[1.0], [0.5], [0.2]])
    sio.savemat(tmp_path / 'sec.mat', {'S': sec_path})

    Fx, Di = generate_anc_training_data(
        str(path_dir),
        ['G_0.mat', 'G_1.mat'],
        str(tmp_path / 'sec.mat'),
        N_epcho=2,
        Len_N=8,
        fs=16000,
    )

    assert Fx.shape == (8, 2)
    assert Di.shape == (8, 2)
    assert np.isfinite(Fx).all() and np.isfinite(Di).all()
    assert not np.allclose(Fx, 0) and not np.allclose(Di, 0)


def test_generate_task_batch_shapes_values():
    np.random.seed(1)
    Ref, Di = generate_task_batch(length=10, num_refs=2, num_errs=3)
    assert Ref.shape == (10, 2)
    assert Di.shape == (10, 3)
    assert np.isfinite(Ref).all() and np.isfinite(Di).all()
    Ref2, Di2, sec = generate_task_batch(length=10, num_refs=2, num_errs=3, with_secondary=True, sec_len=4)
    assert Ref2.shape == (10, 2)
    assert Di2.shape == (10, 3)
    assert sec.shape == (4, 2 * 3)
    assert np.isfinite(sec).all()
