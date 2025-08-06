import numpy as np

from algorithms.fxlms import multi_ref_multi_chan_fxlms


def test_multi_ref_multi_chan_fxlms_converges():
    np.random.seed(0)
    length = 200
    Ref = np.random.randn(length, 1)
    E = -Ref.copy()
    sec_path = np.array([[1.0]])
    W, e = multi_ref_multi_chan_fxlms(Ref, E, filter_len=1, sec_path=sec_path, stepsize=0.5)
    assert W.shape == (1, 1)
    assert e.shape == (length, 1)
    assert np.mean(np.abs(e[-50:])) < 0.1
