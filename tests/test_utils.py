from pathflowai import utils
import numpy as np
from os import path


def test_svs2dask_array():
    tests_dir = path.dirname(path.realpath(__file__))
    ground_truth = np.load(path.join(tests_dir, 'sample.npy'))
    test = utils.svs2dask_array(path.join(tests_dir, 'sample.tif')).compute()
    assert np.allclose(ground_truth, test)
    # assert np.array_equal(ground_truth, test)
