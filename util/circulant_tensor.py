from numpy.lib.stride_tricks import as_strided
import numpy as np


def circulant_tensor(c):
    c = c[..., np.newaxis]
    return np.concatenate([np.roll(c, i, axis=0) for i in range(c.shape[0])], axis=2)
