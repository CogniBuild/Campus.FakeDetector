import numba
import numpy as np


@numba.njit(parallel=True, fastmath=True)
def rgb2gray(rgb: np.array) -> np.array:
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray
