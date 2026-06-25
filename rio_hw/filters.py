import numbers

import numpy as np


class LowPassFilter:
    def __init__(self, alpha: np.ndarray | numbers.Number, initial=0.0):
        assert ((0 < np.array(alpha)) & (np.array(alpha) < 1)).all()
        self.alpha = alpha
        self.s = np.asarray(initial, dtype=np.float64)

    @staticmethod
    def filter(x, s_prev, alpha):
        return alpha * x + (1 - alpha) * s_prev

    def __call__(self, x: np.ndarray | numbers.Number):
        x_arr = np.asarray(x, dtype=np.float64)
        self.s = self.filter(x_arr, self.s, self.alpha)
        if x_arr.shape != ():
            return self.s.copy()
        return self.s
