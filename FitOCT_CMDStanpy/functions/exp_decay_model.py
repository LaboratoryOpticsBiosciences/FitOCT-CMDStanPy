import numpy as np

def exp_decay_model(x, c, data_type=2):
    """
    Reference OCT decay model.

    Parameters:
        x (array-like): Vector of depths.
        c (array-like): Vector of parameters [c1, c2, c3].
        data_type (int): Type of data (1 or 2). Defaults to 2.

    Returns:
        np.ndarray: Model values as a numeric vector.
    """
    return c[0] + c[1] * np.exp(-data_type * x / c[2])
