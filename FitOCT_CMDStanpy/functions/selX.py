import numpy as np
from cmdstanpy import CmdStanModel
def selX(x, y, depthSel=None, subSample=1):
    """
    Subset OCT signal.
    
    Parameters:
    - x: A numpy array of the x-values.
    - y: A numpy array of the y-values (responses).
    - depthSel: A tuple (xmin, xmax) specifying the range of x-values to select.
    - subSample: A numeric factor for regular subsampling (default is 1, meaning no subsampling).
    
    Returns:
    - A dictionary containing the new vectors 'x' and 'y'.
    """
    # Convert x and y to numpy arrays for easier indexing
    x = np.array(x)
    y = np.array(y)
    
    # Apply depth selection if specified
    if depthSel is not None:
        xSel = np.where((x >= depthSel[0]) & (x <= depthSel[1]))[0]
    else:
        xSel = np.arange(len(x))  # Select all indices if no depthSel
    
    x = x[xSel]
    y = y[xSel]
    
    # Apply subsampling if needed
    if subSample != 1:
        xSel = np.arange(0, len(x), subSample)
        x = x[xSel]
        y = y[xSel]
    
    return {'x': x, 'y': y}
