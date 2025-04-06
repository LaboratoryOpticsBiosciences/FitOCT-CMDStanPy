import numpy as np
import pandas as pd  
from cmdstanpy import CmdStanModel
def n_act_ctrl_pts(fit, prob=0.90):
    """
    Count active control points in fitExpGP output.
    
    Args:
        fit (CmdStanFit or dict): An object issued from a Stan code, either
                                   from CmdStanPy fit or a dictionary output.
        prob (float): A probability threshold. A control point is considered 
                      active if its value is non-zero at the given probability level.
    
    Returns:
        int: The number of active points, or None if no valid data is available.
    """
    
    # If the fit is from CmdStanPy's fit (stanfit object)
    if fit['method']=='sample':
        y_gp = fit['fit'].stan_variable('yGP')
    else:
        # If it's a dictionary-like object, extract yGP from theta_tilde
        if 'theta_tilde' in fit:
            S = fit['theta_tilde']
            y_gp = S[:, [i for i, col in enumerate(S.columns) if 'yGP' in col]]
        else:
            return None
    
    # Calculate the p% confidence interval (CI)
    lower_percentile = 0.5 - 0.5 * prob
    upper_percentile = 0.5 + 0.5 * prob
    
    Q = np.percentile(y_gp, [lower_percentile * 100, upper_percentile * 100], axis=0).T
    
    # Count how many control points are non-zero within the p% CI
    active_points = np.sum(np.prod(Q > 0, axis=1))
    
    return active_points