import os
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
from cmdstanpy import CmdStanModel


def estimate_noise(x, y, df=15, max_rate=10000,CMDStan_path=None):
    """
    Estimate and model noise in signal.
    
    Parameters:
        x (array-like): Independent variable.
        y (array-like): Dependent variable (responses).
        df (int): Smoothing factor for splines.
        max_rate (float): Max value of the rate parameter.
    
    Returns:
        dict: A dictionary with the following keys:
            - 'fit': CmdStan fit object.
            - 'theta': Optimal parameters.
            - 'uy': Estimated uncertainty values for `y`.
            - 'y_smooth': Smoothed values of `y`.
            - 'method': Optimization method used ('optim').
            - 'data': Data passed to the Stan model.
    """
    # Smoothing using splines
    knots = np.linspace(np.min(x), np.max(x), num=df - 1)[1:-1]  # exclude endpoints
    # Fit with LSQUnivariateSpline to approximate degrees of freedom
    spline = LSQUnivariateSpline(x, y, t=knots)
    y_smooth = spline(x)
    res_spl = y - y_smooth

    # Stan model data
    stan_data = {
        'N': len(x),
        'x': x.tolist(),
        'y': res_spl.tolist(),
        'maxRate': max_rate
    }
    init = {'theta': [np.max(res_spl), np.mean(x)]}

    # Load Stan model (assumes modHetero.stan exists in the same directory)
    model = CmdStanModel(stan_file=os.path.join(CMDStan_path,'src/stan_files/modHetero.stan'))
    fit = model.optimize(data=stan_data, inits=init)

    # Extract parameters and compute uncertainties
    theta = fit.stan_variable('theta')  # Optimal parameters
    uy = theta[0] * np.exp(-x / theta[1])  # Estimated uncertainties

    # Return results
    return {
        'fit': fit,
        'theta': theta,
        'uy': uy,
        'y_smooth': y_smooth,
        'method': 'optim',
        'data': stan_data
    }