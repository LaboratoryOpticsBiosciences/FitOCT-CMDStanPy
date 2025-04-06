
import numpy as np
import os
from cmdstanpy import CmdStanModel


def fit_exp_gp(x, y, uy,
               data_type=2, Nn=10, theta0=None, Sigma0=None,
               lambda_rate=0.1, lasso=False, method='optim', iter=50000,
               prior_PD=0, alpha_scale=0.1, rho_scale=1/10, grid_type='internal',
               nb_chains=4, nb_warmup=500, nb_iter=1000,
               verbose=False, open_progress=True,Cmdstan_path=None):
    """
    Decay fit with modulation of mean depth by Gaussian Process.
    
    Args:
        x (array): Numeric vector of x-values (depths or independent variable).
        y (array): Numeric vector of y-values (responses).
        uy (array): Numeric vector of uncertainties on y.
        data_type (int): Defines the type of data (1 for amplitude, 2 for intensity).
        Nn (int): Number of control points for the Gaussian Process.
        theta0 (array): Prior mean values for theta (optional).
        Sigma0 (array): Prior covariance matrix for theta (optional).
        lambda_rate (float): Scale of control points prior.
        lasso (bool): Flag to use a lasso prior.
        method (str): 'sample' for sampling (MCMC) or 'optim' for optimization.
        iter (int): Maximum number of iterations for the optimizer (if using 'optim').
        prior_PD (bool): Flag to sample from the prior PDF only.
        alpha_scale (float): Standard deviation scale of the Gaussian Process.
        rho_scale (float): Relative correlation length of the GP.
        grid_type (str): Type of control points grid ('internal' or other).
        nb_chains (int): Number of MCMC chains.
        nb_warmup (int): Number of warmup steps for MCMC.
        nb_iter (int): Number of sampling iterations.
        verbose (bool): Whether to print verbose output.
        open_progress (bool): Whether to display the progress bar.

    Returns:
        dict: A dictionary containing the fit results, method, control points, and other details.
    """

    # Grid of GP control points
    if grid_type == 'internal':
        dx = np.diff(np.array([min(x), max(x)]))[0] / (Nn + 1)
        xGP = np.linspace(min(x) + dx / 2, max(x) - dx / 2, Nn)
    else:
        xGP = np.linspace(min(x), max(x), Nn)

    # Initial monoexp parameters
    if theta0 is None:
        theta0 = [min(y), max(y) - min(y), np.mean(x)]

    if Sigma0 is None:
        Sigma0 = np.diag((np.array(theta0) * 0.05) ** 2)  # 5% uncertainty, no correlation

    stan_data = {
        'N': len(x),
        'x': x.tolist(), 'y': y.tolist(), 'uy': uy.tolist(),
        'dataType': data_type,
        'Np': 3,  # Number of parameters for the exponential model
        'Nn': Nn,  # Number of control points
        'xGP': xGP.tolist(),
        'alpha_scale': alpha_scale,
        'rho_scale': rho_scale,
        'theta0': theta0,
        'Sigma0': Sigma0,
        'prior_PD': prior_PD,
        'lambda_rate': lambda_rate
    }

    init = {
        'theta': theta0,
        'yGP': 0.01 * np.random.randn(Nn),
        'sigma': 1.0
    }

    # Parameters to scatterplot
    parP = ['theta', 'sigma']
    if not lasso:
        parP.append('lambda')

    # Parameters to report
    par_opt = parP + ['yGP']

    # Parameters to save for plots
    pars = par_opt
    if prior_PD == 0:
        pars += ['resid', 'br', 'm', 'dL']

    # Load the Stan model (adjust path to where the Stan model file is located)
    model_dir=os.path.join(Cmdstan_path, "src/stan_files/modFitExpGP.stan")
    model = CmdStanModel(stan_file=model_dir,force_compile=True)

    # Run Stan model fitting based on chosen method
    if method == 'sample':
        fit = model.sample(
            data=stan_data,
            inits=init,
            iter_sampling=nb_iter,
            iter_warmup=nb_warmup,
            chains=nb_chains
            #verbose=verbose,
            #progress=open_progress
        )
    else:  # method == 'optim'
        fit = model.optimize(
            data=stan_data,
            inits=init,
            algorithm='LBFGS',
            #hessian=True,
            iter=nb_iter,
            refresh=500,
            #verbose=verbose
        )
        

    # Return the results
    return {
        'fit': fit,
        'method': method,
        'xGP': xGP,
        'prior_PD': prior_PD,
        'lasso': lasso,
        'data': stan_data
    }