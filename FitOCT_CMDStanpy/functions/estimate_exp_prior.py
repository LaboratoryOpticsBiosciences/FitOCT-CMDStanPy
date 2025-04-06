import os
import numpy as np
from cmdstanpy import CmdStanModel



def estimate_exp_prior(x, uy, data_type=2, prior_type='mono', out=None,
                       ru_theta=0.05, eps=1e-3, nb_chains=4, nb_warmup=800,
                       nb_iter=800+200,CMDStan_path=None):
    """
    Define/estimate normal multivariate prior pdf for exponential decay parameters.

    Args:
        x (numpy.ndarray): Numeric vector of depths.
        uy (numpy.ndarray): Numeric vector of uncertainties.
        data_type (int): Integer defining the type of data (1: amplitude or 2: intensity).
        prior_type (str): Type of prior ('mono' or 'abc').
        out (dict): Output dictionary from fitMonoExp.
        ru_theta (float): Relative uncertainty on parameters (priorType='mono').
        eps (float): Tolerance for moments matching method (priorType='abc').
        nb_chains (int): Number of MCMC chains (priorType='abc').
        nb_warmup (int): Number of warmup steps (priorType='abc').
        nb_iter (int): Number of iterations (priorType='abc').

    Returns:
        dict: Dictionary with the prior parameters.
    """
    
    # Function to calculate statistics for residuals
    def stats_obs(resid):
        return [np.quantile(np.abs(resid), 0.95), 0.]
    
    # Initialize theta0 from the output (fitMonoExp result)
    theta0 = out['best_theta']
    
    # For 'mono' prior type
    if prior_type == 'mono':
        cor_theta = out['cor_theta']
        u_theta = ru_theta * theta0
        r_list = None
    
    # For 'abc' prior type
    else:
        resid = out['fit']['par']['resid']
        Sobs = stats_obs(resid / uy)
        
        stan_data = {
            'N': len(x),
            'x': x,
            'uy': uy,
            'dataType': data_type,
            'Sobs': Sobs,
            'Np': 3,
            'theta': theta0,
            'eps': eps
        }
        
        init = {'u_theta': ru_theta * theta0}
        
        # Compile Stan model
        model = CmdStanModel(stan_file=os.path.join(CMDStan_path,'src/stan_files/modUQExp.stan'))  # Use the actual path to the Stan model file
        
        # Sample from the model
        fit = model.sample(
            data=stan_data,
            init=init,
            chains=nb_chains,
            warmup=nb_warmup,
            iter=nb_iter,
            adapt_delta=0.995
        )
        
        # Get MAP estimate of u_theta
        lp = fit.stan_variable('lp__')
        map_idx = np.argmax(lp)
        u_theta = fit.stan_variable('u_theta')[map_idx, :]
        cor_theta = np.eye(3)  # Hypothetical no correlation assumption
        r_list = {
            'Sobs': Sobs,
            'Ssim': fit.stan_variable('Ssim')[map_idx, :]
        }
    
    # Covariance matrix
    Sigma0 = np.outer(u_theta, u_theta) * cor_theta
    
    return {
        'theta0': theta0,
        'Sigma0': Sigma0,
        'stats': r_list
    }