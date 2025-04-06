import os
import numpy as np
import logging
from cmdstanpy import CmdStanModel


def fit_mono_exp(x, y, uy, method='sample', data_type=2,
                 nb_chains=4, nb_warmup=500, nb_iter=1000, 
                 CMDStan_path=None):
    """
    Monoexponential fit of OCT decay using Bayesian inference.
    
    Parameters:
        x (array-like): Independent variable.
        y (array-like): Dependent variable (responses).
        uy (array-like): Uncertainty on 'y'.
        method (str): Optimization method ('optim' or 'sample').
        data_type (int): Type of data (1 or 2).
        nb_chains (int): Number of chains for sampling.
        nb_warmup (int): Number of warmup iterations.
        nb_iter (int): Total number of iterations (including warmup).
        stan_model_path (str): Path to the Stan model file.

    Returns:
        dict: A dictionary containing the fit results:
            - 'fit': The CmdStanPy fit object.
            - 'best_theta': Optimal parameters (MAP estimate or posterior mean).
            - 'cor_theta': Correlation matrix of parameters.
            - 'unc_theta': Uncertainty of parameters (standard deviation).
            - 'method': Optimization method used.
            - 'data': Stan data used for fitting.
    """

    stan_model_path=os.path.join(CMDStan_path, "src/stan_files/modFitExp.stan")
    # Stan data
    stan_data = {
        'N': len(x),
        'x': x,
        'y': y,
        'uy': uy,
        'Np': 3,
        'dataType': data_type
    }

    init_values = {
    'theta': [min(y), max(y) - min(y), np.mean(x)]
}

    # Load Stan model
    model = CmdStanModel(stan_file=stan_model_path)

    if method == 'sample':
        # Sampling method
        fit = model.sample(
            data=stan_data,
            chains=nb_chains,
            iter_warmup=nb_warmup,
            iter_sampling=nb_iter - nb_warmup,
            inits=init_values,
            adapt_delta=0.99,
            max_treedepth=12
        )

        # Extract parameters
        theta_samples = fit.draws_pd(['theta'])
        best_theta = theta_samples.mean(axis=0).to_numpy()
        unc_theta = theta_samples.std(axis=0).to_numpy()
        cor_theta = theta_samples.corr().to_numpy()

    elif method == 'optim':
        try:
            # Optimization method
            fit = model.optimize(data=stan_data,inits=init_values, jacobian=True)
            # Extract parameters
            best_theta = fit.optimized_params_dict['theta']
            hessian = np.array(fit.optimized_hessian['theta'])
            cov_matrix = np.solve(-hessian, np.eye(len(hessian)))
            unc_theta = np.sqrt(np.diag(cov_matrix))
            cor_theta = cov_matrix / np.outer(unc_theta, unc_theta)

        except RuntimeError as e:
            logging.error(f"Optimization failed: {e}")
            # Optionally, you could store the error or index of the failed run
            return None  # Or any other value indicating failure





    else:
        raise ValueError("Invalid method. Choose 'optim' or 'sample'.")

    return {
        'fit': fit,
        'best_theta': best_theta,
        'cor_theta': cor_theta,
        'unc_theta': unc_theta,
        'method': method,
        'data': stan_data
    }
