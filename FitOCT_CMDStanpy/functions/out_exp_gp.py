import numpy as np
import pandas as pd
from scipy import stats  # if you're using this elsewhere (e.g., Q-Q plots)
from cmdstanpy import CmdStanMCMC, CmdStanModel
from .exp_decay_model import exp_decay_model

def out_exp_gp(x, y, uy, out, data_type):
    """
    Process and extract results from a GP model output.

    Parameters:
    - x: array-like, independent variable.
    - y: array-like, dependent variable.
    - uy: array-like, uncertainty in y.
    - out: dict, output of the GP fitting process.
    - data_type: str, type of the data.

    Returns:
    - A dictionary containing summary statistics, exponential decay model results, mode, and derivatives.
    """
    fit = out['fit']
    method = out['method']
    prior_PD = out['prior_PD']


    if prior_PD != 0:
        return None

    if method == 'sample':
        theta = fit.stan_variable('theta')
        yGP = np.mean(fit.stan_variable('yGP'),axis=0)
        sigma = np.mean((fit.stan_variable('sigma')),axis=0)
        resid = np.mean(fit.stan_variable('resid'),axis=0)
        mod = fit.stan_variable('m')
        dL = fit.stan_variable('dL')
        lp = fit.draws_pd('lp__')

        map_idx = np.argmax(lp)
        mod0 = mod[map_idx, :]
        mExp = exp_decay_model(x, theta[map_idx, :3], data_type)
        dL0 = dL[map_idx, :]

        sum_stats = fit.summary()['Mean'][1:4]
        sum_errs = fit.summary()['StdDev'][1:4]
        sum_df = np.stack([sum_stats, sum_errs], axis=1)

    else:
        theta = fit.stan_variable('theta')
        mod = fit.stan_variable('m')
        resid = fit.stan_variable('resid')
        dL = fit.stan_variable('dL')
        yGP = fit.stan_variable('yGP')
        sigma = fit.stan_variable('sigma')

        mExp = exp_decay_model(x, theta, data_type)
        mod0 = mod
        dL0 = dL

        # Extract parameters and calculate standard errors
        opt = {par: fit['par'][par] for par in ['theta']}
        opt = {k: np.array(v).flatten() for k, v in opt.items()}
        opt_flat = {k: v for d in opt.values() for k, v in zip(opt.keys(), d)}

        se = {k: np.nan for k in opt_flat}
        # if 'hessian' in fit and fit['hessian'] is not None:
        #     H = fit['hessian']
        #     tags = [tag.replace('.', '') for tag in H.columns]
        #     H.columns = H.index = tags
        #     for par in opt_flat:
        #         se[par] = np.sqrt(-1 / H.loc[par, par])

        sum_df = {"mean": opt_flat, "sd": se}

    return {
        'sum': sum_df,
        'mExp': mExp,
        'mod': mod0,
        'dL': dL0
    }