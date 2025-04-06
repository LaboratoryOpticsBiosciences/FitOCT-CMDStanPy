import numpy as np
from scipy import stats
from .n_act_ctrl_pts import n_act_ctrl_pts  # adjust path accordingly
from cmdstanpy import CmdStanModel

def printBr(fit,model, prob=0.90, silent=False):
    """
    Print Birge's ratio and confidence interval.

    Parameters:
    - fit: CmdStanPy CmdStanFit object
    - prob: Probability threshold for active control points
    - silent: If False, print the results; otherwise, suppress output.

    Returns:
    - A dictionary containing the Birge's ratio, degrees of freedom, CI95, and alert message (if any).
    """
    # Extract model and method from the fit object
    method=fit['method']

    # Birge's ratio and number of residuals
    if method == 'optim':
        br = fit['fit'].stan_variable('br')
        N = len(fit['fit'].stan_variable('resid')[0])
    else:
        br = np.mean(fit['fit'].stan_variable('br'))
        N = len(fit['fit'].stan_variable('resid')[0])  # Assuming 'resid' is available
    
    if br is None or N is None:
        return None

    # Number of parameters (Np) and active control points (Nn)
    if model == 'modFitExpGP':
        Np = 5
        if method == 'optim':
            Nn0 = len(fit['par']['yGP'])
        else:
            Nn0 = len(np.mean(fit['fit'].stan_variable('yGP'),axis=0))
        # Function to calculate active control points (you'd need to define this function)
        nAct = n_act_ctrl_pts(fit, prob)
        if nAct is None:
            Nn = Nn0
        else:
            Nn = nAct
    else:
        Np = 3
        Nn0 = 0
        Nn = 0
        nAct = None

    # Degrees of freedom for residuals
    ndf0 = N - (Np + Nn0)  # As computed in Stan model
    ndf = N - (Np + Nn)

    # Adjust br for the correct degrees of freedom
    br = br * ndf0 / ndf

    # Confidence interval on br
    CI95 = [stats.chi2.ppf(0.025, df=ndf) / ndf, stats.chi2.ppf(0.975, df=ndf) / ndf]


    alert = None
    # if not silent:
    #     if nAct is not None:
    #         print(f'Active pts.: {Nn} / {Nn0}')
    #     print(f'ndf        : {ndf}')
    #     print(f'br         : {br:.2f}')
    #     print(f'CI95(br)   : {CI95[0]:.2f} - {CI95[1]:.2f}')

    if CI95[0] >= br or br >= CI95[1]:
        alert = '!!! WARNING: br out of interval !!!'
    else:
        alert = 'Fit OK'
    

    if model == 'modFitExpGP' and nAct is None:
        # Let the user decide by himself
        for n in range(Nn0, -1, -1):
            ndf1 = N - (Np + n)
            CI951 =[stats.chi2.ppf(0.025, df=ndf1) / ndf1, stats.chi2.ppf(0.975, df=ndf1) / ndf1]
            br1 = br * ndf0 / ndf1
            if CI951[0] <= br1 <= CI951[1]:
                break
        alert += f'\n--> OK if there are less than\n{n + 1} active ctrl points'

    # if not silent:
    #     print(alert)

    return {
        'br': br,
        'ndf': ndf,
        'CI95': CI95,
        'alert': alert
    }