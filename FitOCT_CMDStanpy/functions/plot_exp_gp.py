import numpy as np
import matplotlib.pyplot as plt
import traceback
from scipy import stats
from cmdstanpy import CmdStanModel
from .exp_decay_model import exp_decay_model


def plot_exp_gp(x, y, uy, y_smooth, out, g_pars, mod_scale=0.3, nMC=100, data_type=2, br=None, save_path=None):
    """
    Plot outputs from ExpGP fitting.
    
    Parameters:
        x: array-like, input vector
        y: array-like, responses/data vector
        uy: array-like, uncertainties vector
        y_smooth: array-like, smoothed data vector
        out: dict, output from fitExpGP
        g_pars: dict, graphical parameters and colors
        mod_scale: float, plotting scale for yGP
        nMC: int, number of spaghetti lines
        data_type: int, type of data (1=amplitude, 2=intensity)
        br: dict, output from printBr()
        save_path: str, path to save plots
    """
    plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11
})

    try:
        # Extract graphical parameters
        cols = g_pars['cols']
        col_tr = g_pars['col_tr']
        col_tr2 = g_pars['col_tr2']
        plot_title = g_pars['plot_title']
        xlabel = g_pars['xlabel']
        ylabel = "mean amplitude (a.u.)" if data_type == 1 else "mean intensity (a.u.)"
        
        # Extract fit parameters
        fit = out['fit']
        method = out['method']
        x_gp = out['xGP']
        prior_pd = out['prior_PD']
        
        if method == 'sample':
            # Get MCMC samples
            theta = fit.stan_variable('theta')
            y_gp = fit.stan_variable('yGP')
            sigma = np.mean(fit.stan_variable('sigma'))
            
            if prior_pd == 0:
                # Get posterior samples
                resid = fit.stan_variable('resid')
                mod = fit.stan_variable('m')
                dL = fit.stan_variable('dL')
                lp = fit.draws_pd('lp__')
                map_idx = np.argmax(lp)
                y_map = mod[map_idx, :]
                
                # Select random iterations
                n_iterations = mod.shape[0]
                iMC = np.random.choice(n_iterations, min(nMC, n_iterations), replace=False)
                
                # Create figure with 2x2 subplots if prior_pd=0, else 1x2
                fig = plt.figure(figsize=(20, 10 if prior_pd else 20))
                
                # Data vs Model plot
                ax1 = plt.subplot(2 if prior_pd == 0 else 1, 2, 1)
                plt.plot(x, y, 'o', color=cols[5], label='data', markersize=3)
                
                # Plot posterior samples
                if nMC > 0:
                    for idx in iMC:
                        plt.plot(x, mod[idx], color=col_tr[3], alpha=0.1)
                
                # Calculate and plot mean exponential decay
                m_exp = np.zeros_like(x)
                for i in range(theta.shape[0]):
                    m_exp += exp_decay_model(x, theta[i, :3], data_type)
                m_exp /= theta.shape[0]
                plt.plot(x, m_exp, color=cols[6], label='mean exp. fit')
                
                plt.title(plot_title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.grid(True)
                
                # Add two legends: one for data and fits, one for br values
                legend1 = plt.legend(['data', 'mean exp. fit', 'post. sample'], 
                                   loc='upper right', frameon=False)
                if br is not None:
                    legend2 = plt.legend([f"br = {br['br']:.3f}",
                                        f"CI95 = {br['CI95'][0]:.2f}-{br['CI95'][1]:.2f}"],
                                       loc='upper right', frameon=False,
                                       bbox_to_anchor=(1, 0.85))
                    plt.gca().add_artist(legend1)
                
                # Residuals plot
                if prior_pd == 0:
                    ax2 = plt.subplot(2, 2, 2)
                    res = np.mean(resid, axis=0)
                    plt.plot(x, res, 'o', color=cols[5], markersize=3)
                    plt.fill_between(x, -2*uy, 2*uy, color=col_tr2[3], alpha=0.5)
                    plt.plot(x, y_smooth - y_map, color=cols[6], label='smooth - fit')
                    plt.grid(True)
                    plt.title('Residuals')
                    plt.xlabel(xlabel)
                    plt.ylabel('residuals (a.u.)')
                    plt.legend(['mean resid.', 'data 95% uncert.', 'smooth - fit'],
                             loc='upper right', frameon=False)
                
                # Local deviations plot
                ax3 = plt.subplot(2, 2, 3)
                # Plot multiple deviation curves for uncertainty visualization
                if nMC > 0:
                    for idx in iMC:
                        plt.plot(x, dL[idx, :], color=col_tr[3], alpha=0.1)

                # Plot mean deviation
                mean_dL = np.mean(dL, axis=0)
                plt.plot(x, mean_dL, color=cols[6], label='mean deviation', linewidth=2)

                # Plot MAP estimate
                plt.plot(x, dL[map_idx, :], color=cols[5], label='MAP estimate', linewidth=1)

                plt.axhline(0, color='black', lw=1, linestyle='--')
                plt.grid(True)
                plt.ylim(mod_scale * np.array([-1, 1]))
                plt.title('Deviation from mean Ls')
                plt.xlabel(xlabel)
                plt.ylabel('relative deviation')
                plt.legend(frameon=False, loc='upper right')
                
                if prior_pd == 0:
                    ax4 = plt.subplot(2, 2, 4)
                    # Get parameter statistics
                    theta_mean = np.mean(theta, axis=0)
                    theta_std = np.std(theta, axis=0)
                    
                    # Create table data
                    param_names = ['$C$ (a.u.)', '$A_0$', '$L_s$ (µm)']
                    table_data = [
                        [f"{theta_mean[i]:.3g}", f"{theta_std[i]:.2g}"] 
                        for i in range(3)
                    ]
                    
                    # Create table
                    table = plt.table(
                        cellText=table_data,
                        rowLabels=param_names,
                        colLabels=['mean', 'std'],
                        loc='center',
                        cellLoc='center',
                        bbox=[0.1, 0.1, 0.8, 0.8]  # [left, bottom, width, height]
                    )
                    
                    # Customize table appearance
                    table.auto_set_font_size(False)
                    table.set_fontsize(10)
                    for key, cell in table._cells.items():
                        cell.set_edgecolor('none')
                        if key[0] == 0:  # Header row
                            cell.set_text_props(weight='bold')
                            cell.set_facecolor('#f0f0f0')
                    
                    plt.axis('off')
                    plt.title('Parameter Statistics', pad=20)

                    # Save main figure if path provided
                    if save_path:
                        plt.savefig(f"{save_path}_results.png", dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()

                    # Create separate Q-Q plot
                    plt.figure(figsize=(8, 8))
                    uy_expanded = np.tile(uy, (resid.shape[0], 1))
                    resw = (resid / (uy_expanded * sigma)).ravel()
                    stats.probplot(resw, dist="norm", plot=plt)
                    plt.grid(True)
                    plt.title('Norm. Q-Q plot of weighted residuals')
                    
                    # Add reference line
                    ax = plt.gca()
                    xlim = ax.get_xlim()
                    ylim = ax.get_ylim()
                    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
                    plt.plot(lim, lim, 'k--', alpha=0.5)
                    plt.xlim(lim)
                    plt.ylim(lim)
                    
                    if save_path:
                        plt.savefig(f"{save_path}_qq_plot.png", dpi=300, bbox_inches='tight')
                    plt.show()
                    plt.close()
                
        else:
            # Handle optimization method case similar to R code
            pass
            
    except Exception as e:
        print("\nERROR in plot_exp_gp:")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {str(e)}")
        traceback.print_exc()
        raise

    return {'sum': None}  # Return structure similar to R version