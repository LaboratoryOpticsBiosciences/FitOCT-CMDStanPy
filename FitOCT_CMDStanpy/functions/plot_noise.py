import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

def plot_noise(x, y, uy, y_smooth, g_pars, save_path,data_type=2):
    """
    Plot outputs from the estimate_noise function.
    
    Parameters:
        x (array-like): Independent variable.
        y (array-like): Dependent variable (responses).
        uy (array-like): Estimated uncertainties.
        y_smooth (array-like): Smoothed values of y.
        g_pars (dict): Graphical parameters and colors.
        data_type (int): Type of data (1 or 2).
    
    Returns:
        dict: A dictionary containing:
            - 'SNR': Signal-to-noise ratio in dB.
    """
    # Extract graphical parameters
    cols = g_pars.get('cols', ['blue', 'red', 'green', 'cyan', 'magenta', 'orange', 'purple'])
    col_tr2 = g_pars.get('col_tr2', ['lightblue', 'lightcoral'])
    plot_title = g_pars.get('plot_title', 'Signal and Noise')
    xlabel = g_pars.get('xlabel', 'x-axis')
    ylabel = g_pars.get('ylabel', 'y-axis')
    cex_leg = g_pars.get('cex_leg', 1.0)

    # Signal-to-noise ratio (SNR)
    if data_type == 1:
        SNR = 20 * np.log10(np.mean(y) / np.mean(np.abs(y - y_smooth)))
        ylabel = "mean amplitude (a.u.)"
    elif data_type == 2:
        SNR = 10 * np.log10(np.mean(y) / np.mean(np.abs(y - y_smooth)))
        ylabel = "mean intensity (a.u.)"
    else:
        raise ValueError("data_type must be 1 or 2.")
    
    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot 1: Smooth Fit
    axs[0].scatter(x, y, s=10, color=cols[5], label='data')
    axs[0].plot(x, y_smooth, color=cols[6], label='smoother')
    axs[0].set_title(plot_title)
    axs[0].set_xlabel(xlabel)
    axs[0].set_ylabel(ylabel)
    axs[0].grid(True)
    axs[0].legend(loc='upper right', fontsize=cex_leg)
    axs[0].text(0.95, 0.95, f"SNR = {SNR:.3f} dB", 
                horizontalalignment='right', verticalalignment='top', 
                transform=axs[0].transAxes, fontsize=cex_leg)
    
    # Plot 2: Residuals
    residuals = y - y_smooth
    ylim = 1.2 * max(abs(residuals))
    axs[1].set_ylim(-ylim, ylim)
    axs[1].plot(x, residuals, 'o', markersize=3, color=cols[5], label='residuals')
    axs[1].fill_between(x, -2 * uy, 2 * uy, color=col_tr2[1], alpha=0.3, label='data 95% uncert.')
    axs[1].axhline(0, color='black', linewidth=0.8)
    axs[1].set_title('Residuals')
    axs[1].set_xlabel(xlabel)
    axs[1].set_ylabel('residuals (a.u.)')
    axs[1].grid(True)
    axs[1].legend(loc='upper right', fontsize=cex_leg)

    # Show the plot
    plt.savefig(save_path, dpi=600)
    plt.tight_layout()
    # Save the plot to a file

    # Optionally, you can close the plot after saving
    plt.close()

    return {'SNR': SNR}