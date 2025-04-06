import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel

def plotMonoExp(x, y, uy, ySmooth, mod, resid, gPars, dataType, br,save_path):

    # Extract graphical parameters
    pty = gPars.get('pty', 'normal')  # Example, adapt based on actual gPars structure
    mar = gPars.get('mar', [5, 4, 4, 2])  # Adjust margin sizes if necessary
    cols = gPars.get('cols', ['black', 'blue', 'red', 'green'])  # Adjust color palette
    cex = gPars.get('cex', 1)
    cex_leg = gPars.get('cex_leg', 1)

    # Set up the subplots
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot data and model fit
    ax[0].plot(x, y, 'o', markersize=5, color=cols[5], label="data")
    ax[0].plot(x, mod, color=cols[6], label="best fit")
    ax[0].set_title("Fit")
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("mean intensity (a.u.)" if dataType == 2 else "mean amplitude (a.u.)")
    ax[0].grid(True)
    ax[0].legend()

    # Add model fit statistics
    ax[0].legend(loc='upper right', fontsize=cex_leg, frameon=False, 
              labels=[f'Data', f'Best fit Birge ratio = {br:.2f}'])

    # Plot residuals
    ax[1].plot(x, resid, 'o', markersize=5, color=cols[6], label="mean resid.")
    ax[1].fill_between(x, -2*uy, 2*uy, color=cols[4], alpha=0.5, label="data 95% uncert.")
    ax[1].plot(x, ySmooth - mod, color=cols[6], label="best fit - smooth")
    ax[1].axhline(0, color='black', linestyle='--')
    ax[1].set_title("Residuals")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("residuals (a.u.)")
    ax[1].grid(True)
    ax[1].legend()

    # Adjust the layout and show the plot
    plt.savefig(save_path, dpi=600) 
    plt.tight_layout()
    #plt.show()
    plt.close()