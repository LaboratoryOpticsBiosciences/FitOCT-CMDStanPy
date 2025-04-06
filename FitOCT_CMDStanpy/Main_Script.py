#### Loading packages ####
import os
import numpy as np
import pandas as pd
import traceback
import logging
from datetime import datetime
from tqdm import tqdm
from cmdstanpy import CmdStanModel
# Optional but often used with Stan for statistics
from scipy import stats
from functions import (
    estimate_exp_prior, estimate_noise, exp_decay_model, fit_exp_gp,
    fit_mono_exp, n_act_ctrl_pts, out_exp_gp, plot_exp_gp,
    plot_noise, plotMonoExp,printBr,selX
)

### INITIAL PARAMETERS
# Define the parent directory
parent_dir = ''
Cmdstanpy_files_dir=''


# Control parameters ####

### Default values / Set values here
depthSel    = None # Otherwise c(xmin,xmax)
dataType    = 2    # Intensity
subSample   = 1
smooth_df   = 15
methods = ['sample', 'optim', 'vb']
method = methods[1]  # In Python, indexing starts from 0, so we use [0] to select the first element
nb_warmup   = 500
nb_sample   = 1000
modRange    = 0.5
ru_theta    = 0.05
lambda_rate = 0.1
gridType    = 'internal'
Nn          = 10
rho_scale   = 0.1
priPost     = True # Compare prior and posterior pdf ?
priorType   = 'abc'
model='modFitMonoExp'

g_pars_noise = {
    'cols': ['blue', 'green', 'red', 'purple', 'orange', 'gray', 'cyan', 'magenta'],
    'pty': 's', 'mar': [5, 5, 4, 2], 'mgp': [2, 1, 0], 'tcl': 0.5, 'lwd': 2, 
    'cex': 1, 'cex_leg': 0.8, 'plot_title': 'Data vs Smoothed', 'xlabel': 'Stromal Depth (µm)'
    }

# Mono fit plot
gPars_mono = {
    'cols': ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF'],
    'plot_title': 'Monoexponential Fit',
    'xlabel': 'Time (s)',
    'cex': 14,
    'cex_leg': 10,
    'pty': 's',
    'mar': [5, 4, 4, 2],
    'mgp': [3, 1, 0],
    'tcl': 0.5,
    'lwd': 2,
    'col_tr2': ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00']
    }



# Mono fit plot
gPars_expgp = {
    'cols': ['#000000', '#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#FF00FF', '#00FFFF', '#000000'],
    'col_tr': ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00', '#FF00FF'],
    'col_tr2': ['#FF00FF', '#00FFFF', '#FFFF00', '#00FF00', '#FF00FF'],
    'plot_title': 'Modulated exp. fit',
    'xlabel': 'Stromal Depth (µm)',
    'cex': 14,
    'cex_leg': 10
}



################################################
#### Iterate over each directory

cmdstanpy_logger = logging.getLogger("cmdstanpy")
cmdstanpy_logger.disabled = True

# Get all the subdirectories in the parent directory
subdirs = [subdir for subdir in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, subdir))]

# Dictionary to log errors
error_log = {}

# Loop over each folder in the parent directory
for subdir in tqdm(subdirs, desc="Processing directories", unit="folder"):
        
    try:
        subdir_path = os.path.join(parent_dir, subdir)
        Patient_name=subdir
        # Check if it is a directory
        if os.path.isdir(subdir_path):
            courbe_name="Courbe.csv"
            courbe_path = os.path.join(subdir_path,courbe_name)
            ################################################
            #Read curve
            D = pd.read_csv(courbe_path)
        ################################################
            # create path
            final_folder='results_python_expGP_sample/'
            full_save_path=os.path.join(subdir_path, final_folder)
            os.makedirs(full_save_path, exist_ok=True)
            ################################################
            ### NOISE
            # Apply selector

            C = selX(D.iloc[:, 0].values, D.iloc[:, 1].values, depthSel, subSample)
            x = C['x']
            y = C['y']
            depth = np.max(x) - np.min(x)
            ### Estimate data uncertainty
            fits = estimate_noise(x, y, df=15,CMDStan_path=Cmdstanpy_files_dir)  
            uy = fits['uy']
            y_smooth = fits['y_smooth']
            # Call the function to plot
            noise_file_name='noise.png'
            noise_save_path=os.path.join(full_save_path, noise_file_name)
            noise_result = plot_noise(x, y, uy, y_smooth,g_pars_noise,noise_save_path,data_type=2)
            ### MAP Inference of exponential decay parameters
            fit_m = fit_mono_exp(x, y, uy,CMDStan_path=Cmdstanpy_files_dir)
            # Check if fit_m is None (if the fitting failed)
            if fit_m is None:
                print("Error: fit_mono_exp returned None")
            else:
                # Extract the best theta, correlation matrix, and uncertainty from the fit result
                theta0 = np.array(fit_m['best_theta'])  # Use .get() to avoid KeyError if key doesn't exist
                cor_theta = fit_m['cor_theta']
                unc_theta = fit_m['unc_theta']
                # Perform Birge's ratio analysis (equivalent to printBr in R)
                br_result = printBr(fit_m,model)
                # Get the Birge's ratio and CI95
                mono_br = br_result['br']
                mono_br_ci95 = br_result['CI95']
                mono_alert = br_result['alert']
                # Call the plot function
                path_mono_folder='mono_exp/'
                mono_save_path=os.path.join(full_save_path, path_mono_folder)
                os.makedirs(mono_save_path, exist_ok=True)
                mono_exp_file_name='fit_mono.png'
                final_mono_save_path=os.path.join(mono_save_path, mono_exp_file_name)
                # Create ExpGP directory and set save path
                path_expgp_folder = 'ExpGP_exp/'
                expgp_dir = os.path.join(full_save_path, path_expgp_folder)
                os.makedirs(expgp_dir, exist_ok=True)
                final_expgp_save_path = os.path.join(expgp_dir, 'expgp_fit')  # Base filename without extension

                if fit_m['method']=='sample':
                    plotMonoExp(x, y, uy, y_smooth,  np.mean(fit_m['fit'].stan_variable('m'),axis=0) , 
                            np.mean(fit_m['fit'].stan_variable('resid'),axis=0) ,gPars_mono,2,mono_br,final_mono_save_path)
                else:
                    plotMonoExp(x, y, uy, y_smooth, fit_m['fit'].stan_variable('m') , 
                        fit_m['fit'].stan_variable('resid') ,gPars_mono,2,mono_br,final_mono_save_path)
                if mono_alert=='!!! WARNING: br out of interval !!!':
                    priExp=estimate_exp_prior(x, uy,2,'mono', out=fit_m,CMDStan_path=Cmdstanpy_files_dir)
                    fitGP=fit_exp_gp(x, y, uy,
                                2, 10, priExp['theta0'], priExp['Sigma0'],
                                lambda_rate=0.1, lasso=False, method='sample', iter=50000,
                                prior_PD=0, alpha_scale=0.1, rho_scale=1/10, grid_type='internal',
                                nb_chains=4, nb_warmup=500, nb_iter=nb_warmup+nb_sample,
                                verbose=False,Cmdstan_path=Cmdstanpy_files_dir)
                    print('passed here')
                    params=out_exp_gp(x,y,uy,fitGP,2)
                    model_gp='modFitExpGP'
                    br_gp=printBr(fitGP,model_gp)
                    plot_exp_gp(x, y, uy, y_smooth, fitGP, gPars_expgp,mod_scale=0.3, nMC=100, data_type=2, br=br_gp,save_path=final_expgp_save_path)

                    ################################################
                    ### RESULTS
                    results_table = pd.DataFrame([{
                        'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'tag': Patient_name,  # Replace with actual patient name
                        'SNR': round(noise_result['SNR'], 3),
                        'Mono_br': round(mono_br, 3),
                        'Mono_brCI95': f"{round(mono_br_ci95[0], 3)}-{round(mono_br_ci95[1], 3)}",
                        'Mono_alert': mono_alert,
                        'GP_br': round(br_gp['br'],3),
                        'GP_brCI95': f"{round(br_gp['CI95'][0], 3)}-{round(br_gp['CI95'][1], 3)}",
                        'GP_alert': br_gp['alert'],
                        'C0': round(params['sum'][0][0], 4),
                        'uC0': round(params['sum'][0][1], 4),
                        'A0': round(params['sum'][1][0], 4),
                        'uA0': round(params['sum'][1][1], 4),
                        'Ls': round(params['sum'][2][0], 4),
                        'uLs': round(params['sum'][2][1], 4),
                        'eta': round(params['sum'][2][0] / depth, 4)
                    }])



                else:
                    results_table = pd.DataFrame([{
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'tag': Patient_name,  # Replace with actual patient name
                    'SNR': round(noise_result['SNR'], 3),
                    'Mono_br': round(mono_br, 3),
                    'Mono_brCI95': f"{round(mono_br_ci95[0], 3)}-{round(mono_br_ci95[1], 3)}",
                    'Mono_alert': mono_alert,
                    'GP_br': None,
                    'GP_brCI95': None,
                    'GP_alert': None,
                    'C0': round(theta0[0], 4),
                    'uC0': round(unc_theta[0], 4),
                    'A0': round(theta0[1], 4),
                    'uA0': round(unc_theta[1], 4),
                    'Ls': round(theta0[2], 4),
                    'uLs': round(unc_theta[2], 4),
                    'eta': round(theta0[2] / depth, 4)
                }])

                # Save the results table to a CSV
                results_file_name='results_fit.csv'
                results_save_path=os.path.join(full_save_path, results_file_name)
                results_table.to_csv(results_save_path, index=False)

    except Exception as e:
        # Log the error
        error_log[subdir] = {
            "error_message": str(e),
            "traceback": traceback.format_exc()
        }
        # Optional: log the full traceback for debugging
        # print(traceback.format_exc())


# After the loop, handle the error log
if error_log:
    print("The following directories encountered errors:")
    for dir_name, error_info in error_log.items():
        print(f"Directory: {dir_name}")
        print(f"Error: {error_info['error_message']}")
else:
    print("All directories processed successfully!")           
