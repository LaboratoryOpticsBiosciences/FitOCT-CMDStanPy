// UQ by Moments matching
functions{
  vector u_phys_mod(real[] x, vector theta, vector u_theta, int dataType) {
    int N  = size(x);
    vector[N]  up;
    vector[3]  J;
    real       tmp;
    
    // Linear UP for exponential decay model
    for (n in 1:N) {
      tmp  = exp( -dataType*x[n]/theta[3]);
      J[1] = 1.0;
      J[2] = tmp;
      J[3] = dataType*x[n]*theta[2]/theta[3]^2 * tmp;
      up[n] = sqrt( (u_theta[1]*J[1])^2 
                  + (u_theta[2]*J[2])^2 
                  + (u_theta[3]*J[3])^2 );
    }
    return up;
  }
}
data {
  // Data grid
  int<lower=1>          N;
  real                  x[N];
  vector<lower=0>[N]    uy;
  int<lower=1, upper=2> dataType;
  
  // Calibration dataset
  real<lower=0>         Sobs[2];

  // Decay model
  int<lower=1>          Np; // Nb params in decay: should be 3 !!!
  vector<lower=0>[Np]   theta;   // Decay parameters

  // Similarity parameter
  real<lower=0>         eps;
}
parameters {
  vector<lower=0>[Np]   u_theta;   // Decay parameters unc.
}
transformed parameters {
  real<lower=0>         Ssim[2];
  // Compute weighted prediction uncertainty
  vector[N] up = u_phys_mod(x, theta, u_theta, dataType) ./ uy;
  // Statistics
  Ssim[1] = 1.96 * sqrt(mean(up .* up)); // PU95
  Ssim[2] = sd(up);                      // sd(PU)
}
model {
  for (i in 1:2)
    target += normal_lpdf( Ssim[i]-Sobs[i] | 0, eps);
}
