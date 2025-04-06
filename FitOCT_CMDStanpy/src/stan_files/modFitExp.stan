functions{
  // Exp. decay model
  vector phys_mod(array[] real x, vector p, int dataType) {
    int N = size(x);
    vector[N] m;
    for (n in 1:N)
      m[n] = p[1] + p[2] * exp(- dataType * x[n] / p[3]);
    return m;
  }
}

data {
  // Calibration dataset
  int<lower=1>       N;
  array[N] real       x;      // Updated syntax
  vector[N]          y;
  vector<lower=0>[N] uy;

  // Decay model
  int<lower=1>          Np;   // Nb params in decay: should be 3 !!!
  int<lower=1, upper=2> dataType;
}

parameters {
  vector<lower=0>[Np] theta;   // Decay parameters
}

transformed parameters {
  vector[N]     m = phys_mod(x, theta, dataType);
  vector[N]     resid;

  for (n in 1:N)
    resid[n] = (y[n] - m[n]); // Residuals
}

model {
  // Likelihood
  for (n in 1:N)
    resid[n] ~ normal(0, uy[n]);
}

generated quantities {
  // Birge Ratio
  real br = quad_form(inverse(diag_matrix(uy .* uy)), resid)
          / (N - Np);
}
