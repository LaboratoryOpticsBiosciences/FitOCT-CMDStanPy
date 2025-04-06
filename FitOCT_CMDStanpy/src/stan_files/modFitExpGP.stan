functions{
  // Prediction of GP mean at positions x2, knowing
  // x1   : positions of control points
  // y1   : values of control points
  // alpha: dispersion param of GP
  // rho  : correlation factor of GP
  // delta: nugget
  vector gp_pred(array[] real x2,
                 vector y1,
                 array[] real x1,
                 real alpha,
                 real rho,
                 real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2_mu;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      vector[N1] K_div_y2;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N1] K;
      K        = gp_exp_quad_cov(x1, alpha, rho);
      for (n in 1:N1)
        K[n, n] = K[n, n] + delta;
      L_K      = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y2 = mdivide_right_tri_low(K_div_y1',L_K)';
      k_x1_x2  = gp_exp_quad_cov(x1, x2, alpha, rho);
      f2_mu    = (k_x1_x2' * K_div_y2);
    }
    return f2_mu;
  }

  // GP model of decay modulation
  vector inhomo(array[] real x,
                vector yGP,
                array[]real xGP,
                real   alpha,
                real   rho
                ) {
    vector[size(x)] dL = gp_pred(x,yGP,xGP,alpha,rho,1e-9);
    return dL;
  }

  // Modulated exponential decay
  vector phys_mod(array[] real x, // positions
                vector   p, // parameters
                vector   dL,// modulation vector
                int      dataType
                ) {
    int N = size(x);
    vector[N] m;
    for (n in 1:N)
      m[n] = p[1] + p[2] * exp(- dataType*x[n] / (p[3]*(1+dL[n])) );
    return m;
  }

}
data {
  // Calibration dataset
  int<lower=1>          N;
  array[N] real         x;
  vector[N]             y;
  vector<lower=0>[N]    uy;
  int<lower=1, upper=2> dataType;

  // Decay model
  int<lower=1>        Np;          // Nb params in decay
  vector<lower=0>[Np] theta0;      // Reference Decay parameters
  cov_matrix[Np]      Sigma0;      // Prior Corvariance matrix

  // GP
  int<lower=0>        Nn;          // Nb. of control points (CP) for GP
  array[Nn] real xGP;     // GP CP positions
  real<lower=0>       alpha_scale; // GP normalized S.D.
  real<lower=0>       rho_scale;   // GP correl. length in [0,1]
  real<lower=0>       lambda_rate; // Scale of CP S.D.

  // Control
  int<lower=0,upper=1> prior_PD;   // Prior predictive distribution
}
transformed data {
  array[N] real x_scaled;
  array[Nn] real xGP_scaled;

  // Rescale positions for dimensionless GP
  for(n in 1:N)
    x_scaled[n]   = (x[n]  -min(x))/(max(x)-min(x));
  for(n in 1:Nn)
    xGP_scaled[n] = (xGP[n]-min(x))/(max(x)-min(x));
}
parameters {
  vector<lower=0>[Np] theta;   // Decay parameters
  vector<lower=0>[Nn] u;       // Intermediate variable
  vector[Nn]          yGP;     // GP controle values
  real<lower=0>       lambda;  // Lasso rate
  real<lower=0>       sigma;   // Scaling factor for uy
}
transformed parameters {
  real<lower=0> alpha = alpha_scale * sd(yGP);
  vector[N]     dL;
  vector[N]     m;
  vector[N]     resid;

  if(prior_PD == 0) {
    dL = inhomo(x_scaled,yGP,xGP_scaled,alpha,rho_scale);
    m  = phys_mod(x,theta,dL,dataType);
    for (n in 1:N)
      resid[n] = (y[n] - m[n]); // Rediduals
  }

}
model {

  // Hierarchical Prior for Bayesian Lasso
  // (adapted from Mallick2014 and ARD)
  lambda   ~ gamma(2, lambda_rate);
  for (n in 1:Nn)
    u[n]   ~ gamma(2, lambda);
  for (n in 1:Nn)
    yGP[n] ~ normal(0, u[n]);

  // Noise scaling
  sigma ~ normal(1,0.1);

  // Exp. decay parameters
  theta ~ multi_normal(theta0,Sigma0);

  // Likelihood
  if(prior_PD == 0) {
    for (n in 1:N)
      0 ~ normal(resid[n], sigma*uy[n]);
  }

}
generated quantities{
  real br;
  if(prior_PD == 0) {
    // Birge Ratio
    br = quad_form(inverse(sigma^2 * diag_matrix(uy .* uy)),resid)
       / (N-(Np+Nn+2));
  }
}
