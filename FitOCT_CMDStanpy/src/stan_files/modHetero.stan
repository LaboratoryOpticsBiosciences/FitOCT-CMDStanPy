// Model standard deviation of heteroscedastic residuals
// by exponential: uy(x) ~ Normal(mean = 0, sd = a*exp(-x/b) )

data {
  int<lower=1>  N;      // size of vectors
  vector[N]     x;      // positions
  vector[N]     y;      // values
  real<lower=0> maxRate;// upper limit (ensuring homoscedasticity)
}
parameters {
  vector<lower=0, upper=maxRate>[2] theta; 
}
transformed parameters {
  vector[N] sig;
  for (n in 1:N)
    sig[n] = theta[1] * exp(-x[n]/theta[2]);
}
model {
  for (n in 1:N)
    y[n] ~ normal(0., sig[n]);
}
