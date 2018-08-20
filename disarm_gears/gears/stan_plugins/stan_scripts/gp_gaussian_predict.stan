// Gaussian process regression
// ---------------------------

functions {

  /**
   * Exponentiated Quadratic Kernel
   *
   * @param X : Inputs
   * @param cov_var : Variance parameter
   * @param cov_length : Vector of kernel lengthscales
   * @param delta : jitter
   *
   * @return Exponentiated Quadratic Kernel K(X, X)
   */
   //
  matrix k_exp_quad_ard(vector[] X, real cov_var, vector cov_length, real delta) {

    int n = size(X);
    matrix[n, n] K;

    for (i in 1:(n-1)) {
      K[i, i] = cov_var + delta;
      for (j in (i +1):n) {
        K[i, j] = cov_var * exp(-.5 * dot_self((X[i] - X[j]) ./ cov_length));
        K[j, i] = K[i, j];
      }
    }
    K[n, n] = cov_var + delta;

    return K;

  }

}


data {
  
  int<lower=1> n_data;
  int<lower=1> n_pred;
  int<lower=1> n_dim;
  real<lower=0> cov_var;
  real<lower=0> noise_var;

  vector<lower=0>[n_dim] cov_length;
  vector[n_data] mu_data;
  vector[n_pred] mu_pred;

  vector[n_data] y_data;
  vector[n_dim] X_data[n_data];
  vector[n_dim] X_pred[n_pred];

}


transformed data {

  real delta = 1e-6;
  int<lower=1> n_both = n_data + n_pred;

  vector[n_both] mu;

  vector[n_dim] X_both[n_both];
  matrix[n_both, n_both] K;
  matrix[n_both, n_both] L;

  for (i in 1:n_data) mu[i] = mu_data[i];
  for (i in 1:n_pred) mu[n_data + i] = mu_pred[i];

  for (i in 1:n_data) X_both[i] = X_data[i];
  for (i in 1:n_pred) X_both[n_data + i] = X_pred[i];

  K = k_exp_quad_ard(X_both, cov_var, cov_length, delta);
  L = cholesky_decompose(K);

}


parameters {

  vector[n_both] eta;

}


transformed parameters {

  real noise_sigma = sqrt(noise_var);
  vector[n_both] f = mu + L * eta;

}


model {

  eta ~ normal(0, 1);

  y_data ~ normal(f[1:n_data], noise_sigma);

}


generated quantities {

  vector[n_pred] y_pred;

  for (i in 1:n_pred) y_pred[i] = normal_rng(f[n_data + i], noise_sigma);

}

