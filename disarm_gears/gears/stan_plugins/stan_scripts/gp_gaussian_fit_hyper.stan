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
  int<lower=1> n_dim;
  vector[n_data] y_data;
  vector[n_dim] X_data[n_data];
  vector[n_data] mu_data;

}


transformed data {

  real delta = 1e-6;

}


parameters {

  real<lower=0> noise_var;
  real<lower=0> cov_var;
  vector<lower=0>[n_dim] cov_length;

}


model {

  matrix[n_data, n_data] K = k_exp_quad_ard(X_data, cov_var, cov_length, delta);
  matrix[n_data, n_data]  L;

  for (i in 1:n_data) K[i, i] = K[i, i] + noise_var;
  L = cholesky_decompose(K);

  noise_var ~ normal(0, 1);
  cov_var ~ normal(0, 1);
  for (i in 1:n_dim) cov_length[i] ~ inv_gamma(5, 5);
  y_data ~ multi_normal_cholesky(mu_data, L);

}

