#include <RcppEigen.h>
#include <cmath>
#include <vector> // To store pre-calculated matrices

// [[Rcpp::depends(RcppEigen)]]

// Derivative of SCAD penalty
double scad_deriv(double t, double lambda, double a) {
  if (t <= lambda) {
    return lambda;
  } else if (t <= a * lambda) {
    return (a * lambda - t) / (a - 1.0);
  } else {
    return 0.0;
  }
}

// [[Rcpp::export]]
Rcpp::List group_scad_flr_cpp(const Eigen::Map<Eigen::VectorXd> Y,
                              const Eigen::Map<Eigen::MatrixXd> Theta,
                              int p,
                              int s,
                              const double lambda,
                              const double a = 3.7,
                              const int max_iter = 100,
                              const double tol = 1e-6) {
  // Get dimensions and validate them
  int n = Theta.rows();
  if (Theta.cols() != p * s) {
    Rcpp::stop("Number of columns in Theta must be equal to p * s.");
  }

  // Pre-calculate (Theta_j' * Theta_j)^-1 * Theta_j' part for each group.
  // This avoids redundant, expensive calculations inside the main loop.
  std::vector<Eigen::MatrixXd> pre_mat_calc_list(p);
  for (int j = 0; j < p; ++j) {
    const auto& Theta_j = Theta.block(0, j * s, n, s);
    Eigen::MatrixXd Tjt_Tj = Theta_j.transpose() * Theta_j;
    // Solve (Tjt_Tj) * X = Theta_j' for X. X is the s x n pseudo-inverse part.
    pre_mat_calc_list[j] = Tjt_Tj.ldlt().solve(Theta_j.transpose());
  }

  // (i) Initialize parameters
  Eigen::MatrixXd f_hat = Eigen::MatrixXd::Zero(n, p);
  Eigen::MatrixXd f_hat_old = Eigen::MatrixXd::Zero(n, p);

  int iter;
  double n_sqrt = std::sqrt(static_cast<double>(n));
  // double s_sqrt = std::sqrt(static_cast<double>(s));

  // (v) Main iterative loop
  for (iter = 0; iter < max_iter; ++iter) {
    f_hat_old = f_hat;

    Eigen::VectorXd f_sum = f_hat.rowwise().sum();

    // Loop over each group j
    for (int j = 0; j < p; ++j) {
      // (ii) Calculate the partial residual R_j
      Eigen::VectorXd R_j = Y - f_sum + f_hat.col(j);

      // (iii) Calculate P_hat_j using the pre-calculated matrix
      // P_hat_j = Theta_j * [(Theta_j' * Theta_j)^-1 * Theta_j' * R_j]
      const auto& Theta_j = Theta.block(0, j * s, n, s);
      const auto& pre_mat_calc_j = pre_mat_calc_list[j];
      Eigen::VectorXd P_hat_j = Theta_j * (pre_mat_calc_j * R_j);

      // (iv) Update f_hat_j using Group SCAD shrinkage
      double norm_P = P_hat_j.norm();
      double norm_f = f_hat.col(j).norm();

      if (norm_P < 1e-9) {
        f_sum -= f_hat.col(j);
        f_hat.col(j).setZero();
      } else {
        double t = norm_f / n_sqrt;
        double deriv = scad_deriv(t, lambda, a);
        // double deriv = scad_deriv(t, lambda*s_sqrt, a);
        double shrinkage = 1.0 - deriv * n_sqrt / norm_P;

        f_sum -= f_hat.col(j);

        // (v) Update f_hat_j with centering
        f_hat.col(j) = std::max(0.0, shrinkage) * P_hat_j;
        f_hat.col(j).array() -= f_hat.col(j).mean();

        // Update f_sum using the updated f_hat_j
        f_sum += f_hat.col(j);
      }
    }

    // Check for convergence
    double change = (f_hat - f_hat_old).norm() / (f_hat_old.norm() + 1e-8);
    if (change < tol) {
      break;
    }
  }

  // (vii) Final estimate eta_hat_j = (Theta_j' * Theta_j)^-1 * Theta_j' * f_hat_j
  Eigen::MatrixXd eta_hat_j = Eigen::MatrixXd::Zero(s, p);
  for (int j = 0; j < p; ++j) {
    const auto& pre_mat_calc_j = pre_mat_calc_list[j];
    eta_hat_j.col(j) = (pre_mat_calc_j * f_hat.col(j));
  }

  return Rcpp::List::create(
    Rcpp::Named("eta_hat") = eta_hat_j,
    Rcpp::Named("iterations") = iter + 1
  );
}
