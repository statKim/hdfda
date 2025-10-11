#include <RcppEigen.h>
#include <iostream>

// [[Rcpp::depends(RcppEigen)]]

using namespace Eigen;
using namespace Rcpp;


/**
 * @brief Computes the derivative of the SCAD penalty for group norms.
 * @param beta The current vector of all beta coefficients.
 * @param s The uniform size of each group.
 * @param num_groups The total number of groups.
 * @param lambda The lambda tuning parameter for SCAD.
 * @param gamma The gamma tuning parameter for SCAD.
 * @return A vector of weights, one for each group.
 */
VectorXd scad_derivative_group_hdflr(const VectorXd& beta, 
                                     int s,
                                     int num_groups, 
                                     int n,
                                     const MatrixXd& Theta,
                                     double lambda,
                                     double gamma) {
  VectorXd group_weights(num_groups);
  
  for (int g = 0; g < num_groups; ++g) {
    // Use .segment() to get a view of the g-th group without copying
    double norm_beta_g = (Theta.block(0, g * s, n, s) * beta.segment(g * s, s)).norm() / sqrt(n);
    
    if (norm_beta_g <= lambda) {
      group_weights(g) = lambda;
    } else if (norm_beta_g > gamma * lambda) {
      group_weights(g) = 0.0;
    } else {
      group_weights(g) = (gamma * lambda - norm_beta_g) / (gamma - 1.0);
    }
  }
  return group_weights;
}


// --- Inner ADMM Solver ---
// This function solves the z-update subproblem for a SINGLE group.
// It is NOT exported to R and is called by the main outer solver.
VectorXd inner_admm_solver(const VectorXd& v_j, 
                           const MatrixXd& Theta_j,
                           double w_j,
                           double rho,
                           double sigma, 
                           double n, 
                           const MatrixXd& M_j_inv,
                           int max_iter, 
                           double tol_abs,
                           double tol_rel) {
  
  int s = Theta_j.cols(); // size of group j
  
  // Initialize inner loop variables
  VectorXd z_j = VectorXd::Zero(s);
  VectorXd y_j = VectorXd::Zero(n);
  VectorXd d_j = VectorXd::Zero(n); // scaled dual variable
  
  MatrixXd A_j = Theta_j / std::sqrt(n);
  
  for (int m = 0; m < max_iter; ++m) {
    VectorXd y_j_old = y_j; // Store for dual residual
    
    // 1. z_j-Update (Inner)
    VectorXd rhs_z = rho * v_j + sigma * A_j.transpose() * (y_j - d_j);
    z_j = M_j_inv * rhs_z;
    
    // 2. y_j-Update (Inner, Block Soft-Thresholding)
    VectorXd c_j = A_j * z_j + d_j;
    double norm_c_j = c_j.norm();
    double threshold = w_j / sigma;
    
    double scale = 0.0;
    if (norm_c_j > 0) {
      scale = std::max(1.0 - threshold / norm_c_j, 0.0);
    }
    y_j = scale * c_j;
    
    // 3. d_j-Update (Inner, Scaled)
    d_j = d_j + A_j * z_j - y_j;
    
    // --- Inner Loop Convergence Check ---
    double r_norm = (A_j * z_j - y_j).norm();
    double s_norm = (sigma * (y_j - y_j_old)).norm();
    
    double eps_pri = std::sqrt(n) * tol_abs + tol_rel * std::max((A_j * z_j).norm(), y_j.norm());
    double eps_dual = std::sqrt(s) * tol_abs + tol_rel * (sigma * d_j.norm());
    
    if (r_norm < eps_pri && s_norm < eps_dual) {
      break; // Inner loop converged
    }
    
    if (m == max_iter -1){
      Rcpp::Rcout << "Warning: Inner ADMM did not converge within max_iter=" << max_iter << "!" << std::endl;
      // Rcpp::Rcout << z_j.segment(0, 5) << std::endl;
    }
  }
  
  // // Soft-thresholded solution
  // z_j = std::sqrt(n) * (Theta_j.transpose() * Theta_j).inverse() * Theta_j.transpose() * y_j;
  
  return z_j;
}


// --- ADMM Solver for Group Lasso ---

/**
 * @brief Solves the constrained weighted Group Lasso problem using ADMM with block operations.
 * @param Sigma The quadratic matrix.
 * @param delta The vector from the linear constraint.
 * @param s The uniform size of each group.
 * @param group_weights The adaptive weights for each group from LLA.
 * @param rho The ADMM penalty parameter.
 * ... (other params)
 * @return The estimated beta vector.
 */
VectorXd admm_nested_solver(const Eigen::MatrixXd& Sigma,
                            const Eigen::VectorXd& delta,
                            const Eigen::VectorXd& group_weights,
                            const MatrixXd& Theta,
                            int s,
                            int max_iter = 100,
                            int max_iter_nested = 30,
                            double rho = 1.0,
                            double sigma = 1.0,
                            double tol_abs = 1e-4,
                            double tol_rel = 1e-2) {
  // --- Initial Checks ---
  int n = Theta.rows();
  int p = Sigma.rows();
  int num_groups = p / s;
  
  // --- Pre-computation ---
  // 1. Pre-compute the matrix M and its inverse for the beta-update
  MatrixXd M = Sigma + rho * MatrixXd::Identity(p, p) + rho * delta * delta.transpose();
  MatrixXd M_inv = M.inverse();

  // 2. For inner z_j-update
  std::vector<MatrixXd> M_j_inv_list;
  for (int j = 0; j < num_groups; ++j) {
    MatrixXd Theta_j = Theta.block(0, j * s, n, s);
    MatrixXd M_j = (rho * MatrixXd::Identity(s, s)) + (sigma / n) * Theta_j.transpose() * Theta_j;
    M_j_inv_list.push_back(M_j.inverse());
  }

  // --- Initialization ---
  VectorXd beta = VectorXd::Zero(p);
  VectorXd z = VectorXd::Zero(p);
  VectorXd u_2 = VectorXd::Zero(p); // scaled dual variable for beta - z = 0
  double u_1 = 0.0;               // scaled dual variable for delta^T*beta - 1 = 0
  
  // --- Main ADMM Loop ---
  for (int t = 0; t < max_iter; ++t) {
    VectorXd z_old = z; // Store z from previous iteration for convergence check

    // 1. beta-Update
    VectorXd rhs = rho * (z - u_2 + (1.0 - u_1) * delta);
    beta = M_inv * rhs;

    // 2. z-Update (call inner ADMM for each group)
    VectorXd v = beta + u_2;
    for (int j = 0; j < num_groups; ++j) {
      VectorXd v_j = v.segment(j * s, s);
      MatrixXd Theta_j = Theta.block(0, j * s, n, s);

      VectorXd z_j = inner_admm_solver(v_j, Theta_j, group_weights(j), rho, sigma,
                                       n, M_j_inv_list[j], max_iter_nested, tol_abs, tol_rel);
      z.segment(j * s, s) = z_j;
    }

    // 3. u_1-Update (scaled dual update)
    u_1 = u_1 + delta.transpose() * beta - 1.0;

    // 4. u_2-Update (scaled dual update)
    u_2 = u_2 + beta - z;

    // --- Convergence Check ---
    // Primal residuals for consensus constraint
    double r_u1_norm = (beta - z).norm();
    double r_u2_val = delta.transpose() * beta - 1.0;
    double r_u2_norm = std::abs(r_u2_val);
    
    // Dual residual for consensus constraint
    double s_norm = rho * (z - z_old).norm(); 
    
    // Calculate three corresponding tolerances
    double eps_pri_u1 = std::sqrt(p) * tol_abs + tol_rel * std::max(beta.norm(), z.norm());
    double eps_pri_u2 = std::sqrt(1) * tol_abs + tol_rel * std::max((delta.transpose() * beta).norm(), 1.0);
    double eps_dual = std::sqrt(p) * tol_abs + tol_rel * (rho * u_2.norm());
    
    // Check if ALL three conditions are met
    if (r_u1_norm < eps_pri_u1 && r_u2_norm < eps_pri_u2 && s_norm < eps_dual) {
      // Rcpp::Rcout << "Converged at iteration " << t + 1 << std::endl;
      break;
    }
    
    if (t == max_iter -1){
      Rcpp::Rcout << "Warning: ADMM did not converge within max_iter=" << max_iter << "!" << std::endl;
    }
  }

  return beta;
}



// --- Main LLA Solver for Group SCAD ---

// [[Rcpp::export]]
Rcpp::List group_lla_nested_admm(const Eigen::MatrixXd& Sigma,
                                 const Eigen::VectorXd& delta,
                                 int s, // group size
                                 const Eigen::MatrixXd& Theta,
                                 double lambda = 0.1,
                                 double gamma = 3.7,
                                 int max_iter_lla = 100,
                                 int max_iter_admm = 100,
                                 int max_iter_nested_admm = 30,
                                 double rho = 1.0,
                                 double sigma = 1.0,
                                 double tol = 1e-5,
                                 double tol_abs = 1e-4,
                                 double tol_rel = 1e-2) {
  int n = Theta.rows();
  int p = Sigma.rows();
  if (p % s != 0) {
    Rcpp::stop("Total number of variables p must be divisible by group size s.");
  }
  int num_groups = p / s;

  VectorXd beta_hat = VectorXd::Constant(p, 1.0 / p);

  int iter;
  for (iter = 0; iter < max_iter_lla; ++iter) {
    VectorXd beta_hat_old = beta_hat;

    // Compute weights using the block-based helper function
    VectorXd group_weights = scad_derivative_group_hdflr(beta_hat, s, num_groups,
                                                         n, Theta, lambda, gamma);

    // Solve subproblem using the block-based ADMM solver
    beta_hat = admm_nested_solver(Sigma, delta, group_weights, Theta, s,
                                  max_iter_admm, max_iter_nested_admm,
                                  rho, sigma, tol_abs, tol_rel);

    if ((beta_hat - beta_hat_old).norm() / (beta_hat_old.norm() + 1e-8) < tol) {
      // Rcpp::Rcout << "LLA converged at outer iteration " << iter + 1 << std::endl;
      break;
    }
    
    if (iter == max_iter_lla -1){
      Rcpp::Rcout << "Warning: LLA did not converge within max_iter_lla=" << max_iter_lla << "!" << std::endl;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_hat,
    Rcpp::Named("iterations") = iter + 1
  );
}





// --- Inner ADMM Solver (Warm start version) ---
// This function solves the z-update subproblem for a SINGLE group.
// It is NOT exported to R and is called by the main outer solver.
void inner_admm_solver_warm_start(VectorXd& z_j, // Pass by reference for warm start
                                  VectorXd& y_j, // Pass by reference for warm start
                                  VectorXd& d_j, // Pass by reference for warm start
                                  const VectorXd& v_j, 
                                  const MatrixXd& Theta_j,
                                  double w_j,
                                  double rho,
                                  double sigma, 
                                  const MatrixXd& M_j_inv,
                                  int max_iter, 
                                  double tol_abs,
                                  double tol_rel) {
  int n = Theta_j.rows();
  int s = Theta_j.cols(); // size of group j
  
  MatrixXd A_j = Theta_j / std::sqrt(n);
  
  for (int m = 0; m < max_iter; ++m) {
    VectorXd y_j_old = y_j; // Store for dual residual
    
    // 1. z_j-Update (Inner)
    VectorXd rhs_z = rho * v_j + sigma * A_j.transpose() * (y_j - d_j);
    z_j = M_j_inv * rhs_z;
    
    // 2. y_j-Update (Inner, Block Soft-Thresholding)
    VectorXd c_j = A_j * z_j + d_j;
    double norm_c_j = c_j.norm();
    double threshold = w_j / sigma;
    
    double scale = 0.0;
    if (norm_c_j > 0) {
      scale = std::max(1.0 - threshold / norm_c_j, 0.0);
    }
    y_j = scale * c_j;
    
    // 3. d_j-Update (Inner, Scaled)
    d_j = d_j + A_j * z_j - y_j;
    
    // --- Inner Loop Convergence Check ---
    double r_norm = (A_j * z_j - y_j).norm();
    double s_norm = (sigma * (y_j - y_j_old)).norm();
    
    double eps_pri = std::sqrt(n) * tol_abs + tol_rel * std::max((A_j * z_j).norm(), y_j.norm());
    double eps_dual = std::sqrt(s) * tol_abs + tol_rel * (sigma * d_j.norm());
    
    if (r_norm < eps_pri && s_norm < eps_dual) {
      break; // Inner loop converged
    }
    
    if (m == max_iter -1){
      Rcpp::Rcout << "Warning: Inner ADMM did not converge within max_iter=" << max_iter << "!" << std::endl;
      // Rcpp::Rcout << z_j.segment(0, 5) << std::endl;
    }
  }
  
  // // Soft-thresholded solution
  // z_j = std::sqrt(n) * (Theta_j.transpose() * Theta_j).inverse() * Theta_j.transpose() * y_j;
}


// --- ADMM Solver for Group Lasso (Warm start version) ---

/**
 * @brief Solves the constrained weighted Group Lasso problem using ADMM with block operations.
 * @param Sigma The quadratic matrix.
 * @param delta The vector from the linear constraint.
 * @param s The uniform size of each group.
 * @param group_weights The adaptive weights for each group from LLA.
 * @param rho The ADMM penalty parameter.
 * ... (other params)
 * @return The estimated beta vector.
 */
void admm_solver_warm_start(VectorXd& beta, 
                            VectorXd& z, 
                            VectorXd& u_2, 
                            double& u_1,
                            std::vector<VectorXd>& inner_z,
                            std::vector<VectorXd>& inner_y,
                            std::vector<VectorXd>& inner_d,
                            const std::vector<MatrixXd>& M_j_inv_list, // Pass pre-computed inverses
                            const Eigen::MatrixXd& Sigma,
                            const Eigen::VectorXd& delta,
                            const Eigen::VectorXd& group_weights,
                            const MatrixXd& Theta,
                            int s,
                            int max_iter = 100,
                            int max_iter_nested = 30,
                            double rho = 1.0,
                            double sigma = 1.0,
                            double tol_abs = 1e-4,
                            double tol_rel = 1e-2) {
  // --- Initial Checks ---
  int n = Theta.rows();
  int p = Sigma.rows();
  int num_groups = p / s;
  
  // --- Pre-computation ---
  // Pre-compute for the beta-update
  MatrixXd M = Sigma + rho * MatrixXd::Identity(p, p) + rho * delta * delta.transpose();
  MatrixXd M_inv = M.inverse();
  
  // --- Main ADMM Loop ---
  for (int t = 0; t < max_iter; ++t) {
    VectorXd z_old = z; // Store z from previous iteration for convergence check
    
    // 1. beta-Update
    VectorXd rhs = rho * (z - u_2 + (1.0 - u_1) * delta);
    beta = M_inv * rhs;
    
    // 2. z-Update (call inner ADMM for each group)
    VectorXd v = beta + u_2;
    for (int j = 0; j < num_groups; ++j) {
      VectorXd v_j = v.segment(j * s, s);
      MatrixXd Theta_j = Theta.block(0, j * s, n, s);
      
      // Call the inner solver, passing state variables by reference
      inner_admm_solver_warm_start(inner_z[j], inner_y[j], inner_d[j],
                                   v_j, Theta_j, group_weights(j), rho, sigma,
                                   M_j_inv_list[j], max_iter_nested, tol_abs, tol_rel);
      z.segment(j * s, s) = inner_z[j];
    }
    
    // 3. u_1-Update (scaled dual update)
    u_1 = u_1 + delta.transpose() * beta - 1.0;
    
    // 4. u_2-Update (scaled dual update)
    u_2 = u_2 + beta - z;
    
    // --- Convergence Check ---
    // Primal residuals for consensus constraint
    double r_u1_norm = (beta - z).norm();
    double r_u2_norm = std::abs(delta.transpose() * beta - 1.0);
    
    // Dual residual for consensus constraint
    double s_norm = rho * (z - z_old).norm(); 
    
    // Calculate three corresponding tolerances
    double eps_pri_u1 = std::sqrt(p) * tol_abs + tol_rel * std::max(beta.norm(), z.norm());
    double eps_pri_u2 = std::sqrt(1) * tol_abs + tol_rel * std::max((delta.transpose() * beta).norm(), 1.0);
    double eps_dual = std::sqrt(p) * tol_abs + tol_rel * (rho * u_2.norm());
    
    // Check if ALL three conditions are met
    if (r_u1_norm < eps_pri_u1 && r_u2_norm < eps_pri_u2 && s_norm < eps_dual) {
      // Rcpp::Rcout << "Converged at iteration " << t + 1 << std::endl;
      break;
    }
    
    if (t == max_iter -1){
      Rcpp::Rcout << "Warning: ADMM did not converge within max_iter=" << max_iter << "!" << std::endl;
    }
  }
}



// --- Main LLA Solver for Group SCAD (Warm start version) ---

// [[Rcpp::export]]
Rcpp::List group_lla_nested_admm_warm_start(const Eigen::MatrixXd& Sigma,
                                            const Eigen::VectorXd& delta,
                                            int s, // group size
                                            const Eigen::MatrixXd& Theta,
                                            double lambda = 0.1,
                                            double gamma = 3.7,
                                            int max_iter_lla = 100,
                                            int max_iter_admm = 1000,
                                            int max_iter_nested_admm = 30,
                                            double rho = 1.0,
                                            double sigma = 1.0,
                                            double tol = 1e-5,
                                            double tol_abs = 1e-4,
                                            double tol_rel = 1e-2) {
  int n = Theta.rows();
  int p = Sigma.rows();
  if (p % s != 0) {
    Rcpp::stop("Total number of variables p must be divisible by group size s.");
  }
  int num_groups = p / s;
  
  // --- WARM START: All state variables are declared and initialized here ---
  // 1. State variables for admm_solver (outer ADMM)
  VectorXd beta_hat = VectorXd::Constant(p, 1.0 / p);
  VectorXd z = VectorXd::Zero(p);
  VectorXd u_2 = VectorXd::Zero(p);  // scaled dual variable for beta - z = 0
  double u_1 = 0.0;  // scaled dual variable for delta^T*beta - 1 = 0
  
  // 2. State variables for inner_admm_solver
  std::vector<VectorXd> inner_z_states(num_groups);
  std::vector<VectorXd> inner_y_states(num_groups);
  std::vector<VectorXd> inner_d_states(num_groups);
  for (int j = 0; j < num_groups; ++j) {
    inner_z_states[j] = VectorXd::Zero(s);
    inner_y_states[j] = VectorXd::Zero(n);
    inner_d_states[j] = VectorXd::Zero(n);
  }
  
  // Pre-compute M_j_inv matrices for inner solvers
  std::vector<MatrixXd> M_j_inv_list;
  for (int j = 0; j < num_groups; ++j) {
    MatrixXd Theta_j = Theta.block(0, j * s, n, s);
    MatrixXd M_j = (rho * MatrixXd::Identity(s, s)) + (sigma / n) * Theta_j.transpose() * Theta_j;
    M_j_inv_list.push_back(M_j.inverse());
  }
  
  int iter;
  for (iter = 0; iter < max_iter_lla; ++iter) {
    VectorXd beta_hat_old = beta_hat;
    
    // Compute weights using the block-based helper function
    VectorXd group_weights = scad_derivative_group_hdflr(beta_hat, s, num_groups,
                                                         n, Theta, lambda, gamma);
    
    // Call admm_solver, passing all state variables for a full warm start
    admm_solver_warm_start(beta_hat, z, u_2, u_1, 
                           inner_z_states, inner_y_states, inner_d_states, 
                           M_j_inv_list,
                           Sigma, delta, group_weights, Theta, s, 
                           max_iter_admm, max_iter_nested_admm,
                           rho, sigma, tol_abs, tol_rel);
    
    if ((beta_hat - beta_hat_old).norm() / (beta_hat_old.norm() + 1e-8) < tol) {
      // Rcpp::Rcout << "LLA converged at outer iteration " << iter + 1 << std::endl;
      break;
    }
    
    if (iter == max_iter_lla -1){
      Rcpp::Rcout << "Warning: LLA did not converge within max_iter_lla=" << max_iter_lla << "!" << std::endl;
      Rcpp::Rcout << (beta_hat - beta_hat_old).norm() / (beta_hat_old.norm() + 1e-8) << std::endl;
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_hat,
    Rcpp::Named("iterations") = iter + 1
  );
}
