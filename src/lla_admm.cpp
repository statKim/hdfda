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
VectorXd scad_derivative_group(const VectorXd& beta, 
                               int s,
                               int num_groups, 
                               double lambda,
                               double gamma) {
  VectorXd group_weights(num_groups);
  
  for (int g = 0; g < num_groups; ++g) {
    // Use .segment() to get a view of the g-th group without copying
    double norm_beta_g = beta.segment(g * s, s).norm();
    
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
VectorXd admm_solver(const Eigen::MatrixXd& Sigma,
                     const Eigen::VectorXd& delta,
                     const Eigen::VectorXd& group_weights,
                     int s,
                     double rho = 1.0,
                     int max_iter = 2000,
                     double tol_abs = 1e-4,
                     double tol_rel = 1e-2) {
  // --- Initial Checks ---
  int p = Sigma.rows();
  if (p % s != 0) {
    Rcpp::stop("Total number of variables p must be divisible by group size s.");
  }
  int num_groups = p / s;
  if (group_weights.size() != num_groups) {
    Rcpp::stop("Length of group_weights must be equal to p/s.");
  }
  
  // --- Pre-computation ---
  // Pre-compute the matrix M and its inverse for the beta-update
  MatrixXd M = Sigma + rho * MatrixXd::Identity(p, p) + rho * delta * delta.transpose();
  MatrixXd M_inv = M.inverse();
  
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
    
    // 2. z-Update (Block Soft-Thresholding)
    VectorXd v = beta + u_2;
    for (int g = 0; g < num_groups; ++g) {
      VectorXd v_g = v.segment(g * s, s);
      double norm_v_g = v_g.norm();
      double threshold = group_weights(g) / rho;

      double scale = 0.0;
      if (norm_v_g > 0) {
        scale = std::max(1.0 - threshold / norm_v_g, 0.0);
      }

      z.segment(g * s, s) = scale * v_g;
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
      Rcpp::Rcout << "Warning: ADMM did not converge within max_iter." << std::endl;
    }
  }
  
  // return Rcpp::List::create(
  //   Rcpp::Named("beta") = beta,
  //   Rcpp::Named("z") = z
  // );
  return z;
}



// // --- Inner ADMM Solver for Group Lasso ---
//
// /**
//  * @brief Solves the constrained weighted Group Lasso problem using ADMM with block operations.
//  * @param Sigma The quadratic matrix.
//  * @param delta The vector from the linear constraint.
//  * @param s The uniform size of each group.
//  * @param num_groups The total number of groups.
//  * @param group_weights The adaptive weights for each group from LLA.
//  * @param rho The ADMM penalty parameter.
//  * ... (other params)
//  * @return The estimated beta vector.
//  */
// VectorXd admm_solver(const MatrixXd& Sigma,
//                      const VectorXd& delta,
//                      int s,
//                      int num_groups,
//                      const VectorXd& group_weights,
//                      double rho,
//                      int max_iter_inner,
//                      double tol_abs,
//                      double tol_rel) {
//   int p = Sigma.rows();
//
//   VectorXd beta1 = VectorXd::Zero(p);
//   VectorXd beta2 = VectorXd::Zero(p);
//   double u1 = 0.0;
//   VectorXd u2 = VectorXd::Zero(p);
//
//   MatrixXd Sigma_hat_rho_inv = (rho * (Sigma + rho * delta * delta.transpose() + rho * MatrixXd::Identity(p, p))).inverse();
//
//   for (int t = 0; t < max_iter_inner; ++t) {
//     VectorXd beta2_old = beta2;
//
//     // Step 1: beta1-update
//     beta1 = Sigma_hat_rho_inv * ((1.0 - u1) * delta + (beta2 - u2));
//
//     // Step 2: beta2-update
//     VectorXd v = beta1 + u2;
//     for (int g = 0; g < num_groups; ++g) {
//       // Get a block of the vector v
//       VectorXd v_g = v.segment(g * s, s);
//       double threshold = group_weights(g) / rho;
//
//       // Soft thresholding
//       VectorXd v_g_soft_thres = (v_g.array() - threshold).max(0) - (-v_g.array() - threshold).max(0);
//
//       // Update the corresponding block in beta2
//       beta2.segment(g * s, s) = v_g_soft_thres;
//     }
//
//     // Steps 3 & 4 and convergence check are unchanged
//     u1 = u1 + delta.transpose() * beta1 - 1.0;
//     u2 = u2 + beta1 - beta2;
//
//     double r_norm = (beta1 - beta2).norm();
//     double s_norm = rho * (beta2 - beta2_old).norm();
//     double eps_pri = std::sqrt(p) * tol_abs + tol_rel * std::max(beta1.norm(), beta2.norm());
//     double eps_dual = std::sqrt(p) * tol_abs + tol_rel * (rho * u2.norm());
//
//     if (r_norm < eps_pri && s_norm < eps_dual) {
//       break;
//     }
//   }
//   return beta2;
// }


// --- Main LLA Solver for Group SCAD ---

// [[Rcpp::export]]
Rcpp::List group_lla_admm(const Eigen::MatrixXd& Sigma,
                          const Eigen::VectorXd& delta,
                          int s, // group size
                          double lambda = 0.1,
                          double gamma = 3.7,
                          int max_iter_lla = 100,
                          int max_iter_admm = 1000,
                          double rho = 1.0,
                          double tol = 1e-5,
                          double tol_abs = 1e-4,
                          double tol_rel = 1e-2) {
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
    VectorXd group_weights = scad_derivative_group(beta_hat, s, num_groups,
                                                   lambda, gamma);

    // Solve subproblem using the block-based ADMM solver
    beta_hat = admm_solver(Sigma, delta, group_weights, s, rho,
                           max_iter_admm, tol_abs, tol_rel);
    // beta_hat = admm_solver(Sigma, delta, s, num_groups, group_weights, rho,
    //                        max_iter_inner, tol_abs, tol_rel);

    if ((beta_hat - beta_hat_old).norm() / (beta_hat_old.norm() + 1e-8) < tol) {
      // Rcpp::Rcout << "LLA converged at outer iteration " << iter + 1 << std::endl;
      break;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_hat,
    Rcpp::Named("iterations") = iter + 1
  );
}
