#include <RcppEigen.h>
// #include <Eigen/QR>    // For CompleteOrthogonalDecomposition
// [[Rcpp::depends(RcppEigen)]]

using namespace Eigen;
using namespace Rcpp;


/**
 * @brief Computes the derivative of the SCAD penalty.
 * @param lambda The lambda tuning parameter for SCAD.
 * @param a The gamma tuning parameter for SCAD.
 * @return A vector of weights, one for each group.
 */
VectorXd scad_deriv(const VectorXd& norm_vec, 
                    double lambda,
                    double a = 3.7) {
  int num_groups = norm_vec.size();
  VectorXd weights(num_groups);
  
  for (int g = 0; g < num_groups; ++g) {
    if (norm_vec(g) <= lambda) {
      weights(g) = lambda;
    } else if (norm_vec(g) > a * lambda) {
      weights(g) = 0.0;
    } else {
      weights(g) = (a * lambda - norm_vec(g)) / (a - 1.0);
    }
  }
  return weights;
}



// Project a vector onto the L1 ball
// Computes the Euclidean projection of a vector onto the L1 ball.
// This function solves: argmin_v ||v - y||_2^2 s.t. ||v||_1 <= R
VectorXd proj_l1_ball(const VectorXd& y, double R = 1.) {
   // Basic sanity check
   if (R <= 0) {
     return VectorXd::Zero(y.size());
   }
   
   // Step 1: Check if the vector is already inside the L1 ball
   double y_l1_norm = y.lpNorm<1>();
   if (y_l1_norm <= R) {
     return y;
   }
   
   // Step 2: Take absolute values
   VectorXd y_abs = y.cwiseAbs();
   
   // Step 3: Sort the absolute values in descending order.
   // Eigen doesn't have a direct sort, so we copy to std::vector to sort.
   std::vector<double> y_abs_sorted(y_abs.data(), y_abs.data() + y_abs.size());
   std::sort(y_abs_sorted.begin(), y_abs_sorted.end(), std::greater<double>());
   
   // Step 4 & 5: Find the optimal threshold 'tau'
   // This is the core of the efficient algorithm by Duchi et al. (2008).
   // We are looking for the correct Lagrange multiplier for the dual problem,
   // which corresponds to the soft-thresholding value.
   double cum_sum = 0.0;
   double tau = 0.0;
   
   for (int k = 0; k < y_abs_sorted.size(); ++k) {
     cum_sum += y_abs_sorted[k];
     double temp_tau = (cum_sum - R) / (k + 1.0);
     // We find the pivot point where the cumulative sum condition is met.
     // This ensures ||S_tau(y)||_1 = R.
     if (y_abs_sorted[k] > temp_tau) {
       tau = temp_tau;
     }
   }
   
   // Step 6: Apply soft-thresholding with the found 'tau'
   // v_i = sign(y_i) * max(|y_i| - tau, 0)
   // We use Eigen's element-wise (cwise) operations for efficiency.
   VectorXd v = (y_abs.array() - tau).cwiseMax(0);
   return v.array() * y.cwiseSign().array();
 }



// --- ADMM Algorithm ---

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
                     const MatrixXd& Theta,
                     int s,
                     double R,
                     const Eigen::LLT<Eigen::MatrixXd>& M_chol,
                     int max_iter = 100,
                     double rho = 1.0,
                     double tol_abs = 1e-4,
                     double tol_rel = 1e-3) {
  // --- Initial Checks ---
  int n = Theta.rows();
  int p = Sigma.rows();
  int num_groups = p / s;
  
  // --- Initialization ---
  VectorXd beta = VectorXd::Zero(p);
  VectorXd z = VectorXd::Zero(n);
  VectorXd y = VectorXd::Zero(p);
  double u_1 = 0.0;               // scaled dual variable for delta^T*beta - 1 = 0
  VectorXd u_2 = VectorXd::Zero(n); // scaled dual variable for n^(-1/2)*Theta*beta - z = 0
  VectorXd u_3 = VectorXd::Zero(p); // scaled dual variable for beta - y = 0
  MatrixXd z_mat = MatrixXd::Zero(n, p);  // for each covariates; rowSums(z_mat) = z
  MatrixXd u_2_mat = MatrixXd::Zero(n, p);  // for each covariates; rowSums(u_2_mat) = u_2
  
  MatrixXd Theta_2 = Theta / std::sqrt(n);
  
  // MatrixXd M = Sigma + rho/n * Theta.transpose() * Theta + rho * delta * delta.transpose();
  // MatrixXd M_pinv = M.completeOrthogonalDecomposition().pseudoInverse();
  
  // --- ADMM Loop ---
  for (int t = 0; t < max_iter; ++t) {
    VectorXd z_old = z; // Store z from previous iteration for convergence check
    VectorXd y_old = y; // Store y from previous iteration for convergence check

    // 1. beta-Update
    VectorXd rhs = rho * ((1.0 - u_1) * delta + Theta_2.transpose() * (z - u_2) + (y - u_3));
    beta = M_chol.solve(rhs);
    // beta = M_pinv * rhs;

    // 2. z-Update (with updating u_2_mat)
    for (int j = 0; j < num_groups; ++j) {
      MatrixXd Theta_2_j = Theta_2.block(0, j * s, n, s) * beta.segment(j * s, s);
      VectorXd v_j = Theta_2_j + u_2_mat.col(j);
      double norm_v_j = v_j.norm();
      double threshold = group_weights(j) / rho;
      
      double scale = 0.0;
      if (norm_v_j > 0) {
        scale = std::max(1.0 - threshold / norm_v_j, 0.0);
      }
      z_mat.col(j) = scale * v_j;
      
      // Update u_2_mat
      u_2_mat.col(j) = u_2_mat.col(j) + Theta_2_j - z_mat.col(j);
    }
    z = z_mat.rowwise().sum();

    // 3. y-Update
    y = proj_l1_ball(beta + u_3, R);
    
    // 4. Dual-Update (scaled dual update)
    u_1 = u_1 + delta.transpose() * beta - 1.0;
    u_2 = u_2_mat.rowwise().sum();
    u_3 = u_3 + beta - y;
    
    
    // --- Convergence Check ---
    // Primal residuals
    double r_primal_delta = delta.transpose() * beta - 1.0;
    VectorXd r_primal_z = Theta_2 * beta - z;
    VectorXd r_primal_y = beta - y;
    double r_primal = std::sqrt(std::pow(r_primal_delta, 2.) + 
                                r_primal_z.squaredNorm() + 
                                r_primal_y.squaredNorm());
    
    // Dual residuals
    VectorXd r_dual_z = rho * Theta_2.transpose() * (z - z_old);
    VectorXd r_dual_y = rho * (y - y_old);
    double r_dual = std::sqrt(r_dual_z.squaredNorm() + r_dual_y.squaredNorm());
    
    // Thresholds
    double eps_primal = std::sqrt(p + n + 1.0) * tol_abs + tol_rel * std::max({
      (Theta_2 * beta).norm(), z.norm(), y.norm()
    });
    double eps_dual = std::sqrt(p) * tol_abs + tol_rel * rho * std::max({
      (Theta_2.transpose() * u_2).norm(), u_3.norm() 
    });
    
    // MatrixXd A(n + 1, p);
    // A.row(0) = delta;
    // A.block(1, 0, n, p) = Theta_2;
    // MatrixXd B(n + 1, n);
    // B.row(0).setZero();
    // B.block(1, 0, n, n) = -MatrixXd::Identity(n, n);
    // VectorXd c = VectorXd::Zero(n + 1);
    // c(0) = 1;
    // 
    // // Primal and dual residuals
    // double r_primal = (A * beta + B * z - c).norm();
    // double r_dual = rho * (Theta_2.transpose() * (z - z_old)).norm();
    // 
    // // Thresholds
    // double eps_primal = std::sqrt(n + 1.0) * tol_abs + tol_rel * std::max({
    //   (A * beta).norm(), (B * z).norm(), c.norm()
    // });
    // double eps_dual = std::sqrt(p) * tol_abs + tol_rel * (Theta_2.transpose() * u_2).norm();
    
    // Check if ALL three conditions are met
    if (r_primal < eps_primal && r_dual < eps_dual) {
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
Rcpp::List group_lla_admm(const Eigen::MatrixXd& Sigma,
                          const Eigen::VectorXd& delta,
                          int s, // group size
                          const Eigen::MatrixXd& Theta,
                          double lambda = 0.1,
                          double R = 1.,
                          double gamma = 3.7,
                          int max_iter_lla = 100,
                          int max_iter_admm = 100,
                          double rho = 1.0,
                          double tol = 1e-5,
                          double tol_abs = 1e-4,
                          double tol_rel = 1e-3) {
  int n = Theta.rows();
  int p = Sigma.rows();
  if (p % s != 0) {
    Rcpp::stop("Total number of variables p must be divisible by group size s.");
  }
  int num_groups = p / s;

  // --- Cholesky decomposition objects for inverse computations ---
  // Pre-compute M_inv for the beta-update (outer ADMM)
  MatrixXd M = Sigma + rho * delta * delta.transpose() + rho/n * Theta.transpose() * Theta + rho * MatrixXd::Identity(p, p);
  auto M_chol = M.llt();

  VectorXd beta_hat = VectorXd::Zero(p);
  // VectorXd beta_hat = VectorXd::Constant(p, 1.0 / p);
  
  std::vector<double> error_conv;
  int iter;
  for (iter = 0; iter < max_iter_lla; ++iter) {
    VectorXd beta_hat_old = beta_hat;
    
    // Compute group norm
    VectorXd norm_beta_g(num_groups);
    for (int g = 0; g < num_groups; ++g) {
      // Use .segment() to get a view of the g-th group without copying
      norm_beta_g(g) = (Theta.block(0, g * s, n, s) * beta_hat.segment(g * s, s)).norm() / sqrt(n);
    }
    
    // Compute weights using the block-based helper function
    VectorXd group_weights = scad_deriv(norm_beta_g, lambda, gamma);

    // Solve subproblem using the block-based ADMM solver
    beta_hat = admm_solver(Sigma, delta, group_weights, Theta, s, R,
                           M_chol,
                           max_iter_admm, rho, tol_abs, tol_rel);

    // Check the convergence
    double beta_diff = (beta_hat - beta_hat_old).norm() / (beta_hat_old.norm() + 1e-8);
    error_conv.push_back(beta_diff);
    if (beta_diff < tol) {
      break;
    }

    if (iter == max_iter_lla -1){
      Rcpp::Rcout << "Warning: LLA did not converge within max_iter_lla=" << max_iter_lla << "!" << std::endl;
    }
  }

  return Rcpp::List::create(
    Rcpp::Named("beta") = beta_hat,
    Rcpp::Named("iterations") = iter + 1,
    Rcpp::Named("error_conv") = error_conv
  );
}




// // --- ADMM Algorithm (Warm Start) ---
// 
// /**
//  * @brief Solves the constrained weighted Group Lasso problem using ADMM with block operations.
//  * @param Sigma The quadratic matrix.
//  * @param delta The vector from the linear constraint.
//  * @param s The uniform size of each group.
//  * @param group_weights The adaptive weights for each group from LLA.
//  * @param rho The ADMM penalty parameter.
//  * ... (other params)
//  * @return The estimated beta vector.
//  */
// void admm_solver_warm_start(VectorXd& beta, 
//                             VectorXd& z, 
//                             VectorXd& u_2, 
//                             double& u_1,
//                             MatrixXd& z_mat,
//                             MatrixXd& u_2_mat,
//                             const Eigen::MatrixXd& Sigma,
//                             const Eigen::VectorXd& delta,
//                             const Eigen::VectorXd& group_weights,
//                             const MatrixXd& Theta,
//                             int s,
//                             const Eigen::LLT<Eigen::MatrixXd>& M_chol,
//                             int max_iter = 100,
//                             double rho = 1.0,
//                             double tol_abs = 1e-4,
//                             double tol_rel = 1e-3) {
//   // --- Initial Checks ---
//   int n = Theta.rows();
//   int p = Sigma.rows();
//   int num_groups = p / s;
//   
//   MatrixXd Theta_2 = Theta / std::sqrt(n);
//   
//   // --- ADMM Loop ---
//   for (int t = 0; t < max_iter; ++t) {
//     VectorXd z_old = z; // Store z from previous iteration for convergence check
//     
//     // 1. beta-Update
//     VectorXd rhs = rho * (Theta_2.transpose() * (z - u_2) + (1.0 - u_1) * delta);
//     beta = M_chol.solve(rhs);
//     
//     // 2. z-Update (with updating u_2_mat)
//     for (int j = 0; j < num_groups; ++j) {
//       MatrixXd Theta_2_j = Theta_2.block(0, j * s, n, s) * beta.segment(j * s, s);
//       VectorXd v_j = Theta_2_j + u_2_mat.col(j);
//       double norm_v_j = v_j.norm();
//       double threshold = group_weights(j) / rho;
//       
//       double scale = 0.0;
//       if (norm_v_j > 0) {
//         scale = std::max(1.0 - threshold / norm_v_j, 0.0);
//       }
//       z_mat.col(j) = scale * v_j;
//       
//       // Update u_2_mat
//       u_2_mat.col(j) = u_2_mat.col(j) + Theta_2_j - z_mat.col(j);
//     }
//     z = z_mat.rowwise().sum();
//     
//     // 3. u_1-Update (scaled dual update)
//     u_1 = u_1 + delta.transpose() * beta - 1.0;
//     
//     // 4. u_2-Update (scaled dual update)
//     u_2 = u_2_mat.rowwise().sum();
//     
//     
//     // --- Convergence Check ---
//     MatrixXd A(n + 1, p);
//     A.row(0) = delta;
//     A.block(1, 0, n, p) = Theta_2;
//     MatrixXd B(n + 1, n);
//     B.row(0).setZero();
//     B.block(1, 0, n, n) = -MatrixXd::Identity(n, n);
//     VectorXd c = VectorXd::Zero(n + 1);
//     c(0) = 1;
//     
//     // Primal and dual residuals
//     double r_primal = (A * beta + B * z - c).norm();
//     double r_dual = rho * (Theta_2.transpose() * (z - z_old)).norm();
//     
//     // Thresholds
//     double eps_primal = std::sqrt(n + 1.0) * tol_abs + tol_rel * std::max({
//       (A * beta).norm(), (B * z).norm(), c.norm()
//     });
//     double eps_dual = std::sqrt(p) * tol_abs + tol_rel * (Theta_2.transpose() * u_2).norm();
//     
//     // Check if ALL three conditions are met
//     if (r_primal < eps_primal && r_dual < eps_dual) {
//       // Rcpp::Rcout << "Converged at iteration " << t + 1 << std::endl;
//       break;
//     }
//     
//     if (t == max_iter -1){
//       Rcpp::Rcout << "Warning: ADMM did not converge within max_iter=" << max_iter << "!" << std::endl;
//     }
//   }
// }
// 
// 
// 
// // --- Main LLA Solver for Group SCAD (Warm Start) ---
// 
// // [[Rcpp::export]]
// Rcpp::List group_lla_admm_warm_start(const Eigen::MatrixXd& Sigma,
//                                      const Eigen::VectorXd& delta,
//                                      int s, // group size
//                                      const Eigen::MatrixXd& Theta,
//                                      double lambda = 0.1,
//                                      double gamma = 3.7,
//                                      int max_iter_lla = 100,
//                                      int max_iter_admm = 100,
//                                      double rho = 1.0,
//                                      double tol = 1e-5,
//                                      double tol_abs = 1e-4,
//                                      double tol_rel = 1e-3) {
//   int n = Theta.rows();
//   int p = Sigma.rows();
//   if (p % s != 0) {
//     Rcpp::stop("Total number of variables p must be divisible by group size s.");
//   }
//   int num_groups = p / s;
//   
//   // --- WARM START: All state variables are declared and initialized here ---
//   // Initialize variables for admm_solver
//   VectorXd beta_hat = VectorXd::Zero(p);
//   // VectorXd beta_hat = VectorXd::Constant(p, 1.0 / p);
//   VectorXd z = VectorXd::Zero(n);
//   VectorXd u_2 = VectorXd::Zero(n); // scaled dual variable for n^(-1/2)*Theta*beta - z = 0
//   double u_1 = 0.0;               // scaled dual variable for delta^T*beta - 1 = 0
//   MatrixXd z_mat = MatrixXd::Zero(n, p);  // for each covariates; rowSums(z_mat) = z
//   MatrixXd u_2_mat = MatrixXd::Zero(n, p);  // for each covariates; rowSums(u_2_mat) = u_2
//   
//   // --- Cholesky decomposition objects for inverse computations ---
//   // Pre-compute M_inv for the beta-update (outer ADMM)
//   MatrixXd M = Sigma + rho/n * Theta.transpose() * Theta + rho * delta * delta.transpose();
//   // double scale = M.diagonal().mean() * 1e-8; // to avoid singularity
//   double scale = 1e-8; // to avoid singularity
//   MatrixXd M_reg = M + scale * MatrixXd::Identity(p, p);
//   auto M_chol = M_reg.llt();
//   
//   
//   std::vector<double> error_conv;
//   int iter;
//   for (iter = 0; iter < max_iter_lla; ++iter) {
//     VectorXd beta_hat_old = beta_hat;
//     
//     // Compute group norm
//     VectorXd norm_beta_g(num_groups);
//     for (int g = 0; g < num_groups; ++g) {
//       // Use .segment() to get a view of the g-th group without copying
//       norm_beta_g(g) = (Theta.block(0, g * s, n, s) * beta_hat.segment(g * s, s)).norm() / sqrt(n);
//     }
//     
//     // Compute weights using the block-based helper function
//     VectorXd group_weights = scad_deriv(norm_beta_g, lambda, gamma);
//     
//     // Call admm_solver, passing all state variables for a full warm start
//     admm_solver_warm_start(beta_hat, z, u_2, u_1, z_mat, u_2_mat,
//                            Sigma, delta, group_weights, Theta, s, M_chol,
//                            max_iter_admm, rho, tol_abs, tol_rel);
//     
//     // Check the convergence
//     double beta_diff = (beta_hat - beta_hat_old).norm() / (beta_hat_old.norm() + 1e-8);
//     error_conv.push_back(beta_diff);
//     if (beta_diff < tol) {
//       break;
//     }
//     
//     if (iter == max_iter_lla -1){
//       Rcpp::Rcout << "Warning: LLA did not converge within max_iter_lla=" << max_iter_lla << "!" << std::endl;
//     }
//   }
//   
//   return Rcpp::List::create(
//     Rcpp::Named("beta") = beta_hat,
//     Rcpp::Named("iterations") = iter + 1,
//     Rcpp::Named("error_conv") = error_conv
//   );
// }