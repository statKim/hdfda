#include <RcppEigen.h>
#include <vector>

// [[Rcpp::depends(RcppEigen)]]

/**
 * Helper function to compute SCAD weight
 * w(u) = rho'(u) / (2u)
 * u = ||Lambda_j * beta_j||_2
 */
double compute_scad_weight(double u, double lambda, double a, double epsilon) {
  if (u < epsilon) {
    // If u is close to 0, treat similarly to Lasso's majorization
    // rho(u) ~ lambda * u. w(u) ~ lambda / (2u)
    // Use epsilon to avoid division by zero
    return lambda / (2.0 * epsilon);
  } else if (u <= lambda) {
    // Lasso region
    return lambda / (2.0 * u);
  } else if (u <= a * lambda) {
    // Clipped region
    return (a * lambda - u) / (2.0 * u * (a - 1.0));
  } else {
    // Flat region
    return 0.0;
  }
}

/**
 * MM (LQA) + Block Coordinate Descent (BCD) Algorithm
 */
// [[Rcpp::export]]
Rcpp::List solve_mm_bcd(
    const Eigen::Map<Eigen::MatrixXd>& X,     // n x (ps) design matrix
    const Eigen::Map<Eigen::VectorXd>& y,     // n x 1 response vector
    const Eigen::Map<Eigen::VectorXd>& Lambda_diag, // (ps) x 1 concatenated diag vectors
    Eigen::VectorXd beta_init,             // (ps) x 1 initial beta
    int p,                                 // Number of groups
    int s,                                 // Group size
    double lambda,                         // SCAD lambda
    double a = 3.7,                        // SCAD 'a' parameter (usually 3.7)
    int max_iter_outer = 1000,              // Max iterations for Outer loop (MM)
    int sweeps_inner = 2,                  // Number of sweeps for Inner loop (BCD)
    double tol_outer = 1e-5,               // Convergence tolerance for Outer loop
    double epsilon = 1e-6                  // Small value for numerical stability
) {
  double n = X.rows();
  Eigen::VectorXd beta = beta_init;
  Eigen::VectorXd beta_old(beta.size());
  Eigen::VectorXd weights(p);

  // --- Pre-calculation 2: (1/n) * X_j^T * X_j blocks (s x s) ---
  // Pre-compute the most expensive part of the BCD update.
  std::vector<Eigen::MatrixXd> XtX_blocks(p);
  for(int j = 0; j < p; ++j) {
    XtX_blocks[j] = (X.block(0, j * s, n, s).transpose() * X.block(0, j * s, n, s)) / n;
  }

  // --- Initialize full residual for BCD ---
  Eigen::VectorXd r = y - X * beta;

  bool converged = false;

  // ===================================
  // === Outer Loop (MM Algorithm) ===
  // ===================================
  for (int t = 0; t < max_iter_outer; ++t) {
    beta_old = beta;

    // --- 1. Update Weights (MM Step) ---
    for (int j = 0; j < p; ++j) {
      Eigen::VectorXd beta_j = beta.segment(j * s, s);

      // Get diag_j vector from the long vector using .segment()
      Eigen::VectorXd diag_j = Lambda_diag.segment(j * s, s);

      // Calculate u_j = ||Lambda_j * beta_j||_2
      double u_j = (diag_j.asDiagonal() * beta_j).norm();

      weights(j) = compute_scad_weight(u_j, lambda, a, epsilon);
    }

    // ===================================
    // === Inner Loop (BCD Algorithm) ===
    // ===================================
    // (Solves the weighted ridge regression problem using BCD)
    for (int k = 0; k < sweeps_inner; ++k) {
      for (int j = 0; j < p; ++j) {
        Eigen::VectorXd beta_j_old = beta.segment(j * s, s);

        // 1. Calculate partial residual
        // r_j = r + X_j * beta_j_old (r is the current full residual)
        r += X.block(0, j * s, n, s) * beta_j_old;

        // 2. Calculate RHS: (1/n) * X_j^T * r_j
        Eigen::VectorXd rhs = (X.block(0, j * s, n, s).transpose() * r) / n;

        // 3. Calculate LHS: (1/n)X_j^T X_j + 2 * w_j * Lambda_j^2
        // Get diag_j vector and compute Lambda_j^2 on the fly
        Eigen::VectorXd diag_j = Lambda_diag.segment(j * s, s);
        Eigen::MatrixXd Lambda_j_sq_mat = (diag_j.array().square()).matrix().asDiagonal();
        Eigen::MatrixXd LHS = XtX_blocks[j] + 2.0 * weights(j) * Lambda_j_sq_mat;

        // 4. Solve the (s x s) linear system: LHS * beta_j_new = rhs
        Eigen::VectorXd beta_j_new = LHS.llt().solve(rhs); // Use LLT (Cholesky)

        // 5. Update beta
        beta.segment(j * s, s) = beta_j_new;

        // 6. Update full residual: r = r - X_j * beta_j_new
        r -= X.block(0, j * s, n, s) * beta_j_new;
      }
    } // End Inner Loop

    // --- Outer Loop Convergence Check ---
    double change = (beta - beta_old).norm() / (beta_old.norm() + 1e-8);
    if (change < tol_outer) {
      converged = true;
      return Rcpp::List::create(
        Rcpp::Named("beta") = beta,
        Rcpp::Named("weights") = weights,
        Rcpp::Named("iterations") = t + 1,
        Rcpp::Named("converged") = converged
      );
    }
  } // End Outer Loop

  Rcpp::warning("Outer loop (MM) did not converge after %d iterations.", max_iter_outer);
  return Rcpp::List::create(
    Rcpp::Named("beta") = beta,
    Rcpp::Named("weights") = weights,
    Rcpp::Named("iterations") = max_iter_outer,
    Rcpp::Named("converged") = converged
  );
}
