#' Two-sample Projection Test for High-Dimensional Functional Data
#'
#' The data-splitting projection test onto the optimal projection direction for high-dimensional functional data
#' The optimal projection direction is corresponding to the high-dimensional functional linear discriminant direction under the sparsity assumption.
#' The optimal projection direction is estimated by the block coordinate descent (BCD) algorithm with local quadratic approximation (LQA).
#'
#' @param X1 a n1-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param X2 a n2-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param penalty the default penalty type is "scad" (group SCAD). ("l1" is also supported which is the `hdflda` direction)
#' @param split_prop the proportion of data-splitting (split_prop for estimating the optimal projection direction; 1-split_prop for test). Default is 0.5.
#' @param tuning If `tuning = TRUE`, the parameter tuning is performed. (`n_basis`, `lambda`)
#' @param tune_method "cv", "abic" and "abic-aic" are supported. Default is "cv".
#' @param n_basis the number of cubic B-spline bases using `n_basis`-2 knots
#' @param lambda a penalty parameter for penalty function
#' @param n_cores a number cores for parallel computing
#' @param ... additional parameters or `opt_proj_dir` ("scad") or `hdflda` ("l1")
#'
#' @return a `htest` object
#'
#' @references Kim, H. and Park, J. (2025+). Two-sample Projection Test for High-Dimensional Functional Data, Submitted.
#'
#' @importFrom stats pnorm
#'
#' @export
hdfd_proj_test <- function(X1,
                           X2,
                           penalty = "scad",
                           split_prop = 0.5,
                           tuning = FALSE,
                           tune_method = "cv",
                           n_basis = 4,
                           lambda = 0.1,
                           n_cores = 1, ...) {
  n1 <- dim(X1)[1]
  n2 <- dim(X2)[1]
  m <- dim(X1)[2]
  p <- dim(X1)[3]
  data_name <- paste(deparse(substitute(X1)), "and", deparse(substitute(X2)))

  # Split data
  n1_train <- ceiling(n1 * split_prop)
  n2_train <- ceiling(n2 * split_prop)
  idx_n1_train <- sample(n1, n1_train)
  idx_n2_train <- sample(n2, n2_train)

  X_train <- array(NA, c(n1_train + n2_train, m, p))
  X_train[1:n1_train, , ] <- X1[idx_n1_train, , ]
  X_train[(n1_train+1):(n1_train+n2_train), , ] <- X2[idx_n2_train, , ]
  y_train <- c(rep(0, n1_train), rep(1, n2_train))

  # Estimate optimal projection direction based on functional LDA using penalized LSE
  if (penalty == "l1") {
    # High-dim functional LDA (LSE-LDA + L1 penalty)
    if (isTRUE(tuning)) {
      # Parallel computing
      cl <- parallel::makePSOCKcluster(n_cores)
      doParallel::registerDoParallel(cl)
      obj_proj_tune <- tune.hdflda(X_train,
                                   y_train,
                                   tune_method = tune_method,
                                   ...)
      obj_proj <- obj_proj_tune$opt_fit
      parallel::stopCluster(cl)
    } else {
      obj_proj <- hdflda(X_train,
                         y_train,
                         n_basis = n_basis,
                         lambda = lambda, ...)
    }
    beta <- obj_proj$nu_hat   # optimal projection direction
  } else if (penalty == "scad") {
    # LSE-LDA + group SCAD penalty
    if (isTRUE(tuning)) {
      # Parallel computing
      cl <- parallel::makePSOCKcluster(n_cores)
      doParallel::registerDoParallel(cl)
      obj_proj_tune <- tune.opt_proj_dir(X_train,
                                         y_train,
                                         tune_method = tune_method,
                                         ...)
      obj_proj <- obj_proj_tune$opt_fit
      parallel::stopCluster(cl)
    } else {
      obj_proj <- opt_proj_dir(X_train,
                               y_train,
                               n_basis = n_basis,
                               lambda = lambda, ...)
    }
    beta <- obj_proj$beta_hat   # optimal projection direction
  }

  if (sum(beta) == 0) {
    # If there is no active_set, we can obtain same result onto any dirction
    beta <- rep(1/length(beta), length(beta))
  }

  # Projection the remaining data onto estimated optimal projection direction
  X1_coef_test <- predict(obj_proj$basis_obj, X1[-idx_n1_train, , ])
  X2_coef_test <- predict(obj_proj$basis_obj, X2[-idx_n2_train, , ])
  Y1 <- as.numeric(X1_coef_test %*% beta)
  Y2 <- as.numeric(X2_coef_test %*% beta)

  # Compute test statistics (t-test) under the equal variances
  obj_test <- t.test(Y1, Y2, var.equal = TRUE)  # two-sided test
  test_stat <- obj_test$statistic

  # P-value of two-sided test (asymptotic normality)
  p_value <- pnorm(abs(test_stat))
  p_value <- 2 * min(p_value, 1-p_value)

  # Output
  out <- list(
    statistic = test_stat,
    p.value = p_value,
    method = "Two-Sample Projection Test for High-Dimensional Functional Means",
    data.name = data_name,
    obj_test = obj_test,
    obj_proj = obj_proj
  )
  if (isTRUE(tuning)) {
    out$obj_proj_tune <- obj_proj_tune
  }
  class(out) <- "htest"

  return(out)
}

