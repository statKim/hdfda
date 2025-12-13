#' Estimate the Optimal Projection Direction for High-dimensional Functional Data with Group Non-convex Penalty on Two-sample Test problem
#'
#' The optimal projection direction is corresponding to the high-dimensional functional linear discriminant direction under the sparsity assumption.
#' The optimal projection direction is estimated by the block coordinate descent (BCD) algorithm with local quadratic approximation (LQA).
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param grid a vector containing m timepoints
#' @param penalty the penalty type ("scad" is only supported.)
#' @param basis Default is "bspline" ("fpca" is possible but takes huge times if p is high)
#' @param n_basis the number of cubic B-spline bases using `n_basis`-2 knots
#' @param lambda a penalty parameter for L1-regularization
#' @param a a parameter for SCAD penalty
#' @param max_iter a maximum iteration number of the LQA algorithm
#' @param sweeps_bcd a number of sweeps in BCD algorithm
#'
#' @return a `opt_proj_dir` object
#'
#' @references Kim, H. and Park, J. (2025+). Two-sample Projection Test for High-Dimensional Functional Data, Submitted.
#'
#' @importFrom stats cov
#' @export
opt_proj_dir <- function(X,
                         y,
                         grid = NULL,
                         penalty = "scad",
                         basis = "bspline",
                         n_basis = 4,
                         lambda = 0.1,
                         a = 3.7,
                         max_iter = 1000,
                         sweeps_bcd = 2) {
  n <- dim(X)[1]   # number of curves
  m <- dim(X)[2]   # number of timepoints
  p <- dim(X)[3]   # number of variables

  # Basis representation for each functional covariate
  basis_obj <- basis_mfd(X,
                         grid = grid,
                         basis = basis,
                         n_basis = n_basis,
                         gram = TRUE)
  X_coef <- basis_obj$X_coef

  # Observed grid points
  grid <- basis_obj$grid

  # Group indicator for each functional covariate
  groups <- basis_obj$groups

  # Index set
  idx_g1 <- which(y == 0)
  idx_g2 <- which(y == 1)
  n1 <- length(idx_g1)
  n2 <- length(idx_g2)

  # Mean of each basis coefficient
  mu1 <- colMeans(X_coef[idx_g1, ])
  mu2 <- colMeans(X_coef[idx_g2, ])
  delta <- mu2 - mu1  # check the threshold!!
  # (mu2 - mu1 => assign g2 if X\beta > 0)
  # (mu1 - mu2 => assign g2 if X\beta < 0)

  # Pooled sample covariance
  S1 <- cov(X_coef[idx_g1, ])
  S2 <- cov(X_coef[idx_g2, ])
  # S <- ((n1-1)*S1 + (n2-1)*S2) / (n-2)
  S <- ((n1-1)*S1 + (n2-1)*S2) / n

  # Prior probability
  pi1 <- n1 / n
  pi2 <- 1 - pi1

  # Centering the basis coefficient matrix (opt_eq = 1/n...)
  z <- ifelse(y == 1, pi1, -pi2)
  X_coef_c <- scale(X_coef, center = T, scale = F)

  # Optimize the coeffient beta_hat
  beta_init <- rep(0, ncol(S))
  if (penalty == "scad") {
    fit_obj <- solve_mm_bcd(
      X = X_coef_c,
      y = z,
      Lambda_diag = sqrt(diag(S)),
      beta_init = beta_init,
      p = p,
      s = n_basis,
      lambda = lambda,
      a = a,
      max_iter_outer = max_iter,
      sweeps_inner = sweeps_bcd
    )
  } else {
    stop("penalty = `scad` is only supported!")
  }
  beta_hat <- fit_obj$beta

  # Hard-thresholding for small values
  beta_hat[which(abs(beta_hat) < 1e-4)] <- 0  # hard-thresholding

  # Active set from the sparse solution
  active_set <- which(sapply(1:p, function(j){ sum(beta_hat[which(groups == j)]^2) }) > 0)

  ### Discrimination is only valid under Gaussian assumption
  # Obtain the discrimination vector and discrimination threshold
  if (length(active_set) == 0) {
    # stop("All zero coefficients are obtained!")
    threshold <- 0
    idx <- 1:length(beta_hat)
  } else {
    idx <- which(groups %in% active_set)
    threshold <- as.numeric( (t(beta_hat[idx]) %*% S[idx, idx] %*% beta_hat[idx]) / (t(beta_hat[idx]) %*% (mu2 - mu1)[idx]) * log(n1/n2) )
  }

  # Obtain training accuracy
  X_coef_c2 <- apply(X_coef[, idx], 1, function(row){ row - (mu1[idx] + mu2[idx])/2 })
  X_coef_c2 <- t(X_coef_c2)
  pred <- as.integer(ifelse(X_coef_c2 %*% beta_hat[idx] > threshold, 1, 0))
  err_train <- mean(y != pred)   # training error

  # Compute ABIC
  z_hat <- as.numeric(X_coef_c[, idx] %*% beta_hat[idx])
  rss <- sum((z - z_hat)^2)
  g_s_lambda <- length(active_set)
  df <- sum(apply(X_coef[, idx], 2, stats::var))
  abic <- log(rss) + 2*g_s_lambda*n_basis/n + df*log(n)/n

  # Compute AIC
  aic <- log(rss) + 2*g_s_lambda*n_basis/n

  # Prior estimates
  estimates <- list(
    mu1 = mu1,
    mu2 = mu2,
    pi1 = pi1,
    pi2 = pi2
  )

  # Output object
  res <- list(
    beta_hat = beta_hat,   # sparse solution
    active_set = active_set,   # selected active set
    n_basis = basis_obj$n_basis,
    basis_obj = basis_obj,
    groups = groups,
    penalty = penalty,
    lambda = lambda,
    threshold = threshold,   # threshold of discrimination rule
    estimates = estimates,
    abic = abic,
    aic = aic,
    pred_train = pred,
    err_train = err_train
  )
  class(res) <- "opt_proj_dir"

  return(res)
}



#' Predict the class of new data from `opt_proj_dir` object
#'
#' The prediction is obtained using the high-dimensional LDA direction with group non-convex penalty (i.e., optimal projection direction) under the Gaussian assumption
#'
#' @param object a `opt_proj_dir` object
#' @param newdata a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param ... Not used
#'
#' @return a `opt_proj_dir` object
#'
#' @importFrom stats predict
#' @export
predict.opt_proj_dir <- function(object, newdata, ...) {
  # Make basis coefficient matrix
  X_coef_test <- predict(object$basis_obj, newdata)

  # Estimated sparse solutions
  beta <- object$beta_hat
  # Non-zero indices of active coefficient vector
  idx <- which(abs(beta) > 1e-8)

  # Prediction
  X_coef_test_c2 <- apply(X_coef_test[, idx, drop=FALSE], 1, function(row){ row - (object$estimates$mu1[idx] + object$estimates$mu2[idx])/2 })
  if (length(idx) == 1) {
    X_coef_test_c2 <- matrix(X_coef_test_c2, ncol = 1)
  } else {
    X_coef_test_c2 <- t(X_coef_test_c2)
  }
  pred <- as.integer(ifelse(X_coef_test_c2 %*% beta[idx] > object$threshold, 1, 0))

  return(pred)
}


#' Tuning Parameter election for the Estimation of Optimal Projection Direction for High-dimensional Functional Data
#'
#' Tuning parameter (lambda, n_basis) selection for `opt_proj_dir`
#'
#' Select the optimal `n_basis` and `lambda` for `opt_proj_dir` using K-fold cross-validation with accuracy and Mahalanobis distance or ABIC, ABIC-AIC.
#' Default tuning method is K-fold cross-validation with a measure classification error rate.
#' Parallel computing can be used by using the `doParallel` package usages.
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param tune_method "cv", "abic" or "abic-aic" (default is "cv")
#' @param grid a vector containing m timepoints
#' @param penalty the penalty type ("scad" is only supported)
#' @param basis Default is "bspline" ("fpca" is possible but takes huge times if p is high)
#' @param n_basis_list a vector containing the candidate of `n_basis` (the number of cubic B-spline bases using `n_basis`-2 knots)
#' @param lambda_list a vector containing the candidate of `lambda` (a penalty parameter for L1-regularization)
#' @param measure the measure for the K-fold cross-validation. "accuracy", "cross.entropy", "mahalanobis" (Default is "accuracy")
#' @param K the number of folds for K-fold CV
#' @param tie_break the tie breaking rule for cross-validation. "sparse"(default) choose the largest `lambda` and the smallest `n_basis`; "random" choose randomly
#' @param ... additional parameters for `opt_proj_dir`
#'
#' @return a `opt_proj_dir` object
#'
#' @references Kim, H. and Park, J. (2025+). Two-sample Projection Test for High-Dimensional Functional Data, Submitted.
#'
#' @importFrom foreach %dopar% foreach
#' @importFrom stats median t.test
#' @export
tune.opt_proj_dir <- function(X,
                              y,
                              tune_method = "cv",
                              grid = NULL,
                              penalty = "scad",
                              basis = "bspline",
                              n_basis_list = NULL,
                              lambda_list = NULL,
                              measure = "accuracy",
                              K = 5,
                              tie_break = "sparse",
                              ...) {
  n <- dim(X)[1]   # number of curves
  m <- dim(X)[2]   # number of timepoints
  p <- dim(X)[3]   # number of variables

  # Candidates of grid search
  if (is.null(n_basis_list)) {
    n_basis_list <- 4:8
  }
  if (is.null(lambda_list)) {
    lambda_list <- seq(1e-3, 0.1, length.out = 10)
  }
  cand_tune <- expand.grid(n_basis = n_basis_list,
                           lambda = lambda_list)

  # Hyperparameter tuning
  if (tune_method == "cv") {
    # K-fold cross-validation
    fold_list <- rep(NA, n)
    fold_list[y == 0] <- sample(1:sum(y == 0) %% K + 1)
    fold_list[y == 1] <- sample(1:sum(y == 1) %% K + 1)
    # fold_list <- sample(1:K, n, replace = T)
    i <- NULL
    loss_list <- foreach::foreach(i = 1:nrow(cand_tune),
                                  .packages=c("fda","hdfda"),
                                  .export = c("opt_proj_dir","predict.opt_proj_dir"),
                                  .combine = c,
                                  .errorhandling = "pass") %dopar% {
      loss_i <- rep(NA, K)
      n_basis <- cand_tune[i, 1]
      lambda <- cand_tune[i, 2]
      for (j in 1:K) {
        # Split data
        X_train <- X[fold_list != j, , , drop=FALSE]
        X_test <- X[fold_list == j, , , drop=FALSE]
        y_train <- y[fold_list != j]
        y_test <- y[fold_list == j]

        # Fit optimal projection direction
        fit_obj <- opt_proj_dir(X_train,
                                y_train,
                                grid = grid,
                                penalty = penalty,
                                basis = basis,
                                n_basis = n_basis,
                                lambda = lambda, ...)

        # Validation error
        if (measure == "accuracy") {
          # Misclassification error rate (we divide by n after the all folds are run)
          pred <- predict(fit_obj, X_test)  # prediction of validation set
          loss_i[j] <- sum(y_test != pred)
        } else if (measure == "cross.entropy") {
          # Cross-entropy loss
          pred <- predict(fit_obj, X_test)  # prediction of validation set
          loss_i[j] <- -sum( log(pred[y_test == 1]) ) - sum( log(1-pred[y_test == 0]) )
        } else if (measure == "mahalanobis") {
          # Mahalanobis distance of 1-dimensional data
          # = Hotellings t^2 statistic
          # = t statistics^2
          beta <- fit_obj$beta_hat   # optimal projection direction
          if (sum(beta) == 0) {
            # If there is no active_set, we can obtain same result onto any dirction
            beta <- rep(1/length(beta), length(beta))
          }
          X_coef_test <- predict(fit_obj$basis_obj, X_test)
          Y <- as.numeric(X_coef_test %*% beta)
          # proj_center <- sum((fit_obj$estimates$mu1 + fit_obj$stimates$mu2)/2 * beta)

          # Compute Mahalanobis distance = - (t statistic)^2
          obj_test <- t.test(Y[y_test == 0],
                             Y[y_test == 1],
                             var.equal = TRUE)  # two-sided test
          loss_i[j] <- -obj_test$statistic^2
        }

      }

      loss_i <- sum(loss_i) / n
      return(loss_i)
    }

    # Optimal hyperparameters
    cand_tune$cv_error <- loss_list
    hyperparam_ties <- cand_tune[which(cand_tune$cv_error == min(cand_tune$cv_error)), 1:2, drop=FALSE]
    if (nrow(hyperparam_ties) > 1) {
      if (tie_break == "sparse") {
        # If there exist ties, we choose larger lambda and lower n_basis for the sparse solution
        hyperparam_ties <- hyperparam_ties[which(hyperparam_ties$lambda == max(hyperparam_ties$lambda)), , drop=FALSE]
        if (nrow(hyperparam_ties) > 1) {
          # Ties for n_basis with the same lambda => choose lower n_basis
          hyperparam_ties <- hyperparam_ties[which.min(hyperparam_ties$n_basis), , drop=FALSE]
        }
      } else if (tie_break == "random"){
        # If there exist ties, we randomly choose the combination.
        hyperparam_ties <- hyperparam_ties[sample(1:nrow(hyperparam_ties), 1), , drop=FALSE]
      }
    }
    n_basis <- hyperparam_ties[, 1]
    lambda <- hyperparam_ties[, 2]

    # Fit `opt_proj_dir()` using the optimal parameters
    fit <- opt_proj_dir(X,
                        y,
                        grid = grid,
                        penalty = penalty,
                        basis = basis,
                        n_basis = n_basis,
                        lambda = lambda, ...)

  } else if (tune_method == "abic") {
    # Fit optimal projection direction for each hyperparemeter candidate
    i <- NULL
    fit_list <- foreach::foreach(i = 1:nrow(cand_tune),
                                 .packages=c("fda","hdfda"),
                                 .export = c("opt_proj_dir"),
                                 .errorhandling = "pass") %dopar% {
       fit_obj <- opt_proj_dir(X,
                               y,
                               grid = grid,
                               penalty = penalty,
                               basis = basis,
                               n_basis = cand_tune[i, 1],
                               lambda = cand_tune[i, 2],
                               ...)
       return(fit_obj)
     }

    # Optimal hyperparameters
    cand_tune$abic <- sapply(fit_list, function(fit_obj){ fit_obj$abic })
    hyperparam_ties <- cand_tune[which(cand_tune$abic == min(cand_tune$abic)), 1:2, drop=FALSE]
    if (nrow(hyperparam_ties) > 1) {
      if (tie_break == "sparse") {
        # If there exist ties, we choose larger lambda and lower n_basis for the sparse solution
        hyperparam_ties <- hyperparam_ties[which(hyperparam_ties$lambda == max(hyperparam_ties$lambda)), , drop=FALSE]
        if (nrow(hyperparam_ties) > 1) {
          # Ties for n_basis with the same lambda => choose lower n_basis
          hyperparam_ties <- hyperparam_ties[which.min(hyperparam_ties$n_basis), , drop=FALSE]
        }
      } else if (tie_break == "random"){
        # If there exist ties, we randomly choose the combination.
        hyperparam_ties <- hyperparam_ties[sample(1:nrow(hyperparam_ties), 1), , drop=FALSE]
      }
    }
    n_basis <- hyperparam_ties[, 1]
    lambda <- hyperparam_ties[, 2]

    # Choose the optimal object
    fit <- fit_list[[which(cand_tune$n_basis == n_basis & cand_tune$lambda == lambda)]]
  } else if (tune_method == "abic_aic") {
    # Find optimal lambda for fixed n_basis using ABIC (n_basis = median(n_basis_list))
    i <- NULL
    fit_list <- foreach::foreach(i = 1:length(lambda_list),
                                 .packages=c("fda","hdfda"),
                                 .export = c("opt_proj_dir"),
                                 .errorhandling = "pass") %dopar% {
       fit_obj <- opt_proj_dir(X,
                               y,
                               grid = grid,
                               penalty = penalty,
                               basis = basis,
                               n_basis = median(n_basis_list),
                               lambda = lambda_list[i],
                               ...)
       return(fit_obj)
     }

    # Optimal lambda
    abic <- sapply(fit_list, function(fit_obj){ fit_obj$abic })
    cand_tune$abic <- NA
    cand_tune$abic[cand_tune$n_basis == median(n_basis_list)] <- abic
    hyperparam_ties <- lambda_list[which(abic == min(abic))]
    if (length(hyperparam_ties) > 1) {
      if (tie_break == "sparse") {
        # If there exist ties, we choose larger lambda for the sparse solution
        hyperparam_ties <- hyperparam_ties[which(hyperparam_ties == max(hyperparam_ties))]
      } else if (tie_break == "random"){
        # If there exist ties, we randomly choose the combination.
        hyperparam_ties <- hyperparam_ties[sample(1:length(hyperparam_ties), 1)]
      }
    }
    lambda <- hyperparam_ties

    # Find optimal n_basis for optimal lambda using AIC
    i <- NULL
    fit_list <- foreach::foreach(i = 1:length(n_basis_list),
                                 .packages=c("fda","hdfda"),
                                 .export = c("opt_proj_dir"),
                                 .errorhandling = "pass") %dopar% {
       fit_obj <- opt_proj_dir(X,
                               y,
                               grid = grid,
                               penalty = penalty,
                               basis = basis,
                               n_basis = n_basis_list[i],
                               lambda = lambda,
                               ...)
       return(fit_obj)
     }

    # Optimal n_basis
    aic <- sapply(fit_list, function(fit_obj){ fit_obj$aic })
    cand_tune$aic <- NA
    cand_tune$aic[cand_tune$lambda == lambda] <- aic
    hyperparam_ties <- n_basis_list[which(aic == min(aic))]
    if (length(hyperparam_ties) > 1) {
      if (tie_break == "sparse") {
        # If there exist ties, we choose lower n_basis for the sparse solution
        hyperparam_ties <- hyperparam_ties[which(hyperparam_ties == min(hyperparam_ties))]
      } else if (tie_break == "random"){
        # If there exist ties, we randomly choose the combination.
        hyperparam_ties <- hyperparam_ties[sample(1:length(hyperparam_ties), 1)]
      }
    }
    n_basis <- hyperparam_ties

    # Choose the optimal object
    fit <- fit_list[[which(n_basis_list == n_basis)]]
  }

  # Final object
  tune_obj <- list(
    opt_fit = fit,
    opt_params = c(n_basis = n_basis,
                   lambda = lambda),
    tune_error = cand_tune
  )

  return(tune_obj)
}

