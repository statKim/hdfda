#' LDA for high-dimensional functional data
#'
#' Linear discriminant analysis for high-dimensional functional data
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param grid a vector containing m timepoints
#' @param basis "bspline" is only supported
#' @param n_basis the number of cubic B-spline bases using `n_basis`-2 knots
#' @param lambda a penalty parameter for L1-regularization
#' @param tol a tolerance rate to define the sparse discriminant set
#'
#' @return a `hdflda` object
#'
#' @references Xue, K., Yang, J., & Yao, F. (2023). Optimal linear discriminant analysis for high-dimensional functional data. Journal of the American Statistical Association, 1-10.
#'
#' @importFrom stats cov
#' @export
hdflda <- function(X, y,
                   grid = NULL,
                   basis = "bspline",
                   n_basis = 4,
                   lambda = 0.1,
                   tol = 1e-7) {
  n <- dim(X)[1]   # number of curves
  m <- dim(X)[2]   # number of timepoints
  p <- dim(X)[3]   # number of variables

  # Basis representation for each functional covariate
  basis_obj <- basis_mfd(X,
                         grid = grid,
                         basis = basis,
                         # FVE = FVE,
                         # K = K,
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
  n <- nrow(X_coef)

  # Prior probability
  pi1 <- n1 / n
  pi2 <- 1 - pi1

  # Mean of each basis coefficient
  mu <- colMeans(X_coef)
  mu1 <- colMeans(X_coef[idx_g1, ])
  mu2 <- colMeans(X_coef[idx_g2, ])

  # Pooled sample covariance
  S1 <- cov(X_coef[idx_g1, ])
  S2 <- cov(X_coef[idx_g2, ])
  S <- ((n1-1)*S1 + (n2-1)*S2) / (n-2)

  # Diagonal elements of covariance of basis coefficients
  omega <- diag(S)

  # Estimate of nu^(1)
  nu_1 <- mu2 - mu1

  # Optimize using equation (13) using "glmnet"
  z <- ifelse(y == 1, pi1, -pi2) * sqrt(n / (n-2))
  X_coef_c <- scale(X_coef, center = T, scale = F) %*% diag(1 / sqrt(omega)) * sqrt(n / (n-2))

  glmnet.obj <- glmnet::glmnet(X_coef_c,
                               z,
                               family = "gaussian",
                               alpha = 1,
                               lambda = lambda,
                               standardize = F,
                               intercept = F)
  nu_hat <- as.numeric(glmnet.obj$beta) / sqrt(omega)
  # sum(nu_hat)

  # # Optimize using equation (13) using "CVXR"
  # z <- ifelse(y == 1, pi1, -pi2)
  # X_coef_c <- scale(X_coef, center = T, scale = F)
  # nu <- CVXR::Variable(ncol(X_coef))
  # obj <- CVXR::sum_squares(z - X_coef_c %*% nu) / (2*(n-2)) + lambda*CVXR::p_norm(sqrt(omega) * nu, 1)
  # # lambda_n <- lambda*sqrt(omega)
  # # obj <- sum((z - X_coef_c %*% nu)^2) / (2*(n-2)) + p_norm(lambda_n*nu, 1)
  # prob <- CVXR::Problem(CVXR::Minimize(obj))
  # result <- CVXR::psolve(prob, verbose = F)
  #
  # if (result$status != "optimal") {
  #   warnings("The solution is not optimal!")
  # }
  #
  # # Optimal estimate of nu
  # nu_hat <- result$getValue(nu)
  # nu_hat <- as.numeric(nu_hat)
  # nu_hat[nu_hat < tol] <- 0   # thresholding to obtain sparse solution
  # # nu_hat_13 <- nu_hat
  #
  # # # Very slow!!
  # # # Optimize using equation (14)
  # # nu <- Variable(ncol(X_coef))
  # # obj <- quad_form(nu, (S + n1*n2/(n*(n-2)) * outer(nu_1, nu_1) )) / 2 - n1*n2/(n*(n-2)) * sum(nu * nu_1) + lambda*p_norm(sqrt(omega) * nu, 1)
  # # prob <- Problem(Minimize(obj))
  # # result <- psolve(prob, verbose = T)
  # # result$status
  # # nu_hat <- result$getValue(nu)
  # # nu_hat_14 <- nu_hat
  # #
  # # sum((nu_hat_13 - nu_hat_14)^2)   # similar results

  # Obtain the discrimination vector and discrimination threshold
  # L2 norm of the functional covariates
  nu_hat_l2norm <- sapply(1:p, function(j){
    idx <- which(groups == j)
    sqrt(sum(nu_hat[idx]^2))
  })
  # Sparse discriminant set
  discrim_set_idx <- which(nu_hat_l2norm > 0)
  if (length(discrim_set_idx) == 0) {
    # stop("All zero coefficients are obtained!")
    threshold <- 0
    idx <- 1:length(nu_hat)
  } else {
    idx <- which(groups %in% discrim_set_idx)
    # nu_hat[-idx] <- 0   # sparse solution
    threshold <- as.numeric( (t(nu_hat[idx]) %*% S[idx, idx] %*% nu_hat[idx]) / (t(nu_hat[idx]) %*% nu_1[idx]) * log(n1/n2) )
  }

  # Obtain training error
  X_coef_c2 <- apply(X_coef[, idx], 1, function(row){ row - (mu1[idx] + mu2[idx])/2 })
  X_coef_c2 <- t(X_coef_c2)
  pred <- as.integer(ifelse(X_coef_c2 %*% nu_hat[idx] > threshold, 1, 0))
  err_train <- mean(y != pred)   # training error

  # Output object
  res <- list(
    nu_hat = nu_hat,   # sparse discriminant solution
    discrim_set = discrim_set_idx,   # selected discriminant set
    # idx = idx,         # indices of zero coefficients of nu_hat
    threshold = threshold,   # threshold of discrimination rule
    estimates = list(   # prior estimates
      mu = mu,
      mu1 = mu1,
      mu2 = mu2,
      pi1 = pi1,
      pi2 = pi2
    ),
    n_basis = basis_obj$n_basis,
    lambda = lambda,
    groups = groups,
    basis_obj = basis_obj,
    pred_train = pred,
    err_train = err_train
  )
  class(res) <- "hdflda"

  return(res)
}


#' Predict the class labels using the high-dimensional functional LDA
#'
#' Obtain the predicted class labels using `hdflda` object
#'
#' @param object a `hdflda` object
#' @param newdata a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param ... Not used
#'
#' @return a `hdflda` object
#'
#' @references Xue, K., Yang, J., & Yao, F. (2023). Optimal linear discriminant analysis for high-dimensional functional data. Journal of the American Statistical Association, 1-10.
#'
#' @importFrom stats predict
#' @export
predict.hdflda <- function(object, newdata, ...) {
  # Make basis coefficient matrix
  X_coef_test <- predict(object$basis_obj, newdata)

  # Fitted solutions
  nu_hat <- object$nu_hat
  threshold <- object$threshold

  # Non-zero indices of discriminant vector
  idx <- which(nu_hat > 0)

  # Prediction
  if (length(idx) == 1) {
    X_coef_test_c2 <- X_coef_test[, idx] - (object$estimates$mu1[idx] + object$estimates$mu2[idx])/2
    X_coef_test_c2 <- matrix(X_coef_test_c2, ncol = 1)
  } else {
    X_coef_test_c2 <- apply(X_coef_test[, idx], 1, function(row){ row - (object$estimates$mu1[idx] + object$estimates$mu2[idx])/2 })
    X_coef_test_c2 <- t(X_coef_test_c2)
  }
  pred <- as.integer(ifelse(X_coef_test_c2 %*% nu_hat[idx] > threshold, 1, 0))

  return(pred)
}

#' K-fold cross-validation for `hdflda`
#'
#' Select the optimal `n_basis` and `lambda` for `hdflda` using K-fold cross-validation
#' Parallel computing can be used by using the `doParallel` package usages.
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param grid a vector containing m timepoints
#' @param basis "bspline" is only supported
#' @param n_basis_list a vector containing the candidate of `n_basis` (the number of cubic B-spline bases using `n_basis`-2 knots)
#' @param lambda_list a vector containing the candidate of `lambda` (a penalty parameter for L1-regularization)
#' @param measure the loss function for the cross-validation. "accuracy" or "cross.entropy" (Default is "accuracy")
#' @param tol a tolerance rate to define the sparse discriminant set
#' @param K the nuber of folds for K-fold CV
#'
#' @return a `hdflda` object
#'
#' @examples
#' \dontrun{
#' library(doParallel)
#' cl <- makePSOCKcluster(detectCores()/2)
#' registerDoParallel(cl)
#' fit <- cv.hdflda(X_train, y_train)
#' stopCluster(cl)
#' pred <- predict(fit$opt_fit, X_test)
#' mean(pred != y_test)
#' }
#'
#' @importFrom foreach %dopar% foreach
#' @export
cv.hdflda <- function(X,
                      y,
                      grid = NULL,
                      basis = "bspline",
                      n_basis_list = NULL,
                      lambda_list = NULL,
                      measure = "accuracy",
                      tol = 1e-7,
                      K = 10) {

  n <- dim(X)[1]   # number of curves
  m <- dim(X)[2]   # number of timepoints
  p <- dim(X)[3]   # number of variables

  # Candidates of grid search
  if (is.null(n_basis_list)) {
    n_basis_list <- 4:8
    # # n_basis_list <- seq(5, round(m/4), 5)
    # # n_basis_list <- seq(4, min(20, round(m/2)), 1)
    # n_basis_list <- c(4:9, seq(10, min(40, round(m/2)), by = 5))
  }
  if (is.null(lambda_list)) {
    lambda_list <- seq(1e-3, 0.1, length.out = 10)
    # lambda_list <- 5^seq(-3, -1, length.out = 5)
    # lambda_list <- 10^seq(-4, -1.5, length.out = 100)
  }
  cand_cv <- expand.grid(n_basis = n_basis_list,
                         lambda = lambda_list)

  fold_list <- sample(1:K, n, replace = T)
  loss_list <- rep(0, length(lambda_list))

  # K-fold CV
  i <- NULL
  loss_list <- foreach::foreach(i = 1:nrow(cand_cv),
                                .packages=c("glmnet","fda","hdfda"),
                                # .packages=c("CVXR","fda","hdfda"),
                                .combine = c,
                                .errorhandling = "pass") %dopar% {
     loss_i <- rep(NA, K)
     n_basis <- cand_cv[i, 1]
     lambda <- cand_cv[i, 2]
     for (j in 1:K) {
       # Split data
       X_train <- X[fold_list != j, , ]
       X_test <- X[fold_list == j, , ]
       y_train <- y[fold_list != j]
       y_test <- y[fold_list == j]

       tryCatch({
         # Fit hdflda
         fit_hdflda <- hdflda(X_train, y_train,
                              grid = grid,
                              basis = basis,
                              n_basis = n_basis,
                              lambda = lambda,
                              tol = tol)

         # Prediction of validation set
         pred <- predict(fit_hdflda, X_test)

         # Validation error
         if (measure == "accuracy") {
           # Validation misclassification error rate
           # loss_i[j] <- mean(y_test != pred)
           loss_i[j] <- sum(y_test != pred)
         } else if (measure == "cross.entropy") {
           # # Cross-entropy loss
           loss_i[j] <- -sum( log(pred[y_test == 1]) ) - sum( log(1-pred[y_test == 0]) )
         }

       }, error = function(e){
         # If all zero coefficients error is occured, return all predictions are false.
         print(e)
         # It can be resonable to assign n_test as error when we use the cross-entropy or accuracy
         loss_i[j] <<- length(y_test)
         # loss_i[j] <<- Inf
       })

     }

     # loss_i <- mean(loss_i)
     loss_i <- sum(loss_i) / n
     return(loss_i)
   }
  # stopCluster(cl)

  # Optimal hyperparameters
  n_basis <- cand_cv[which.min(loss_list), 1]
  lambda <- cand_cv[which.min(loss_list), 2]
  cand_cv$cv_error <- loss_list

  # Fit hdflda using the optimal parameters
  fit <- hdflda(X, y,
                grid = grid,
                basis = basis,
                n_basis = n_basis,
                lambda = lambda,
                tol = tol)

  res <- list(
    opt_fit = fit,
    opt_params = c(n_basis = n_basis,
                   lambda = lambda),
    # opt_n_basis = n_basis,
    # opt_lambda = lambda,
    cv_error = cand_cv
  )

  return(res)
}

