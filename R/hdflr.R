#' Functional linear regression for high-dimensional functional data with group SCAD penalty
#'
#' Scalar on function linear regression with group SCAD penalty in high-dimension
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param grid a vector containing m timepoints
#' @param penalty the penalty type ("gr_scad" is only supported)
#' @param type "regress" for regression and "classif" for LDA direction estimation
#' @param basis "bspline" is only supported
#' @param n_basis the number of cubic B-spline bases using `n_basis`-2 knots
#' @param lambda a penalty parameter for L1-regularization
#' @param tol a tolerance rate to define the sparse discriminant set
#'
#' @return a `hdflr` object
#'
#' @importFrom stats cov
#' @export
hdflr <- function(X, 
                  y,
                  grid = NULL,
                  penalty = "gr_scad",
                  type = "regress",
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

  # Check whether response is continuous or binary 
  if (type == "regress") {
    # Centering response
    mean_y <- mean(y)
    z <- y - mean_y
  } else if (type == "classif") {
    # Index set
    idx_g1 <- which(y == 0)
    idx_g2 <- which(y == 1)
    n1 <- length(idx_g1)
    n2 <- length(idx_g2)
    n <- nrow(X_coef)
    
    # Prior probability
    pi1 <- n1 / n
    pi2 <- 1 - pi1
    
    z <- ifelse(y == 1, pi1, -pi2)
  }
 
  # Centering the basis coefficient matrix
  X_coef_c <- scale(X_coef, center = T, scale = F)
  
  # Optimize the coeffient eta_hat using coordinate descent algorithm
  fit_obj <- group_scad_flr_cpp(z,
                                X_coef_c,
                                p,
                                n_basis,
                                lambda,
                                a = 3.7,
                                max_iter = 100,
                                tol = tol) 
  eta_hat <- as.numeric(fit_obj$eta_hat)
  iter <- fit_obj$iterations
  
  # Active set from the spasrse solution
  active_set <- which(apply(fit_obj$eta_hat, 2, function(x){ sum(abs(x)) }) > 0)
  

  if (type == "regress") {
    pred <- mean_y + as.numeric(X_coef_c %*% matrix(eta_hat, ncol = 1))
    err_train <- mean((z - pred)^2)   # training MSE
    
    threshold <- NULL
    estimates <- list(
      mean_y = mean_y,
      mean_x = colMeans(X_coef)
    )
  } else if (type == "classif") {
    # Mean of each basis coefficient
    mu <- colMeans(X_coef)
    mu1 <- colMeans(X_coef[idx_g1, ])
    mu2 <- colMeans(X_coef[idx_g2, ])

    # Pooled sample covariance
    S1 <- cov(X_coef[idx_g1, ])
    S2 <- cov(X_coef[idx_g2, ])
    S <- ((n1-1)*S1 + (n2-1)*S2) / (n-2)

    # Obtain the discrimination vector and discrimination threshold
    if (length(active_set) == 0) {
      # stop("All zero coefficients are obtained!")
      threshold <- 0
      idx <- 1:length(eta_hat)
    } else {
      idx <- which(groups %in% active_set)
      threshold <- as.numeric( (t(eta_hat[idx]) %*% S[idx, idx] %*% eta_hat[idx]) / (t(eta_hat[idx]) %*% (mu2 - mu1)[idx]) * log(n1/n2) )
    }

    # Obtain training error
    X_coef_c2 <- apply(X_coef[, idx], 1, function(row){ row - (mu1[idx] + mu2[idx])/2 })
    X_coef_c2 <- t(X_coef_c2)
    pred <- as.integer(ifelse(X_coef_c2 %*% eta_hat[idx] > threshold, 1, 0))
    err_train <- mean(y != pred)   # training error
    
    # prior estimates
    estimates <- list(   
      mu = mu,
      mu1 = mu1,
      mu2 = mu2,
      pi1 = pi1,
      pi2 = pi2
    )
  }
  
  # Output object
  res <- list(
    eta_hat = eta_hat,   # sparse solution
    active_set = active_set,   # selected active set
    n_basis = basis_obj$n_basis,
    basis_obj = basis_obj,
    groups = groups,
    type = type,
    penalty = penalty,
    lambda = lambda,
    threshold = threshold,   # threshold of discrimination rule
    estimates = estimates,
    pred_train = pred,
    err_train = err_train
  )
  class(res) <- "hdflr"

  return(res)
}



#' Predict the new data using the high-dimensional functional linear regression
#'
#' Obtain the prediction using `hdflr` object
#'
#' @param object a `hdflr` object
#' @param newdata a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param ... Not used
#'
#' @return a `hdflr` object
#'
#' @importFrom stats predict
#' @export
predict.hdflr <- function(object, newdata, ...) {
  # Make basis coefficient matrix
  X_coef_test <- predict(object$basis_obj, newdata)

  # Estimated sparse solutions
  eta_hat <- object$eta_hat
  # Non-zero indices of active coefficient vector
  idx <- which(eta_hat > 0)
  
  # Prediction
  if (object$type == "regress") {
    X_coef_test_c2 <- apply(X_coef_test[, idx, drop=FALSE], 1, function(row){ row - object$estimates$mean_x[idx] })
    X_coef_test_c2 <- t(X_coef_test_c2)
    pred <- as.numeric(X_coef_test_c2 %*% eta_hat[idx]) + object$estimates$mean_y
    
  } else if (object$type == "classif") {
    X_coef_test_c2 <- apply(X_coef_test[, idx, drop=FALSE], 1, function(row){ row - (object$estimates$mu1[idx] + object$estimates$mu2[idx])/2 })
    X_coef_test_c2 <- t(X_coef_test_c2)
    pred <- as.integer(ifelse(X_coef_test_c2 %*% eta_hat[idx] > object$threshold, 1, 0))
  }
  
  return(pred)
}


#' K-fold cross-validation for `hdflr`
#'
#' Select the optimal `n_basis` and `lambda` for `hdflr` using K-fold cross-validation
#' Parallel computing can be used by using the `doParallel` package usages.
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param grid a vector containing m timepoints
#' @param penalty the penalty type ("gr_scad" is only supported)
#' @param type "regress" for regression and "classif" for LDA direction estimation
#' @param basis "bspline" is only supported
#' @param n_basis_list a vector containing the candidate of `n_basis` (the number of cubic B-spline bases using `n_basis`-2 knots)
#' @param lambda_list a vector containing the candidate of `lambda` (a penalty parameter for L1-regularization)
#' @param measure the loss function for the cross-validation. "accuracy" or "cross.entropy" (Default is "accuracy")
#' @param tol a tolerance rate to define the sparse discriminant set
#' @param K the nuber of folds for K-fold CV
#'
#' @return a `hdflr` object
#'
#' @importFrom foreach %dopar% foreach
#' @export
cv.hdflr <- function(X,
                     y,
                     grid = NULL,
                     penalty = "gr_scad",
                     type = "regress",
                     basis = "bspline",
                     n_basis_list = NULL,
                     lambda_list = NULL,
                     measure = "accuracy",
                     tol = 1e-7,
                     K = 5) {
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
  cand_cv <- expand.grid(n_basis = n_basis_list,
                         lambda = lambda_list)

  fold_list <- sample(1:K, n, replace = T)
  # loss_list <- rep(0, length(lambda_list))

  # K-fold CV
  i <- NULL
  loss_list <- foreach::foreach(i = 1:nrow(cand_cv),
                                .packages=c("fda","hdfda"),
                                .combine = c,
                                .errorhandling = "pass") %dopar% {
     loss_i <- rep(NA, K)
     n_basis <- cand_cv[i, 1]
     lambda <- cand_cv[i, 2]
     for (j in 1:K) {
       # Split data
       X_train <- X[fold_list != j, , , drop=FALSE]
       X_test <- X[fold_list == j, , , drop=FALSE]
       y_train <- y[fold_list != j]
       y_test <- y[fold_list == j]

       tryCatch({
         # Fit hdflr
         fit_obj <- hdflr(X_train, 
                          y_train,
                          grid = grid,
                          penalty = penalty,
                          type = type,
                          basis = basis,
                          n_basis = n_basis,
                          lambda = lambda,
                          tol = tol)

         # Prediction of validation set
         pred <- predict(fit_obj, X_test)

         # Validation error
         if (type == "regress") {
           # mean squared error
           loss_i[j] <- sum((y_test - pred)^2)
         } else if (type == "classif") {
           if (measure == "accuracy") {
             # Validation misclassification error rate
             # loss_i[j] <- mean(y_test != pred)
             loss_i[j] <- sum(y_test != pred)
           } else if (measure == "cross.entropy") {
             # # Cross-entropy loss
             loss_i[j] <- -sum( log(pred[y_test == 1]) ) - sum( log(1-pred[y_test == 0]) )
           }
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

  # Fit hdflr using the optimal parameters
  fit <- hdflr(X, 
               y,
               grid = grid,
               penalty = penalty,
               type = type,
               basis = basis,
               n_basis = n_basis,
               lambda = lambda,
               tol = tol)
  
  res <- list(
    opt_fit = fit,
    opt_params = c(n_basis = n_basis,
                   lambda = lambda),
    cv_error = cand_cv
  )

  return(res)
}

