#' Lasso for functional generalized linear model based on group lasso
#'
#' Scalar on function generalized linear model
#' It implements the group lasso for FPC scores or B-spline bases coefficients.
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param family the parameter of family for `glm()`. Default is "binomial".
#' @param grid a vector containing m timepoints
#' @param basis "fpca" (FPCA) or "bspline" (B-spline). Default is "bspline".
#' @param lambda a penalty parameter for group lasso. Default is 0.1. If it is `NULL` or the vector containing the candidates of `lambda`, cross-validation is performed.
#' @param alpha a relative weight for L1-regularization (1-alpha for L2-regularization). Default is 0.05.
#' @param FVE the fraction of variance explained to choose the number of the FPCs
#' @param K the number of FPCs
#' @param n_basis the number of cubic B-spline bases using `n_basis`-2 knots. If it is `NULL` or the vector containing the candidates of `n_basis`, cross-validation is performed.
#'
#' @return a `flasso` object
#'
#' @importFrom stats glm
#' @export
flasso <- function(X, y,
                   family = "binomial",
                   grid = NULL,
                   basis = "bspline",
                   lambda = 0.1,
                   alpha = 0.05,
                   FVE = 0.90,
                   K = NULL,
                   n_basis = 20) {
  if (is.null(n_basis) | length(n_basis) > 1) {
    # Perform cross-validation for n_basis
    # Fit all candidate of n_basis_list and find the best model
    n_basis_list <- n_basis
    basis_obj_list <- list()
    model_list <- list()
    min_cv_error <- rep(NA, length(n_basis_list))
    for (i in 1:length(n_basis_list)) {
      # Basis representation for each functional covariate
      n_knots <- n_basis_list[i] - 2   # cubic B-spline
      basis_obj <- make_basis_mf(X, grid = grid,
                                 basis = basis,
                                 FVE = FVE,
                                 K = K,
                                 n_knots = n_knots)
      X_coef <- basis_obj$X_coef

      # Observed grid points
      grid <- basis_obj$grid

      # Group indicator for each functional covariate
      groups <- basis_obj$groups

      # Sparse group lasso type functional regression
      if (is.null(lambda)) {
        lambda_list <- 10^seq(-4, -1.5, length.out = 100)
      } else {
        lambda_list <- lambda
      }
      cv_fit <- sparsegl::cv.sparsegl(X_coef, y,
                                      groups,
                                      family = family,
                                      asparse = alpha,
                                      lambda = lambda_list,
                                      standardize = FALSE)
      model_list[[i]] <- cv_fit
      min_cv_error[i] <- min(cv_fit$cvm)
      basis_obj_list[[i]] <- basis_obj
    }

    # Optimal n_basis
    opt_idx_n_basis <- which.min(min_cv_error)

    basis_obj <- basis_obj_list[[opt_idx_n_basis]]
    cv_fit <- model_list[[opt_idx_n_basis]]
    n_basis <- n_basis_list[opt_idx_n_basis]
    n_knots <- n_basis - 2
    lambda <- cv_fit$lambda.min
  } else {
    # Fit fglm given n_basis (Does not perform cross-validation for n_basis)

    # Basis representation for each functional covariate
    n_knots <- n_basis - 2   # cubic B-spline
    basis_obj <- make_basis_mf(X, grid = grid,
                               basis = basis,
                               FVE = FVE,
                               K = K,
                               n_knots = n_knots)
    X_coef <- basis_obj$X_coef

    # Observed grid points
    grid <- basis_obj$grid

    # Group indicator for each functional covariate
    groups <- basis_obj$groups

    # Sparse group lasso type functional regression
    if (is.null(lambda)) {
      lambda_list <- 10^seq(-4, -1.5, length.out = 100)
    } else {
      lambda_list <- lambda
    }
    cv_fit <- sparsegl::cv.sparsegl(X_coef, y,
                                    groups,
                                    family = family,
                                    asparse = alpha,
                                    lambda = lambda_list,
                                    standardize = FALSE)
    lambda <- cv_fit$lambda.min
  }


  # if (isTRUE(cv)) {
  #   if (is.null(lambda)) {
  #     lambda_list <- 10^seq(-4, -1.5, length.out = 100)
  #   } else {
  #     lambda_list <- lambda
  #   }
  #   cv_fit <- sparsegl::cv.sparsegl(X_coef, y,
  #                                   groups,
  #                                   family = family,
  #                                   asparse = alpha,
  #                                   lambda = lambda_list,
  #                                   standardize = FALSE)
  #
  #   # # cv_fit <- cv.sparsegl(X_coef, y, groups, family = "binomial", asparse = alpha, standardize = FALSE)
  #   # cv_fit$lambda.min
  #   # lambda <- cv_fit$lambda[which.min(cv_fit$cvm)]
  #   # beta <- coef(cv_fit, s = "lambda.min") %>% as.numeric()
  #   # sum(abs(beta) > 0)
  #   # plot(cv_fit)
  #   # pred <- predict(cv_fit, newx = X_coef, s = "lambda.min", type = "response")
  #   # pred <- ifelse(pred > 0.5, 1, 0)
  #   # pred <- as.integer(pred)
  #   # mean(pred == y)
  #   # cv_fit
  #   # plot(cv_fit$sparsegl.fit)
  #
  #   lambda <- cv_fit$lambda.min
  # } else {
  #
  # }

  if (basis == "bspline") {
    res <- list(
      basis = basis,
      # basis_ftn = basis_ftn,
      grid = grid,
      n_knots = n_knots,
      n_basis = basis_obj$n_basis,
      lambda = lambda,
      groups = groups,
      basis_obj = basis_obj,
      model.obj = cv_fit
    )
  } else if (basis == "fpca") {
    res <- list(
      basis = basis,
      # uFPCA.obj = uFPCA.obj.list,
      grid = grid,
      num_pc = basis_obj$num_pc,
      lambda = lambda,
      groups = groups,
      basis_obj = basis_obj,
      model.obj = cv_fit
    )
  }

  class(res) <- "flasso"

  return(res)
}


#' Predict the class labels using `flasso`
#'
#' Obtain the predicted class labels using `flasso` object
#'
#' @param object a `flasso` object
#' @param newdata a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param threshold the cutoff value for the binary classification. Default is 0.5.
#' @param ... Not used
#'
#' @return a predicted class
#'
#' @importFrom stats predict
#' @export
predict.flasso <- function(object, newdata, threshold = 0.5, ...) {
  # Make basis coefficient matrix
  X_coef <- predict.make_basis_mf(object$basis_obj, newdata)

  cv_fit <- object$model.obj

  # Prediction using "sparsegl" package
  pred <- predict(cv_fit, newx = X_coef, s = "lambda.min", type = "response")
  pred <- ifelse(pred > threshold, 1, 0)
  pred <- as.integer(pred)

  return(pred)
}
