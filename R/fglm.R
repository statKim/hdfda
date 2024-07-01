#' Functional generalized linear model (FGLM)
#'
#' Scalar on function generalized linear model (It is recommended for the number of functional variables p < 5.)
#'
#' @param X a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param y a integer vector containing class label of X (n x 1 vector)
#' @param family the parameter of family for `glm()`. Default is "binomial".
#' @param grid a vector containing m timepoints
#' @param basis "fpca" (FPCA) or "bspline" (B-spline). Default is "bspline".
#' @param FVE the fraction of variance explained to choose the number of the FPCs
#' @param K the number of FPCs
#' @param n_basis the number of cubic B-spline bases using `n_basis`-2 knots
#'
#' @return a `fglm` object
#'
#' @importFrom stats glm
#' @export
fglm <- function(X, y,
                 family = "binomial",
                 grid = NULL,
                 basis = "bspline",
                 FVE = 0.90,
                 K = NULL,
                 n_basis = 20) {
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

  # GLM
  df <- data.frame(
    y = y,
    X_coef
  )
  # colnames(df) <- c("y", X_names)
  fit <- glm(y ~ ., data = df, family = family)


  if (basis == "bspline") {
    res <- list(
      basis = basis,
      # basis_ftn = basis_ftn,
      grid = grid,
      n_knots = n_knots,
      n_basis = basis_obj$n_basis,
      # X_names = X_names,
      basis_obj = basis_obj,
      model.obj = fit
    )
  } else if (basis == "fpca") {
    res <- list(
      basis = basis,
      # uFPCA.obj = uFPCA.obj.list,
      grid = grid,
      num_pc = basis_obj$num_pc,
      # X_names = X_names,
      basis_obj = basis_obj,
      model.obj = fit
    )
  }

  class(res) <- "fglm"

  return(res)
}


#' Predict the class labels using the FGLM
#'
#' Obtain the predicted class labels using `fglm` object
#'
#' @param object a `fglm` object
#' @param newdata a n-m-p array (p-variate functional data; each functional data consists of n curves observed from m timepoints)
#' @param threshold the cutoff value for the binary classification. Default is 0.5.
#' @param ... Not used
#'
#' @return a predicted class
#'
#' @importFrom stats predict
#' @export
predict.fglm <- function(object, newdata, threshold = 0.5, ...) {
  # Make basis coefficient matrix
  X_coef <- predict.make_basis_mf(object$basis_obj, newdata)
  df <- data.frame(X_coef)
  # colnames(df) <- object$X_names

  fit <- object$model.obj

  # Prediction using "sparsegl" package
  pred <- predict(fit, newdata = df, type = "response")
  pred <- ifelse(pred > threshold, 1, 0)
  pred <- as.integer(pred)

  return(pred)
}
