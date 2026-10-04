# Reference values for tests/test_r_parity.py, from the HotellingEllipse R package.
# Usage: Rscript tests/r_parity_reference.R
# then paste the output into tests/test_r_parity.py. X must match DATA there.
suppressPackageStartupMessages(library(HotellingEllipse))
stopifnot(packageVersion("HotellingEllipse") == "1.3.0")

X <- matrix(c(
  -2.38, -0.59, -2.49, 0.28,
  1.91, 0.33, 0.24, 0.13,
  -0.80, -0.66, 0.77, 0.16,
  -0.19, -0.21, 0.20, -0.27,
  -1.21, 0.34, -0.77, -0.48,
  -1.43, 0.41, -1.00, 0.02,
  1.93, 3.51, -1.85, 1.14,
  -3.69, -1.21, -1.92, 0.23,
  2.50, 2.71, -1.38, 0.69,
  -1.56, -1.31, 0.61, 0.16,
  0.61, -0.70, -0.22, 0.35,
  1.78, 1.79, 1.90, 0.23
), ncol = 4, byrow = TRUE)

py <- function(v) paste0("[", paste(sprintf("%.17g", unname(v)), collapse = ", "), "]")
emit <- function(name, res) {
  cat(sprintf("    '%s': {\n", name))
  cat(sprintf("        'Tsquared': %s,\n", py(res$Tsquare$value)))
  for (nm in grep("^cutoff", names(res), value = TRUE))
    cat(sprintf("        '%s': %.17g,\n", sub("^cutoff\\.", "cutoff_", nm), res[[nm]]))
  cat(sprintf("        'nb_comp': %d,\n", res$nb.comp))
  if (!is.null(res$Ellipse)) {
    e <- res$Ellipse
    cat(sprintf("        'Ellipse': {%s},\n", paste(sprintf("'%s': %.17g", sub("^([ab])\\.", "\\1_", names(e)), unlist(e)), collapse = ", ")))
  }
  cat("    },\n")
}

cat("PARAMETERS = {\n")
emit("k2_pcx1_pcy3", ellipseParam(X, pcx = 1, pcy = 3))
emit("k2_pcx2_pcy1_beta_levels", ellipseParam(X, pcx = 2, pcy = 1, method = "beta", conf.limit = c(0.9, 0.975)))
emit("k3", ellipseParam(X, k = 3))
emit("threshold_0.9_beta", ellipseParam(X, threshold = 0.9, method = "beta"))
cat("}\n\n")
xy <- ellipseCoord(X, pcx = 2, pcy = 3, conf.limit = 0.9, pts = 6)
cat(sprintf("COORD_2D = {'x': %s,\n            'y': %s}\n\n", py(xy$x), py(xy$y)))
xyz <- ellipseCoord(X, pcx = 1, pcy = 2, pcz = 4, pts = 4, method = "beta")
cat(sprintf("COORD_3D = {'x': %s,\n            'y': %s,\n            'z': %s}\n", py(xyz$x), py(xyz$y), py(xyz$z)))
