# Generate ordinary cokriging and LMC fit fixtures (`fixtures/kriging/cokriging.json`) with R gstat.
#
# Run from the repository root (under a memory cap):
#
# (R_ENV as in test/reference/README.md)
#
#     prlimit --as=3221225472 $R_ENV/env/bin/Rscript test/reference/generate_cokriging.R
#
# Meuse log-zinc, log-copper and log-lead. Linear models of coregionalisation (a nugget and one
# structure sharing its range) with fixed sills: two variables isotopic and heterotopic (copper
# at every other point), three variables with an exponential structure. Each variable has its own
# constant mean (`~ 1`), so this is ordinary cokriging. Predictions at 10 random points and two
# data points: each variable's estimate and variance and the cross covariances.
#
# `fit.lmc` on gstat's multivariable sample variogram of the isotopic pair (and of the three
# variables), ranges fixed: the sample variograms and the fitted sills are stored.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "common.R"))

meuse <- meuse_data()
vars <- list(zn = log(meuse$zinc), cu = log(meuse$copper), pb = log(meuse$lead))
set.seed(16)
tx <- c(runif(10, min(meuse$x), max(meuse$x)), meuse$x[1:2])
ty <- c(runif(10, min(meuse$y), max(meuse$y)), meuse$y[1:2])
targets <- data.frame(x = tx, y = ty)
coordinates(targets) <- ~ x + y

frame <- function(rows, name) {
  f <- data.frame(x = meuse$x[rows], y = meuse$y[rows], v = vars[[name]][rows])
  coordinates(f) <- ~ x + y
  f
}

# An LMC: structures list(list(model, range, sills = matrix)), the first a nugget.
build <- function(names, rows, structures) {
  g <- NULL
  for (i in seq_along(names)) {
    g <- gstat(g, names[i], v ~ 1, frame(rows[[i]], names[i]), model = lmc_model(structures, i, i))
  }
  for (i in seq_along(names)) for (j in seq_along(names)) if (i < j) {
    g <- gstat(g, c(names[i], names[j]), model = lmc_model(structures, i, j))
  }
  g
}

lmc_model <- function(structures, i, j) {
  nug <- structures[[1]]$sills[i, j]
  s <- structures[[2]]
  vgm_of(s$model, s$sills[i, j], s$range, nug)
}

cases <- list()
run <- function(name, names, rows, structures) {
  g <- build(names, rows, structures)
  p <- as.data.frame(predict(g, targets, debug.level = 0))
  estimates <- lapply(names, function(n) p[[paste0(n, ".pred")]])
  variances <- lapply(names, function(n) p[[paste0(n, ".var")]])
  covariances <- list()
  for (i in seq_along(names)) for (j in seq_along(names)) if (i < j) {
    covariances[[length(covariances) + 1]] <- list(i = i - 1, j = j - 1, values = p[[paste0("cov.", names[i], ".", names[j])]])
  }
  cases[[length(cases) + 1]] <<- list(
    name = name,
    variables = lapply(seq_along(names), function(i) list(name = names[i], rows = rows[[i]] - 1)),
    structures = lapply(structures, function(s) list(model = s$model, range = if (is.null(s$range)) 0 else s$range, sills = unname(s$sills))),
    estimates = estimates, variances = variances, covariances = covariances
  )
}

all <- seq_len(nrow(meuse))
two <- list(
  list(model = "nugget", sills = matrix(c(0.05, 0.03, 0.03, 0.04), 2)),
  list(model = "spherical", range = 900, sills = matrix(c(0.55, 0.40, 0.40, 0.35), 2))
)
run("isotopic", c("zn", "cu"), list(all, all), two)
run("heterotopic", c("zn", "cu"), list(all, seq(1, nrow(meuse), 2)), two)
three <- list(
  list(model = "nugget", sills = matrix(c(0.05, 0.02, 0.03, 0.02, 0.04, 0.02, 0.03, 0.02, 0.06), 3)),
  list(model = "exponential", range = 1200, sills = matrix(c(0.55, 0.38, 0.45, 0.38, 0.35, 0.30, 0.45, 0.30, 0.45), 3))
)
run("three_exponential", c("zn", "cu", "pb"), list(all, all, all), three)

# fit.lmc with fixed ranges on gstat's sample variograms.
fits <- list()
fit <- function(name, names, model) {
  g <- NULL
  for (n in names) g <- gstat(g, n, v ~ 1, frame(all, n))
  v <- variogram(g)
  f <- fit.lmc(v, g, model)
  ids <- unique(as.character(v$id))
  fits[[length(fits) + 1]] <<- list(
    name = name, names = names,
    model = list(model = as.character(model$model[2]), range = model$range[2], initial = c(model$psill[1], model$psill[2])),
    sample = lapply(ids, function(id) {
      s <- v[v$id == id, ]
      list(id = id, np = s$np, dist = s$dist, gamma = s$gamma)
    }),
    fitted = lapply(names(f$model), function(id) list(id = id, nugget = f$model[[id]]$psill[1], psill = f$model[[id]]$psill[2]))
  )
}
fit("pair_spherical", c("zn", "cu"), vgm(0.5, "Sph", 900, 0.05))
fit("three_spherical", c("zn", "cu", "pb"), vgm(0.5, "Sph", 900, 0.05))

dump_json("kriging/cokriging.json", list(
  name = "kriging_cokriging",
  description = "gstat ordinary cokriging with a linear model of coregionalisation, and fit.lmc",
  meta = meta(),
  data = list(x = meuse$x, y = meuse$y, zn = vars$zn, cu = vars$cu, pb = vars$pb),
  targets = list(x = tx, y = ty),
  cases = cases,
  fits = fits
))
cat(length(cases), "cases,", length(fits), "fits\n")
