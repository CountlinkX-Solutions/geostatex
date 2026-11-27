# Generate block kriging fixtures (`fixtures/kriging/block.json`) with R gstat.
#
# Run from the repository root (under a memory cap):
#
# (R_ENV as in test/reference/README.md)
#
#     prlimit --as=3221225472 $R_ENV/env/bin/Rscript test/reference/generate_kriging_block.R
#
# Ordinary block kriging of Meuse log-zinc (`krige(..., block = )`): rectangular blocks (gstat's
# default discretisation, 4 x 4 Gauss-Legendre points), an irregular block given by offsets
# (equal weights), the three models with a sill, and point kriging for reference. Targets are
# block centres: 10 random ones and two data points.

source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "common.R"))

meuse <- meuse_data()
d <- data.frame(x = meuse$x, y = meuse$y, z = log(meuse$zinc))
coordinates(d) <- ~ x + y
set.seed(15)
tx <- c(runif(10, min(meuse$x), max(meuse$x)), meuse$x[1:2])
ty <- c(runif(10, min(meuse$y), max(meuse$y)), meuse$y[1:2])
targets <- data.frame(x = tx, y = ty)
coordinates(targets) <- ~ x + y

cases <- list()
run <- function(name, model, params, block = NULL, offsets = NULL) {
  m <- vgm_of(model, params[1], params[2], params[3])
  b <- if (!is.null(offsets)) offsets else if (!is.null(block)) block else numeric(0)
  p <- as.data.frame(krige(z ~ 1, d, targets, m, block = b, debug.level = 0))
  spec <- if (!is.null(offsets)) list(points = unname(as.matrix(offsets))) else if (!is.null(block)) list(size = block) else NULL
  cases[[length(cases) + 1]] <<- list(
    name = name, variogram_model = model, parameters = params, block = spec,
    z = p$var1.pred, variance = p$var1.var
  )
}

sph <- c(0.59, 897, 0.05)
run("spherical_40", "spherical", sph, block = c(40, 40))
run("exponential_40", "exponential", c(0.6, 1100, 0.04), block = c(40, 40))
run("gaussian_40", "gaussian", c(0.55, 800, 0.06), block = c(40, 40))
run("rectangle", "spherical", sph, block = c(100, 30))
offsets <- expand.grid(dx = c(-20, 0, 20), dy = c(-20, 0, 20))
run("offsets", "spherical", sph, offsets = offsets)
run("point", "spherical", sph)

dump_json("kriging/block.json", list(
  name = "kriging_block",
  description = "gstat krige(block =) ordinary block kriging of Meuse log-zinc",
  meta = meta(),
  data = list(x = meuse$x, y = meuse$y, values = log(meuse$zinc)),
  targets = list(x = tx, y = ty),
  cases = cases
))
cat(length(cases), "cases\n")
