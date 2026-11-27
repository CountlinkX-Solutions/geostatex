# JSON conventions of the R fixture generators (as common.py): 17 significant digits, so every
# double round-trips, and a `meta` block with the versions that generated the fixture.

suppressMessages({
  library(sp)
  library(gstat)
  library(jsonlite)
})

fixtures_dir <- file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))), "fixtures")

meta <- function() {
  list(
    R = paste(R.version$major, R.version$minor, sep = "."),
    gstat = as.character(packageVersion("gstat")),
    sp = as.character(packageVersion("sp"))
  )
}

dump_json <- function(path, obj) {
  path <- file.path(fixtures_dir, path)
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write_json(obj, path, digits = I(17), auto_unbox = TRUE, pretty = TRUE)
}

# Geostatex's (PyKrige) models in gstat: Exp and Gau take the scale parameter.
vgm_of <- function(model, psill, range, nugget = 0) {
  switch(model,
    spherical = vgm(psill, "Sph", range, nugget),
    exponential = vgm(psill, "Exp", range / 3, nugget),
    gaussian = vgm(psill, "Gau", range * 4 / 7, nugget)
  )
}

meuse_data <- function() {
  data(meuse, package = "sp", envir = environment())
  meuse
}
