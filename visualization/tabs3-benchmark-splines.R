library(jsonlite)
library(tibble)

dims = c(8, 16, 32, 64)
splines = c("linear", "quadratic")
params = expand.grid(dim = dims, spline = splines)

read_result = function(dim, spline)
{
    filename = sprintf("../tf/model/clayton-dim%d-%s-alpha0.7-benchmark.json",
                       dim, spline)
    dat = read_json(filename, simplifyVector = TRUE)
    res = tibble(method = names(dat), time = sapply(dat, median),
                 dim = dim, spline = spline)
    res
}

res = mapply(read_result, dim = params$dim, spline = params$spline,
             SIMPLIFY = FALSE)
res = do.call(rbind, res)
print(res)

training_time = function(dim, spline)
{
    filename = sprintf("../tf/model/clayton-dim%d-%s-alpha0.7-losses.json",
                       dim, spline)
    dat = read_json(filename, simplifyVector = TRUE)
    cat(sprintf("dim = %d, spline = %s, nbeta = %d, training time = %.1f\n",
                dim, spline, length(dat$logbeta) + 1, dat$total_time))
}

invisible(
    mapply(training_time, dim = params$dim, spline = params$spline,
           SIMPLIFY = FALSE)
)
