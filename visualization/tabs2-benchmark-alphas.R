library(jsonlite)
library(tibble)

dims = c(8, 16, 32, 64)
alphas = c(0.5, 0.6, 0.7, 0.8, 0.9)
params = expand.grid(dim = dims, alpha = alphas)

training_time = function(dim, alpha)
{
    filename = sprintf("../tf/model/clayton-dim%d-linear-alpha%.1f-losses.json",
                       dim, alpha)
    dat = read_json(filename, simplifyVector = TRUE)
    cat(sprintf("dim = %d, alpha = %s, nbeta = %d, training time = %.1f\n",
                dim, alpha, length(dat$logbeta) + 1, dat$total_time))
}

invisible(
    mapply(training_time, dim = params$dim, alpha = params$alpha,
           SIMPLIFY = FALSE)
)
