library(jsonlite)
library(tibble)

dims = c(8, 16, 32, 64)

read_result = function(dim)
{
    filename = sprintf("../tf/model/clayton-dim%d-benchmark.json", dim)
    dat = read_json(filename, simplifyVector = TRUE)
    res = tibble(method = names(dat), time = sapply(dat, median), dim = dim)
    res
}

res = lapply(dims, read_result)
res = do.call(rbind, res)
print(res)

training_time = function(dim)
{
    filename = sprintf("../tf/model/clayton-dim%d-losses.json", dim)
    dat = read_json(filename, simplifyVector = TRUE)
    cat(sprintf("dim = %d, nbeta = %d, training time = %.2f\n", dim, length(dat$logbeta) + 1, dat$total_time))
}

invisible(lapply(dims, training_time))
