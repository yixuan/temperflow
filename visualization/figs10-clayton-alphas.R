library(reticulate)
library(transport)
library(kernlab)
library(tibble)
library(reshape2)
library(dplyr)
library(ggplot2)
library(showtext)
font_add_google("Lato")

py_config()
np = import("numpy")

source("clayton.R")

# Utility function to read .npz files
read_npz = function(npz_file)
{
    npz = np$load(npz_file)
    files = npz$files
    nfiles = length(files)

    res = vector("list", nfiles)
    for(i in seq_along(files))
    {
        res[[i]] = npz$f[[files[i]]]
    }
    names(res) = files

    res
}

# Compute distances between samples
compute_stats = function(xtrue, xsamp)
{
    nsim = dim(xtrue)[1]
    w = numeric(nsim)
    mmd = numeric(nsim)
    for(j in 1:nsim)
    {
        print(j)
        w[j] = wasserstein(pp(xsamp[j, , ]), pp(xtrue[j, , ]))
        mmd[j] = kmmd(xsamp[j, , ], xtrue[j, , ])@mmdstats[1]
    }

    err = tibble(w = w, mmd = mmd)
    err
}

set.seed(123)
alphas = c(0.5, 0.6, 0.7, 0.8, 0.9)
dims = c(8, 16, 32, 64)
errs = list()
for(d in dims)
{
    nsim = 100
    nsamp = 1000
    
    # True samples
    distr = ClaytonNormalMix$new(dim = d, mix_dim = 8, theta = 2)
    xtrue = distr$sample(nsim * nsamp)
    dim(xtrue) = c(nsim, nsamp, d)
    xtrue2 = distr$sample(nsim * nsamp)
    dim(xtrue2) = c(nsim, nsamp, d)
    
    # Compute baseline statistics (difference between xtrue and xtrue2)
    base_err = compute_stats(xtrue, xtrue2)
        
    for(alpha in alphas)
    {
        # Sample from the model
        filename = sprintf("../tf/model/clayton-dim%d-linear-alpha%.1f.npz", d, alpha)
        dat = read_npz(filename)
        xsamp = dat$xest[1:nsim, 1:nsamp, ]

        # Compute errors
        err = compute_stats(xtrue, xsamp)
        # Subtract baseline errors
        err$w = err$w - base_err$w
        err$mmd = err$mmd - base_err$mmd
        # Add meta information
        err = err %>% mutate(dim = d, alpha = alpha)
        errs = c(errs, list(err))
    }
}

gdat = do.call(rbind, errs)
gdat$dim = factor(sprintf("Dim = %d", gdat$dim), levels = sprintf("Dim = %d", dims))
colnames(gdat) = c("Adjusted 1-Wasserstein", "Adjusted MMD", "dim", "alpha")
gdat = melt(gdat, id.vars = c("dim", "alpha"),
            variable.name = "metric", value.name = "err")
gdat$alpha = factor(gdat$alpha, levels = as.character(alphas))
ggplot(gdat, aes(x = alpha, y = err)) +
    facet_grid(rows = vars(metric), cols = vars(dim), scales = "free_y") +
    geom_boxplot(aes(color = alpha), width = 0.6) +
    scale_x_discrete("Discounting Factor α") +
    ylab("Sampling Error") +
    scale_color_hue("Discounting Factor", labels = sprintf("α = %s", alphas)) +
    theme_bw(base_size = 16, base_family = "sans") +
    theme(panel.grid = element_blank(),
          axis.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/experiments_copula_alphas.pdf", width = 15, height = 6)
