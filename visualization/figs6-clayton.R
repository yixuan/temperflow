library(reticulate)
library(transport)
library(kernlab)
library(tibble)
library(reshape2)
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

# Compute sampling errors
summarize_error = function(dim)
{
    nsamp = 1000
    # Read in TemperFlow and MCMC samples
    model_name = sprintf("clayton-dim%d-mcmc10", dim)
    dat = read_npz(paste0("../tf/model/", model_name, ".npz"))
    mod = lapply(dat, function(x) x[, 1:nsamp, ])
    nsim = dim(mod[[1]])[1]
    
    # True samples
    distr = ClaytonNormalMix$new(dim = dim, mix_dim = 8, theta = 2)
    xtrue = distr$sample(nsim * nsamp)
    dim(xtrue) = c(nsim, nsamp, dim)
    xtrue2 = distr$sample(nsim * nsamp)
    dim(xtrue2) = c(nsim, nsamp, dim)

    # Compute distances between samples
    compute_stats = function(xtrue, xsamp, name)
    {
        print(name)
        nsim = dim(xtrue)[1]
        w = numeric(nsim)
        mmd = numeric(nsim)
        for(j in 1:nsim)
        {
            print(j)
            w[j] = wasserstein(pp(xsamp[j, , ]), pp(xtrue[j, , ]))
            mmd[j] = kmmd(xsamp[j, , ], xtrue[j, , ])@mmdstats[1]
        }

        err = tibble(w = w, mmd = mmd, method = name)
        err
    }
    
    # Compute baseline statistics (difference between xtrue and xtrue2)
    base_err = compute_stats(xtrue, xtrue2, name = "Baseline")
    
    # Compute errors
    names(mod) = c("TemperFlow", "TemperFlow + Rej.",
                    "MH", "HMC", "Parallel Tempering")
    methods = c("MH", "HMC", "Parallel Tempering", "TemperFlow", "TemperFlow + Rej.")
    n = length(mod)
    errs = vector("list", length = n)
    for(i in 1:n)
    {
        name = names(mod)[i]
        err = compute_stats(xtrue, mod[[name]], name)
        # Subtract baseline errors
        err$w = err$w - base_err$w
        err$mmd = err$mmd - base_err$mmd
        errs[[i]] = err
    }
    errs = do.call(rbind, errs)

    errs$method = factor(errs$method, levels = methods)
    errs
}

set.seed(123)
clayton8 = summarize_error(dim = 8)
clayton8$model = "Dim = 8"
clayton16 = summarize_error(dim = 16)
clayton16$model = "Dim = 16"
clayton32 = summarize_error(dim = 32)
clayton32$model = "Dim = 32"
clayton64 = summarize_error(dim = 64)
clayton64$model = "Dim = 64"

gdat = rbind(clayton8, clayton16, clayton32, clayton64)
gdat$model = factor(gdat$model, levels = sprintf("Dim = %d", c(8, 16, 32, 64)))
colnames(gdat) = c("Adjusted 1-Wasserstein", "Adjusted MMD", "method", "model")
gdat = melt(gdat, id.vars = c("method", "model"),
            variable.name = "metric", value.name = "err")
ggplot(gdat, aes(x = method, y = err)) +
    facet_grid(rows = vars(metric), cols = vars(model), scales = "free_y") +
    geom_boxplot(aes(color = method), width = 0.6) +
    scale_x_discrete("Method", labels = c("MH", "HMC", "PT", "TF", "TF+R")) +
    ylab("Sampling Error") + # ylim(0, NA) +
    scale_color_hue("Method", labels = c("MH = Metropolis-Hastings",
                                         "HMC = Hamiltonian Monte Carlo",
                                         "PT = Parallel tempering",
                                         "TF = TemperFlow",
                                         "TF+R = TemperFlow with rejection")) +
    theme_bw(base_size = 16, base_family = "Lato") +
    theme(panel.grid = element_blank(),
          axis.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/experiments_copula_errors_mcmc10.pdf", width = 15, height = 6)
