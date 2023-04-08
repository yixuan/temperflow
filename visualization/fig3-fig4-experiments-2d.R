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
summarize_result = function(model_name)
{
    nsamp = 1000
    # Read in TemperFlow and MCMC samples
    dat = read_npz(paste0("../tf/model/", model_name, ".npz"))
    mod_all = lapply(dat, function(x) x[1, 1:nsamp, ])
    mod = mod_all[names(mod_all) != "xtrue2"]
    names(mod) = c("True", "TemperFlow", "TemperFlow + Rej.",
                   "MH", "HMC", "Parallel Tempering")
    n = length(mod)
    dat = vector("list", length = n)
    for(i in 1:n)
    {
        dat[[i]] = tibble(
            x = mod[[i]][, 1],
            y = mod[[i]][, 2],
            method = names(mod)[i]
        )
    }
    dat = do.call(rbind, dat)

    methods = c("MH", "HMC", "Parallel Tempering",
                "TemperFlow", "TemperFlow + Rej.", "True")
    dat$method = factor(dat$method, levels = methods)
    
    # Compute baseline statistics (difference between xtrue and xtrue2)
    w0 = wasserstein(pp(mod_all[["xtrue"]]), pp(mod_all[["xtrue2"]]))
    mmd0 = kmmd(mod_all[["xtrue"]], mod_all[["xtrue2"]])@mmdstats[1]

    # Compute errors
    err = vector("list", length = n)
    for(i in 1:n)
    {
        name = names(mod)[i]
        if(name == "True")
        {
            text = ""
        } else {
            w = wasserstein(pp(mod[[name]]), pp(mod[["True"]]))
            mmd = kmmd(mod[[name]], mod[["True"]])@mmdstats[1]
            text = sprintf("Adj. W = %.3f\nAdj. MMD = %.3f", w - w0, mmd - mmd0)
        }
        err[[i]] = tibble(
            text = text,
            method = names(mod)[i]
        )
    }
    err = do.call(rbind, err)
    err$method = factor(err$method, levels = methods)
    list(dat = dat, err = err)
}

set.seed(123)
circle = summarize_result("circle")
circle$dat$model = "Circle"
circle$err$model = "Circle"
grid = summarize_result("grid")
grid$dat$model = "Grid"
grid$err$model = "Grid"
cross = summarize_result("cross")
cross$dat$model = "Cross"
cross$err$model = "Cross"

gdat = rbind(circle$dat, grid$dat, cross$dat)
gtext = rbind(circle$err, grid$err, cross$err)
gtext$x = 0
gtext$y = -9
ggplot(gdat, aes(x = x, y = y)) +
    facet_grid(rows = vars(model), cols = vars(method), scales = "fixed") +
    geom_point(size = 0.1, alpha = 0.6, color = "steelblue") +
    geom_text(aes(label = text), data = gtext, size = 5) +
    coord_equal(xlim = c(-7, 7), ylim = c(-10.5, 7)) +
    theme_bw(base_size = 18, base_family = "Lato") +
    theme(panel.grid = element_blank())
showtext_auto()
ggsave("images/experiments_2d.pdf", width = 15, height = 9.5)



summarize_error = function(model_name)
{
    nsamp = 1000
    # Read in TemperFlow and MCMC samples
    dat = read_npz(paste0("../tf/model/", model_name, ".npz"))
    # True samples
    xtrue = dat[["xtrue"]]
    xtrue2 = dat[["xtrue2"]]
    # Other samples
    dat = dat[(names(dat) != "xtrue") & (names(dat) != "xtrue2")]
    mod = lapply(dat, function(x) x[, 1:nsamp, ])
    
    # Compute baseline statistics (difference between xtrue and xtrue2)
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
circle = summarize_error("circle")
circle$model = "Circle"
grid = summarize_error("grid")
grid$model = "Grid"
cross = summarize_error("cross")
cross$model = "Cross"

gdat = rbind(circle, grid, cross)
ggplot(gdat, aes(x = w, y = method)) +
    facet_grid(cols = vars(model), scales = "fixed") +
    geom_boxplot(aes(color = method)) +
    xlab("1-Wasserstein Distance") + ylab("Method") +
    guides(color = "none") +
    theme_bw(base_size = 18, base_family = "Lato") +
    theme(panel.grid = element_blank(),
          axis.title = element_text(face = "bold"))

colnames(gdat) = c("Adjusted 1-Wasserstein", "Adjusted MMD", "method", "model")
gdat = melt(gdat, id.vars = c("method", "model"),
            variable.name = "metric", value.name = "err")
ggplot(gdat, aes(x = method, y = err)) +
    facet_grid(rows = vars(metric), cols = vars(model), scales = "free_y") +
    geom_boxplot(aes(color = method), width = 0.6) +
    scale_x_discrete("Method", labels = c("MH", "HMC", "PT", "TF", "TF+R")) +
    ylab("Sampling Error") +
    scale_color_hue("Method", labels = c("MH = Metropolis-Hastings",
                                         "HMC = Hamiltonian Monte Carlo",
                                         "PT = Parallel tempering",
                                         "TF = TemperFlow",
                                         "TF+R = TemperFlow with rejection")) +
    theme_bw(base_size = 16, base_family = "Lato") +
    theme(panel.grid = element_blank(),
          axis.title = element_text(face = "bold"),
          strip.text.y = element_text(size = 12))
showtext_auto()
ggsave("images/experiments_2d_errors.pdf", width = 12, height = 5)
