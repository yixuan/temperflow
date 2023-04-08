library(reticulate)
library(tibble)
library(reshape2)
library(ggplot2)
library(showtext)

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

dat = read_npz("../tf/model/2d-trunc-sample.npz")
gdat = lapply(seq_along(dat$states), function(i) {
    d = dat$samples[i, 1:2000, ]
    colnames(d) = c("X1", "X2")
    d = as_tibble(d)
    d$state = dat$states[i]
    d
})
gdat = do.call(rbind, gdat)
gdat$state = factor(gdat$state, levels = dat$states)

ggplot(gdat, aes(x = X1, y = X2)) +
    facet_wrap(vars(state), nrow = 2) +
    geom_point(size = 0.1, alpha = 0.6, color = "steelblue") +
    scale_x_continuous(breaks = c(-5.0, -2.5, 0, 2.5, 5)) +
    scale_y_continuous(breaks = c(-5.0, -2.5, 0, 2.5, 5)) +
    coord_equal(xlim = c(-5.5, 5.5), ylim = c(-5.5, 5.5)) +
    theme_bw(base_size = 18, base_family = "sans")
showtext_auto()
ggsave("images/experiments_2d_trunc.pdf", width = 12, height = 6)
