library(reticulate)
library(tibble)
library(ggplot2)
library(GGally)
library(grid)
library(gridExtra)
library(showtext)
font_add_google("Lato")

py_config()
np = import("numpy")

source("clayton.R")

# Simulate data
set.seed(123)
p = 8
theta = 2
nsim = 100 * 1000
distr = ClaytonNormalMix$new(dim = p, mix_dim = 8, theta = theta)
xtrue = distr$sample(nsim)
dim(xtrue) = c(100, 1000, p)

# Scatterplot matrix of true samples
lowerfun = function(data, mapping)
{
    ggplot(data = data, mapping = mapping) +
        geom_point(size = 0.2, alpha = 0.1) +
        scale_x_continuous(limits = c(-2, 2), expand = expansion()) +
        scale_y_continuous(limits = c(-2, 2), expand = expansion())
}
upperfun = function(data, mapping)
{
    ggplot(data = data, mapping = mapping) +
        geom_density2d(h = 0.4, alpha = 0.7) +
        scale_x_continuous(limits = c(-2, 2), expand = expansion()) +
        scale_y_continuous(limits = c(-2, 2), expand = expansion())
}
gdat = as.data.frame(xtrue[1, , ])
names(gdat) = paste0("X", 1:ncol(gdat))
ggpairs(gdat,
        lower = list(continuous = wrap(lowerfun)),
        upper = list(continuous = wrap(upperfun)),
        diag = list(continuous = wrap("barDiag", binwidth = 0.1, fill = "#f8766d"))) +
    theme_bw(base_size = 12)



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

# Scatterplot matrix of TemperFlow samples
dat = read_npz("../tf/model/clayton-dim8.npz")
gdat = as.data.frame(dat[["xacc"]][1, , ])
names(gdat) = paste0("X", 1:ncol(gdat))
g = ggpairs(gdat,
        lower = list(continuous = wrap(lowerfun)),
        upper = list(continuous = wrap(upperfun)),
        diag = list(continuous = wrap("barDiag", binwidth = 0.1, fill = "#f8766d"))) +
    theme_bw(base_size = 12, base_family = "Lato")
showtext_auto()
pdf("images/experiments_copula_tf.pdf", width = 11, height = 10)
label = textGrob("TemperFlow", vjust = 0, rot = -90,
                 gp = gpar(fontsize = 28, fontface = "bold"))
grid.arrange(ggmatrix_gtable(g), right = label, padding = unit(2, "line"))
dev.off()

# Scatterplot matrix of parallel tempering samples
gdat = as.data.frame(dat[["pt"]][1, , ])
names(gdat) = paste0("X", 1:ncol(gdat))
g = ggpairs(gdat,
        lower = list(continuous = wrap(lowerfun)),
        upper = list(continuous = wrap(upperfun)),
        diag = list(continuous = wrap("barDiag", binwidth = 0.1, fill = "#f8766d"))) +
    theme_bw(base_size = 12, base_family = "Lato")
showtext_auto()
pdf("images/experiments_copula_pt.pdf", width = 11, height = 10)
label = textGrob("Parallel Tempering", vjust = 0, rot = -90,
                 gp = gpar(fontsize = 28, fontface = "bold"))
grid.arrange(ggmatrix_gtable(g), right = label, padding = unit(2, "line"))
dev.off()
