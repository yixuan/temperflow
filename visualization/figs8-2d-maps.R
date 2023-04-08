library(reticulate)
library(tibble)
library(reshape2)
library(dplyr)
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

maps = c("affine", "lrspline", "qrspline")
map_names = c("Inverse autoregressive flow",
              "Linear rational spline",
              "Quadratic rational spline")
datasets = c("circle", "grid", "cross")
dataset_names = c("Circle", "Grid", "Cross")
gdat = list()
for(i in seq_along(datasets))
{
    dataset = datasets[i]
    for(j in seq_along(maps))
    {
        map = maps[j]
        filename = sprintf("../tf/model/%s-%s-sample.npz", dataset, map)
        dat = read_npz(filename)$samp[1:1000, ]
        colnames(dat) = c("X1", "X2")
        dat = as_tibble(dat) %>%
            mutate(dataset = dataset_names[i], map = map_names[j])
        gdat = c(gdat, list(dat))
    }
}
gdat = do.call(rbind, gdat)
gdat$map = factor(gdat$map, levels = map_names)
ggplot(gdat, aes(x = X1, y = X2)) +
    facet_grid(rows = vars(dataset), cols = vars(map)) +
    geom_point(size = 0.1, alpha = 0.6, color = "steelblue") +
    coord_equal(xlim = c(-10, 10), ylim = c(-6, 6)) +
    theme_bw(base_size = 18, base_family = "Lato")
showtext_auto()
ggsave("images/experiments_2d_maps.pdf", width = 10, height = 6)
