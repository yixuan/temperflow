library(reticulate)
library(tibble)
library(ggplot2)
library(showtext)
font_add_google("Lato")

py_config()
np = import("numpy")

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

dat = read_npz("cache/fig2-l2-sampler.npz")
plot(dat$x, dat$den, type = "l", ylim = c(0, 0.45))
lines(dat$x, dat$dens_kl[1, ], col = "blue")       # iter 0
lines(dat$x, dat$dens_kl[2, ], col = "green")      # iter 10
lines(dat$x, dat$dens_kl[3, ], col = "orange")     # iter 20
lines(dat$x, dat$dens_kl[6, ], col = "red")        # iter 50
lines(dat$x, dat$dens_kl[11, ], col = "grey")      # iter 100

n = length(dat$x)
iters = c(1, 2, 3, 6, 11)
npanel = length(iters)
gdat1 = tibble(
    x = rep(dat$x, 2 * npanel),
    f = c(rep(dat$den, npanel), as.numeric(t(dat$dens_kl[iters, ]))),
    type = rep(c("Target Density", "Estimated Density"), each = n * npanel),
    iter = rep(rep(sprintf("Iter %d", 10 * (iters - 1)), each = n), 2)
)
gdat1$distr = "KL Gradient Flow"

plot(dat$x, dat$den, type = "l", ylim = c(0, 0.45))
lines(dat$x, dat$dens_l2[1, ], col = "blue")       # iter 0
lines(dat$x, dat$dens_l2[2, ], col = "green")      # iter 10
lines(dat$x, dat$dens_l2[3, ], col = "orange")     # iter 20
lines(dat$x, dat$dens_l2[6, ], col = "red")        # iter 50
lines(dat$x, dat$dens_l2[11, ], col = "grey")      # iter 100

n = length(dat$x)
gdat2 = tibble(
    x = rep(dat$x, 2 * npanel),
    f = c(rep(dat$den, npanel), as.numeric(t(dat$dens_l2[iters, ]))),
    type = rep(c("Target Density", "Estimated Density"), each = n * npanel),
    iter = rep(rep(sprintf("Iter %d", 10 * (iters - 1)), each = n), 2)
)
gdat2$distr = "L2 Gradient Flow"

gdat = rbind(gdat1, gdat2)
gdat$distr = factor(gdat$distr, levels = c("KL Gradient Flow", "L2 Gradient Flow"))
gdat$type = factor(gdat$type, levels = c("Target Density", "Estimated Density"))
gdat$iter = factor(gdat$iter, levels = sprintf("Iter %d", 10 * (iters - 1)))

ggplot(gdat, aes(x = x, y = f)) +
    facet_grid(rows = vars(distr), cols = vars(iter)) +
    geom_line(aes(color = type, group = type, linetype = type), size = 1) +
    scale_color_hue("Distribution") +
    scale_linetype_manual("Distribution", values = c("21", "solid")) +
    guides(color = guide_legend(keywidth = 2, keyheight = 2)) +
    xlab("x") + ylab("Density") +
    theme_bw(base_family = "Lato", base_size = 20) +
    theme(legend.position = "bottom",
          legend.title = element_text(face = "bold"),
          legend.box.margin = margin(),
          legend.margin = margin(),
          axis.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/demo_l2.pdf", width = 15, height = 6)
