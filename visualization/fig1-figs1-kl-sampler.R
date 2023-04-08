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

dat = read_npz("cache/fig1-figs1-kl-sampler.npz")
plot(dat$x, dat$denm, type = "l", ylim = c(0, 0.45))
lines(dat$x, dat$densm[1, ], col = "blue")
lines(dat$x, dat$densm[2, ], col = "purple")
lines(dat$x, dat$densm[6, ], col = "green")
lines(dat$x, dat$densm[11, ], col = "orange")
lines(dat$x, dat$densm[51, ], col = "red")

n = length(dat$x)
iters = c(1, 2, 6, 11, 51)
npanel = length(iters)
gdat1 = tibble(
    x = rep(dat$x, 2 * npanel),
    f = c(rep(dat$denm, npanel), as.numeric(t(dat$densm[iters, ]))),
    type = rep(c("Target", "Estimated"), each = n * npanel),
    iter = rep(rep(sprintf("Iter %d", 10 * (iters - 1)), each = n), 2)
)
gdat1$distr = "Multimodal"

plot(dat$x, dat$denu, type = "l", ylim = c(0, 0.45))
# curve(exp(x - exp(x / 3)) / 6, -5, 10, add = TRUE, col = "grey")
lines(dat$x, dat$densu[1, ], col = "blue")
lines(dat$x, dat$densu[2, ], col = "purple")
lines(dat$x, dat$densu[6, ], col = "green")
lines(dat$x, dat$densu[11, ], col = "orange")
lines(dat$x, dat$densu[51, ], col = "red")

n = length(dat$x)
gdat2 = tibble(
    x = rep(dat$x, 2 * npanel),
    f = c(rep(dat$denu, npanel), as.numeric(t(dat$densu[iters, ]))),
    type = rep(c("Target", "Estimated"), each = n * npanel),
    iter = rep(rep(sprintf("Iter %d", 10 * (iters - 1)), each = n), 2)
)
gdat2$distr = "Unimodal"

gdat = rbind(gdat1, gdat2)
gdat$distr = factor(gdat$distr, levels = c("Unimodal", "Multimodal"))
gdat$type = factor(gdat$type, levels = c("Target", "Estimated"))
gdat$iter = factor(gdat$iter, levels = sprintf("Iter %d", 10 * (iters - 1)))

ggplot(gdat, aes(x = x, y = f)) +
    facet_grid(rows = vars(distr), cols = vars(iter)) +
    geom_line(aes(color = type, group = type, linetype = type), size = 1) +
    scale_color_hue("Distribution") +
    scale_linetype_manual("Distribution", values = c("21", "solid")) +
    guides(color = guide_legend(keywidth = 2, keyheight = 2)) +
    xlab("x") + ylab("Density") +
    theme_bw(base_family = "Lato", base_size = 20) +
    theme(legend.title = element_text(face = "bold"),
          axis.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/demo_uni_multi_modal.pdf", width = 16, height = 6)



# Smaller gap
n = length(dat$x)
gdat6 = tibble(
    x = rep(dat$x, 2 * npanel),
    f = c(rep(dat$denm6, npanel), as.numeric(t(dat$densm6[iters, ]))),
    type = rep(c("Target", "Estimated"), each = n * npanel),
    iter = rep(rep(sprintf("Iter %d", 10 * (iters - 1)), each = n), 2)
)
gdat6$distr = "Mean Gap = 6"
gdat4 = tibble(
    x = rep(dat$x, 2 * npanel),
    f = c(rep(dat$denm4, npanel), as.numeric(t(dat$densm4[iters, ]))),
    type = rep(c("Target", "Estimated"), each = n * npanel),
    iter = rep(rep(sprintf("Iter %d", 10 * (iters - 1)), each = n), 2)
)
gdat4$distr = "Mean Gap = 4"

gdat8 = gdat1
gdat8$distr = "Mean Gap = 8"

gdat = rbind(gdat8, gdat6, gdat4)
gdat$distr = factor(gdat$distr, levels = c("Mean Gap = 8", "Mean Gap = 6", "Mean Gap = 4"))
gdat$type = factor(gdat$type, levels = c("Target", "Estimated"))
gdat$iter = factor(gdat$iter, levels = sprintf("Iter %d", 10 * (iters - 1)))

ggplot(gdat, aes(x = x, y = f)) +
    facet_grid(rows = vars(distr), cols = vars(iter)) +
    geom_line(aes(color = type, group = type, linetype = type), size = 1) +
    scale_color_hue("Distribution") +
    scale_linetype_manual("Distribution", values = c("21", "solid")) +
    guides(color = guide_legend(keywidth = 2, keyheight = 2)) +
    xlab("x") + ylab("Density") +
    theme_bw(base_family = "Lato", base_size = 20) +
    theme(legend.title = element_text(face = "bold"),
          axis.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/demo_multi_modal_gaps.pdf", width = 16, height = 8)
