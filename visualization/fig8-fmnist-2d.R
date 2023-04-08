library(readr)
library(tibble)
library(ggplot2)
library(showtext)
font_add_google("Lato")

# Log density and scatterplot of the true distribution
logp = read_csv("../torch/model/fmnist-2d-logp.csv")
scatter = read_csv("../torch/model/fmnist-2d-scatter.csv")
logp$type = "Ground Truth"
scatter$type = "Ground Truth"

# Log density of the distribution learned by the KL flow
logp_kl = read_csv("../torch/model/fmnist-2d-logp-kl.csv")
scatter_kl = read_csv("../torch/model/fmnist-2d-scatter-kl.csv")
logp_kl$type = "KL Sampler"
scatter_kl$type = "KL Sampler"

# Log density of the distribution learned by TemperFlow
logp_temper = read_csv("../torch/model/fmnist-2d-logp-l2.csv")
scatter_temper = read_csv("../torch/model/fmnist-2d-scatter-l2.csv")
logp_temper$type = "TemperFlow"
scatter_temper$type = "TemperFlow"

# Combine results
logp = rbind(logp, logp_kl, logp_temper)
logp$type = factor(logp$type, levels = c("Ground Truth", "KL Sampler", "TemperFlow"))
scatter = rbind(scatter, scatter_kl, scatter_temper)
scatter$type = factor(scatter$type, levels = c("Ground Truth", "KL Sampler", "TemperFlow"))

# Visualization of log-density
ggplot(logp, aes(x = Z1, y = Z2)) +
    facet_grid(cols = vars(type)) +
    geom_contour_filled(aes(z = density), binwidth = 0.8) +
    geom_contour(aes(z = density), binwidth = 0.8, size = 0.1, color = "black") +
    guides(fill = "none") +
    xlab("Z1") + ylab("Z2") +
    coord_fixed(expand = FALSE) +
    theme_bw(base_size = 18, base_family = "Lato") +
    theme(axis.title = element_text(face = "bold"),
          legend.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/fmnist_2d_logp.pdf", width = 12.5, height = 5)

# Scatterplots
ggplot(scatter, aes(x = Z1, y = Z2)) +
    facet_grid(cols = vars(type)) +
    geom_point(size = 0.5, alpha = 0.1) +
    xlab("Z1") + ylab("Z2") +
    coord_fixed(xlim = c(-4, 4), ylim = c(-4, 4), expand = FALSE) +
    theme_bw(base_size = 18, base_family = "Lato") +
    theme(axis.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/fmnist_2d_scatter.pdf", width = 4.5, height = 4.5)
