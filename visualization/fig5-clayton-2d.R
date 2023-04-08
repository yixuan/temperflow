library(nor1mix)
library(copula)
library(rgl)
library(ggplot2)
library(showtext)
font_add_google("Lato")

# Marginal distribution
# 0.7 * N(-1, 0.2) + 0.3 * N(1, 0.2)
###########################################################################
# mpdf = function(x)
# {
#     0.7 * dnorm(x, mean = -1, sd = 0.2) + 0.3 * dnorm(x, mean = 1, sd = 0.2)
# }
# mcdf = function(x)
# {
#     0.7 * pnorm(x, mean = -1, sd = 0.2) + 0.3 * pnorm(x, mean = 1, sd = 0.2)
# }
# x0 = seq(-2.5, 2.5, length.out = 10000)
# u0 = mcdf(x0)
# micdf = splinefun(u0, x0)
###########################################################################
mix = norMix(mu = c(-1, 1), sigma = c(0.2, 0.2), w = c(0.7, 0.3))
mpdf = function(x) dnorMix(x, mix)
mcdf = function(x) pnorMix(x, mix)
micdf = function(x) qnorMix(x, mix)

curve(mpdf, -3, 3)
curve(mcdf, -3, 3)
curve(micdf, 0, 1, n = 1001)

# Simulate data
set.seed(123)
p = 8
theta = 2
nsim = 100 * 1000
cop = claytonCopula(param = theta, dim = p)
u = rCopula(nsim, cop)
u = micdf(u)
dim(u) = c(nsim, p)
# Visualize 2D data
plot(u[1:10000, 1], u[1:10000, 2], asp = 1)
# Visualize 3D data
plot3d(u[1:10000, 1], u[1:10000, 2], u[1:10000, 3], asp = 1)

# 2D scatterplot
gdat = as.data.frame(u[1:1000, 1:2])
ggplot(gdat, aes(x = V1, y = V2)) +
    geom_point(size = 2, alpha = 0.2) +
    xlab("Dimension 1") + ylab("Dimension 2") +
    coord_fixed(xlim = c(-2, 2), ylim = c(-2, 2), expand = FALSE) +
    theme_bw(base_size = 18, base_family = "Lato") +
    theme(axis.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/clayton_normal_mix.pdf", width = 4.5, height = 4.5)

# Visualize 2D density
cop = claytonCopula(param = theta, dim = 2)
ngrid = 200
u0 = seq(-2, 2, length.out = ngrid)
denu = data.frame(
    u1 = rep(u0, ngrid),
    u2 = rep(u0, each = ngrid),
    den = as.numeric(ngrid^2)
)
Fu = mcdf(c(denu$u1, denu$u2))
dim(Fu) = c(ngrid^2, 2)
denu$den = mpdf(denu$u1) * mpdf(denu$u2) * dCopula(Fu, cop)
ggplot(denu, aes(x = u1, y = u2)) +
    geom_raster(aes(fill = den)) +
    geom_contour(aes(z = den), color = "white", size = 0.1, alpha = 0.3, bins = 50) +
    scale_fill_distiller("Density", palette = "Spectral", direction = -1) +
    guides(fill = guide_colorbar(barheight = 15, order = 1)) +
    xlab("Dimension 1") + ylab("Dimension 2") +
    coord_fixed(expand = FALSE) +
    theme_bw(base_size = 18, base_family = "Lato") +
    theme(axis.title = element_text(face = "bold"),
          legend.title = element_text(face = "bold"))
showtext_auto()
ggsave("images/clayton_normal_mix_den.pdf", width = 6, height = 4.5)
