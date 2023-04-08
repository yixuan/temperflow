import math
import os
import time
import numpy as np
import torch
import torch.nn as nn

# An abstract class that returns the tempered energy function value
# Users need to implement the energy() member function
class Energy(nn.Module):
    def __init__(self):
        super(Energy, self).__init__()

    def energy(self, x, **kwargs):
        raise NotImplementedError

    def tempered_energy(self, x, beta, **kwargs):
        return beta * self.energy(x, **kwargs)

    def forward(self, x, beta, **kwargs):
        return self.tempered_energy(x, beta, **kwargs)

# f(x) = log|exp(x) - 1|
# For x > 0, f(x) = x + log(1 - exp(-x))
# For x < 0, f(x) = log(1 - exp(x))
# Therefore, f(x) = max(x, 0) + log(1 - exp(-|x|))
@torch.jit.script
def log_expm1(x):
    return torch.relu(x) + torch.log(1.000001 - torch.exp(-torch.abs(x)))

# Exponential moving average
def exp_move_avg(x, fac=0.9):
    x = np.array(x)
    n = x.size
    if n < 1:
        return None
    i = np.linspace(n - 1, 0, num=n)
    weights = fac ** i
    return np.sum(x * weights) / np.sum(weights)

def min_max_grad(params):
    min_grad = 1e10
    max_grad = -1e10
    for param in params:
        min_grad = np.fmin(min_grad, torch.min(param.grad).item())
        max_grad = np.fmax(max_grad, torch.max(param.grad).item())
    return min_grad, max_grad

# Training a tempered distribution flow
class TemperFlow(nn.Module):
    def __init__(self, energy, bijector):
        super(TemperFlow, self).__init__()
        self.energy = energy
        self.bijector = bijector

    def kl_variance(self, beta, bs, ref_bij=None):
        with torch.no_grad():
            if ref_bij is None:
                theta = self.bijector.distr.sample((bs,))
            else:
                theta = ref_bij.distr.sample((bs,))
                logr = ref_bij.distr.log_prob(theta)  # .clip_(-30.0, 10.0)

            E = self.energy(theta, 1.0)
            Ebeta = -self.bijector.distr.log_prob(theta)  # .clip_(-30.0, 10.0)
            ratio = 1.0 if ref_bij is None else torch.exp(-Ebeta - logr)
            dEbeta = Ebeta / beta
            dEbeta = dEbeta - torch.mean(ratio * dEbeta)
            Ediff = E - Ebeta

            ratio_Ediff = ratio * Ediff
            m1 = torch.mean(ratio_Ediff)
            # We don't directly compute nd.square(Ediff) since it may overflow
            m2 = torch.mean(ratio_Ediff * Ediff)
            var = m2 - m1 * m1

            ratio_dEbeta_Ediff = ratio_Ediff * dEbeta
            dm1beta = -torch.mean(ratio_dEbeta_Ediff)
            dm2beta = -torch.mean(ratio_dEbeta_Ediff * Ediff) + 2.0 * dm1beta
            grad = dm2beta - 2.0 * m1 * dm1beta
        return var.item(), grad.item()

    def kl_grad(self, beta, bs, ref_bij=None):
        with torch.no_grad():
            ref = self.bijector.distr if ref_bij is None else ref_bij.distr
            theta = ref.sample((bs,))
            logr = ref.log_prob(theta)
            E = self.energy.energy(theta)

            logn = math.log(bs)
            log_ratio = -beta * E - logr
            zeta_1 = torch.logsumexp(-E - logr, dim=0) - logn
            zeta_beta = torch.logsumexp(log_ratio, dim=0) - logn
            ratio = torch.softmax(log_ratio - zeta_beta, dim=0) * bs

            Emean = torch.mean(ratio * E)
            kl = (1.0 - beta) * Emean + zeta_1 - zeta_beta
            grad = (beta - 1.0) * torch.mean(ratio * torch.square(E - Emean))
        return kl.item(), grad.item()

    def sampling(self, beta, bs, ref_bij=None):
        if ref_bij is None:
            # Simple Monte Carlo estimator
            theta = self.bijector.distr.rsample((bs,))
            logq = self.bijector.distr.log_prob(theta)
            logr = logq
        else:
            # Importance sampling estimator
            with torch.no_grad():
                theta = ref_bij.distr.sample((bs,))
                logr = ref_bij.distr.log_prob(theta)
            logq = self.bijector.distr.log_prob(theta)

        energy = self.energy(theta, beta)
        return energy, logr, logq

    @staticmethod
    def log_partition(logr, energy, oldval=None, decay=0.8):
        bs = logr.nelement()
        with torch.no_grad():
            logz = torch.logsumexp(-energy - logr, dim=0) - math.log(bs)
        if oldval is not None:
            logz = decay * logz + (1.0 - decay) * oldval
        return logz


    @staticmethod
    def scale_ref_distr(ref_bij, scale):
        if ref_bij is None:
            return None
        try:
            ref_bij.distr.base_dist.scale.fill_(scale)
        except:
            pass
        try:
            ref_bij.distr.base_dist.base_dist.scale.fill_(scale)
        except:
            pass

    def train_one_beta(self, beta, fn="kl", ref_bij=None, bs=1000, nepoch=1001, lr=0.001, verbose=False):
        # ref_scale = math.sqrt(2.0 - beta)
        # self.scale_ref_distr(ref_bij, scale=ref_scale)

        opt = torch.optim.Adam(self.bijector.parameters(), lr=lr, amsgrad=False)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=nepoch, cycle_momentum=False)

        losses = []
        logz = None

        for i in range(nepoch):
            energy, logr, logq = self.sampling(beta, bs, ref_bij)
            
            if "l2" in fn:
                # ||f-g||^2 = E_r[(f-g)^2/r], g is the target distribution
                # log[(f-g)^2/r] = 2 * log|f-g| - log(r) = 2 * log(f) + 2 * log|1 - g/f| - log(r)
                # z = E_r[exp(-energy) / r] = E_r[exp(-energy - logr)]
                logz = self.log_partition(logr, energy, oldval=logz, decay=0.8)
                logg = -energy - logz
                w = 2.0 * torch.maximum(logg, logq) + 2.0 * log_expm1(logg - logq) - logr
                log_l2 = torch.logsumexp(w, dim=0) - math.log(bs)
                loss_l2 = log_l2
                # loss_l2 = torch.exp(log_l2 - normalizer) + normalizer - 1.0
            if "kl" in fn or "h2" in fn:
                logratio = 0.0 if ref_bij is None else logq - logr
                ratio = 1.0 if ref_bij is None else torch.exp(logratio)

            if fn == "kl":
                loss = torch.mean(ratio * (logq + energy))
            elif fn == "mh2":
                logz = self.log_partition(logr, energy, oldval=logz, decay=0.8)
                logg = -energy - logz
                # loss = torch.mean(ratio * torch.square(logg / logq - 1.0))
                ratio2 = torch.exp(logg - logr)
                loss = torch.mean(ratio2 * torch.square(logq / logg - 1.0))
            elif fn == "h2":
                # loss = torch.mean(ratio * (2.0 - 2.0 * torch.exp(-0.5 * (logq + energy))))
                # loss = torch.mean(ratio * torch.square(torch.exp(-0.5 * (logq + energy)) - 1.0))
                t = -0.5 * (logq + energy)
                loss = torch.mean(torch.exp(logratio + 2.0 * t) - 2.0 * torch.exp(logratio + t) + ratio)
            elif fn == "l2":
                loss = loss_l2
            elif fn == "l2+kl":
                loss_kl = torch.mean(ratio * (logq - logg))
                loss = loss_l2 + loss_kl
            elif fn == "l2+h2":
                # loss_h2 = torch.mean(ratio * (2.0 - 2.0 * torch.exp(-0.5 * (logq - logg))))
                # loss_h2 = torch.mean(ratio * torch.square(torch.exp(-0.5 * (logq - logg)) - 1.0))
                t = -0.5 * (logq - logg)
                loss_h2 = torch.mean(torch.exp(logratio + 2.0 * t) - 2.0 * torch.exp(logratio + t) + ratio)
                loss = loss_l2 + loss_h2
            else:
                raise NotImplementedError

            opt.zero_grad()
            loss.backward()
            # print(min_max_grad(self.bijector.parameters()))
            # nn.utils.clip_grad_value_(self.bijector.parameters(), 1.0)
            opt.step()
            scheduler.step()

            lossi = loss.item()
            losses.append(lossi)
            if verbose and i % 50 == 0:
                if fn == "l2" or fn == "mh2":
                    print(f"epoch {i}, logz = {logz}, loss = {loss.item()}")
                elif fn == "l2+h2":
                    print(f"epoch {i}, logz = {logz}, loss_l2 = {loss_l2.item()}, loss_h2 = {loss_h2.item()}, loss = {lossi}")
                elif fn == "l2+kl":
                    print(f"epoch {i}, logz = {logz}, loss_l2 = {loss_l2.item()}, loss_kl = {loss_kl.item()}, loss = {lossi}")
                else:
                    print(f"epoch {i}, loss = {lossi}")

        # self.scale_ref_distr(ref_bij, scale=1.0)
        return losses

    def train_flow(self, beta0=0.1, kl_decay=0.8, nepoch=(2001, 2001, 2001),
                   bs=1000, lr=0.0001, max_betas=100, verbose=1,
                   ref=None, ref_cutoff=[0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.01, 0.001],
                   checkpoint=None, recover=False):
        t0 = time.time()
        # Epochs at different stages
        if type(nepoch) is tuple:
            nepoch_init, nepoch_mid, nepoch_final = nepoch
        else:
            nepoch_init, nepoch_mid, nepoch_final = nepoch, nepoch, nepoch

        # History of log(beta)
        logbeta = math.log(beta0)
        logbetas = [logbeta]
        deltas = []
        # History of KL divergence
        kls = []
        # History of losses
        losses = []
        # Elapsed times
        times = []

        # Reference distributions for importance sampling
        # ref = None
        ref_cutoff = ref_cutoff.copy()

        for i in range(max_betas):
            t1 = time.time()
            beta = math.exp(logbeta)
            print(f"i = {i}, logbeta = {logbeta}, beta = {beta}\n") if verbose > 0 else None

            if recover and os.path.isfile(f"{checkpoint}_{i}.pt"):
                self.load_state_dict(torch.load(f"{checkpoint}_{i}.pt"))
            else:
                # Train one tempered distribution
                # Heuristic choice of adaptive learning rate
                lri = lr / math.sqrt(beta) * math.pow(0.9, i)
                fni, nepochi = ("kl", nepoch_init) if i == 0 else ("l2", nepoch_mid)
                lossesi = self.train_one_beta(beta, fn=fni, ref_bij=ref, bs=bs, nepoch=nepochi, lr=lri, verbose=(verbose > 1))
                losses.append(lossesi)

            if checkpoint is not None:
                torch.save(self.state_dict(), f"{checkpoint}_{i}.pt")

            # Update reference distribution after the first iteration
            ref = self.bijector.clone() if i == 0 else ref

            # Estimate KL divergence
            kl, kl_grad = self.kl_grad(beta, bs=10 * bs, ref_bij=ref)
            kls.append(float(kl))

            # Compute delta
            delta_abs_avg = exp_move_avg(np.abs(deltas), fac=0.9)
            delta = -(1 - kl_decay) * kl / beta / kl_grad
            delta = np.minimum(delta, 1.0) if i == 0 else delta
            if delta_abs_avg is not None:
                delta = np.minimum(math.fabs(delta), 2.0 * delta_abs_avg)
                delta = np.maximum(delta, 0.5 * delta_abs_avg)
            deltas.append(delta)
            t2 = time.time()
            times.append(t2 - t1)
            if verbose > 0:
                print(f"\nkl = {kl}, kl_grad = {kl_grad}")
                print(f"delta = {delta}, delta_avg = {delta_abs_avg}")
                print(f"{t2 - t1} seconds\n")

            # Update beta
            logbeta += delta
            if logbeta >= 0.0:
                break
            logbetas.append(logbeta)

            if len(ref_cutoff) > 0:
                if kl < ref_cutoff[0] * kls[0]:
                    print("\n****** Reference distribution updated ******\n") if verbose > 0 else None
                    ref = self.bijector.clone()
                    ref_cutoff = ref_cutoff[1:]

        # Final refining with beta=1
        t1 = time.time()
        print("\n****** Final training with beta=1 ******\n") if verbose > 0 else None
        lri = lr * math.pow(0.9, i + 1)
        lossesi1 = self.train_one_beta(beta=1.0, fn="l2", ref_bij=ref, bs=bs, nepoch=nepoch_final // 2, lr=lri, verbose=(verbose > 1))
        lossesi2 = self.train_one_beta(beta=1.0, fn="kl", ref_bij=None, bs=bs, nepoch=nepoch_final // 2, lr=lri, verbose=(verbose > 1))
        t2 = time.time()
        losses.append(lossesi1)
        losses.append(lossesi2)
        times.append(t2 - t1)

        trace = dict(logbeta=logbetas, kl=kls,
                     loss_init=losses[0], loss=losses[1:],
                     time=times, total_time=t2 - t0)
        return trace
