# -*- coding: utf-8 -*-
"""PARF: Polarity-Aware Ordinal Relation Fusion.


"""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


# ---------------------------------------------------------------------------
# Basic filters (copied from the reference implementation)
# ---------------------------------------------------------------------------

def box_filter(x: torch.Tensor, radius: int) -> torch.Tensor:
    """Box mean filter implemented with avg_pool2d. x: (B, C, H, W)."""
    kernel_size = 2 * radius + 1
    pad = radius
    x_padded = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    return F.avg_pool2d(x_padded, kernel_size=kernel_size, stride=1)


def guided_filter(guide: torch.Tensor, src: torch.Tensor,
                  radius: int, eps: float) -> torch.Tensor:
    """Self-guided edge-preserving filter."""
    mean_I = box_filter(guide, radius)
    mean_p = box_filter(src, radius)
    corr_I = box_filter(guide * guide, radius)
    corr_Ip = box_filter(guide * src, radius)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, radius)
    mean_b = box_filter(b, radius)
    return mean_a * guide + mean_b


def compute_saliency(ir: torch.Tensor, radius: int = 15,
                     lambda_s: float = 0.5,
                     lambda_g: float = 0.2) -> torch.Tensor:
    """IR thermal saliency S_ir in [0, 1]."""
    local_mean = box_filter(ir, radius)
    intensity_sal = torch.abs(ir - local_mean)

    grad_x = torch.abs(ir[:, :, :, 1:] - ir[:, :, :, :-1])
    grad_y = torch.abs(ir[:, :, 1:, :] - ir[:, :, :-1, :])
    grad_x = F.pad(grad_x, (1, 0, 0, 0), mode="replicate")
    grad_y = F.pad(grad_y, (0, 0, 1, 0), mode="replicate")
    grad_mag = grad_x + grad_y

    sal_raw = lambda_s * intensity_sal + lambda_g * grad_mag
    b = sal_raw.shape[0]
    flat = sal_raw.reshape(b, -1)
    vmin = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    vmax = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    return (sal_raw - vmin) / (vmax - vmin + 1e-8)


class TraditionalFusion(nn.Module):
    """Guided-filter decomposition and classical base/detail fusion."""

    def __init__(self, gf_radius: int = 8, gf_eps: float = 0.01,
                 saliency_radius: int = 15, lambda_s: float = 0.5,
                 lambda_g: float = 0.2, base_eps: float = 1e-6,
                 detail_eps: float = 1e-6):
        super().__init__()
        self.gf_radius = gf_radius
        self.gf_eps = gf_eps
        self.saliency_radius = saliency_radius
        self.lambda_s = lambda_s
        self.lambda_g = lambda_g
        self.base_eps = base_eps
        self.detail_eps = detail_eps

    def _decompose(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = guided_filter(img, img, self.gf_radius, self.gf_eps)
        D = img - B
        return B, D

    def forward(self, ir: torch.Tensor, vis: torch.Tensor):
        B_ir, D_ir = self._decompose(ir)
        B_vis, D_vis = self._decompose(vis)

        S_ir = compute_saliency(ir, self.saliency_radius,
                                self.lambda_s, self.lambda_g)

        E_ir_B = box_filter(B_ir * B_ir, self.gf_radius)
        E_vis_B = box_filter(B_vis * B_vis, self.gf_radius)

        W_ir_B = (E_ir_B + self.lambda_s * S_ir + self.base_eps) / (
            E_ir_B + E_vis_B + self.lambda_s * S_ir + 2 * self.base_eps
        )
        W_vis_B = 1.0 - W_ir_B
        B_f = W_ir_B * B_ir + W_vis_B * B_vis

        grad_D_ir_x = D_ir[:, :, :, 1:] - D_ir[:, :, :, :-1]
        grad_D_ir_y = D_ir[:, :, 1:, :] - D_ir[:, :, :-1, :]
        grad_D_vis_x = D_vis[:, :, :, 1:] - D_vis[:, :, :, :-1]
        grad_D_vis_y = D_vis[:, :, 1:, :] - D_vis[:, :, :-1, :]

        grad_D_ir = F.pad(grad_D_ir_x, (1, 0, 0, 0)) ** 2 \
            + F.pad(grad_D_ir_y, (0, 0, 1, 0)) ** 2
        grad_D_vis = F.pad(grad_D_vis_x, (1, 0, 0, 0)) ** 2 \
            + F.pad(grad_D_vis_y, (0, 0, 1, 0)) ** 2

        E_ir_D = box_filter(grad_D_ir, self.gf_radius)
        E_vis_D = box_filter(grad_D_vis, self.gf_radius)

        W_ir_D = E_ir_D / (E_ir_D + E_vis_D + self.detail_eps)
        W_vis_D = 1.0 - W_ir_D
        D_f = W_ir_D * D_ir + W_vis_D * D_vis

        F_0 = torch.clamp(B_f + D_f, 0.0, 1.0)
        aux = {
            "B_ir": B_ir, "B_vis": B_vis,
            "D_ir": D_ir, "D_vis": D_vis,
            "S_ir": S_ir, "B_f": B_f, "D_f": D_f,
            "W_ir_B": W_ir_B, "W_ir_D": W_ir_D,
        }
        return F_0, aux


# ---------------------------------------------------------------------------
# Riesz transform and signed relation field
# ---------------------------------------------------------------------------

class RieszFilterBank(nn.Module):
    """Frequency-domain Riesz transform H1=-i u/|f|, H2=-i v/|f|."""

    def __init__(self):
        super().__init__()
        self._cached_size = None
        self._h1 = None
        self._h2 = None

    def _build(self, h: int, w: int, device: torch.device):
        fy = torch.fft.fftfreq(h, d=1.0, device=device).view(h, 1)
        fx = torch.fft.fftfreq(w, d=1.0, device=device).view(1, w)
        freq_mag = torch.sqrt(fx ** 2 + fy ** 2)
        freq_mag[0, 0] = 1.0
        h1 = (-1j * fx / freq_mag).to(torch.complex64)
        h2 = (-1j * fy / freq_mag).to(torch.complex64)
        h1[0, 0] = 0
        h2[0, 0] = 0
        self._h1 = h1
        self._h2 = h2
        self._cached_size = (h, w, device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        if self._cached_size != (h, w, x.device):
            self._build(h, w, x.device)
        x_freq = torch.fft.fft2(x)
        r1 = torch.fft.ifft2(x_freq * self._h1).real
        r2 = torch.fft.ifft2(x_freq * self._h2).real
        return r1, r2


class RelationPrior(nn.Module):
    """Signed local relation field from multi-scale monogenic vectors.

    This is the inference part of the reference RelationPrior: no support
    losses, no routing ablation switches, and no detail-gain calibration.
    """

    def __init__(self, lambdas=(4.0, 8.0, 16.0, 32.0, 64.0),
                 kappa: float = 0.65, resolution_scale: float = 1.0,
                 eps: float = 1e-12, tau_gate: float = 3.0):
        super().__init__()
        self.lambdas = tuple(float(v) for v in lambdas)
        self.kappa = kappa
        self.resolution_scale = resolution_scale
        self.eps = eps
        self.tau_gate = tau_gate
        self.riesz = RieszFilterBank()
        self.pad = max(1, int(max(self.lambdas) // 2))
        self._filt_cache: Dict = {}

    def _bands(self, h: int, w: int, device: torch.device):
        key = (h, w, str(device))
        if key not in self._filt_cache:
            fy = torch.fft.fftfreq(h, device=device).view(h, 1)
            fx = torch.fft.fftfreq(w, device=device).view(1, w)
            rho = torch.sqrt(fx ** 2 + fy ** 2)
            bands = []
            for lam in self.lambdas:
                lg = torch.exp(
                    -(torch.log(rho.clamp(min=1e-8) * lam)) ** 2
                    / (2.0 * np.log(self.kappa) ** 2)
                )
                lg[0, 0] = 0.0
                bands.append(lg)
            self._filt_cache[key] = bands
        return self._filt_cache[key]

    def _band(self, x: torch.Tensor, lg: torch.Tensor):
        e = torch.fft.ifft2(torch.fft.fft2(x) * lg).real
        r1, r2 = self.riesz(e)
        return e, r1, r2

    @torch.no_grad()
    def forward(self, ir: torch.Tensor, vis: torch.Tensor) -> Dict[str, torch.Tensor]:
        orig_size = ir.shape[-2:]
        if self.resolution_scale != 1.0:
            size = (max(8, int(orig_size[0] * self.resolution_scale)),
                    max(8, int(orig_size[1] * self.resolution_scale)))
            ir_s = F.interpolate(ir.float(), size=size, mode="bilinear",
                                 align_corners=False)
            vis_s = F.interpolate(vis.float(), size=size, mode="bilinear",
                                  align_corners=False)
        else:
            ir_s, vis_s = ir.float(), vis.float()

        hs, ws = ir_s.shape[-2:]
        pad_n = self.pad if (hs > 2 * self.pad + 8 and ws > 2 * self.pad + 8) else 0
        if pad_n > 0:
            ir_s = F.pad(ir_s, (pad_n, pad_n, pad_n, pad_n), mode="reflect")
            vis_s = F.pad(vis_s, (pad_n, pad_n, pad_n, pad_n), mode="reflect")
        hp, wp = ir_s.shape[-2:]
        bands = self._bands(hp, wp, ir_s.device)

        zero = torch.zeros_like(ir_s)
        num_g, den = zero.clone(), zero.clone()
        num_c, num_a, num_w = zero.clone(), zero.clone(), zero.clone()

        for lg in bands:
            e_i, r1_i, r2_i = self._band(ir_s, lg)
            e_v, r1_v, r2_v = self._band(vis_s, lg)

            o_i = torch.sqrt(r1_i ** 2 + r2_i ** 2 + self.eps)
            o_v = torch.sqrt(r1_v ** 2 + r2_v ** 2 + self.eps)
            A_i = torch.sqrt(e_i ** 2 + o_i ** 2 + self.eps)
            A_v = torch.sqrt(e_v ** 2 + o_v ** 2 + self.eps)

            gamma = ((e_i * e_v + r1_i * r1_v + r2_i * r2_v)
                     / (A_i * A_v + self.eps)).clamp(-1.0, 1.0)
            c = gamma.clamp(min=0.0)
            a = (-gamma).clamp(min=0.0)
            w_s = A_i / (A_i + A_v + self.eps)

            w_e = A_i * A_v
            den = den + w_e
            num_g = num_g + w_e * gamma
            num_c = num_c + w_e * c
            num_a = num_a + w_e * a
            num_w = num_w + w_e * w_s

        def _crop(t: torch.Tensor) -> torch.Tensor:
            return t[..., pad_n:hp - pad_n, pad_n:wp - pad_n] if pad_n > 0 else t

        gamma_bar = _crop(num_g / (den + self.eps))
        c_bar = _crop(num_c / (den + self.eps))
        a_bar = _crop(num_a / (den + self.eps))
        w_bar = _crop(num_w / (den + self.eps))
        gate = torch.exp(-self.tau_gate * (1.0 - gamma_bar.abs()))

        def _up(t: torch.Tensor) -> torch.Tensor:
            if self.resolution_scale == 1.0:
                return t
            return F.interpolate(t, size=orig_size, mode="bilinear",
                                 align_corners=False)

        return {
            "Gamma": _up(gamma_bar),
            "c": _up(c_bar),
            "a": _up(a_bar),
            "p": _up(1.0 - gamma_bar.abs()),
            "w": _up(w_bar),
            "G": _up(gate),
        }

    def extra_repr(self) -> str:
        return (f"lambdas={self.lambdas}, kappa={self.kappa}, "
                f"scale={self.resolution_scale}, tau_gate={self.tau_gate}, "
                f"params=0 (fixed)")


# ---------------------------------------------------------------------------
# H6a ordinal decision
# ---------------------------------------------------------------------------

def target_mask(ir: torch.Tensor, percentile: float = 0.9,
                radius: int = 2) -> torch.Tensor:
    """IR target mask: top (1-percentile) IR pixels, box-smoothed."""
    q = torch.quantile(ir.flatten(1), percentile, dim=1).view(-1, 1, 1, 1)
    return box_filter((ir >= q).to(ir.dtype), radius)


def h6_detail(d_ir: torch.Tensor, d_vis: torch.Tensor, d_f: torch.Tensor,
              gamma: torch.Tensor, mask: torch.Tensor,
              rho0: float = 0.6, gamma_tau: float = 0.6) -> torch.Tensor:
    """Ordinal selection/mixing detail operator."""
    ai, av = d_ir.abs(), d_vis.abs()
    strong = torch.where(ai >= av, d_ir, d_vis)
    r = torch.minimum(ai, av) / (torch.maximum(ai, av) + EPS)
    w_amp = ((rho0 - r) / max(rho0, EPS)).clamp(0.0, 1.0)
    w_anti = (-gamma / max(gamma_tau, EPS)).clamp(0.0, 1.0)
    w = 1.0 - (1.0 - w_amp) * (1.0 - w_anti)
    pick = w_anti * (mask * d_ir + (1.0 - mask) * strong) \
        + (1.0 - w_anti) * strong
    return w * pick + (1.0 - w) * d_f


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class PARF(nn.Module):
    """Zero-training PARF operator.

    Args:
        relation_input: "raw" (paper formulation: Gamma from I_i, I_v) or
            "residual" (variant used by the trained-model configuration:
            Gamma from D_i, D_v).
        rho0: amplitude-gate threshold.
        gamma_tau: anti-phase gate threshold.
        target_percentile: IR target percentile for the soft mask.
        mask_radius: box radius of the target mask (2 -> 5x5).
    """

    def __init__(self, gf_radius: int = 8, gf_eps: float = 0.01,
                 saliency_radius: int = 15, lambda_s: float = 0.5,
                 lambda_g: float = 0.2, base_eps: float = 1e-6,
                 detail_eps: float = 1e-6,
                 lambdas=(4.0, 8.0, 16.0, 32.0, 64.0),
                 kappa: float = 0.65, resolution_scale: float = 1.0,
                 tau_gate: float = 3.0, relation_input: str = "raw",
                 rho0: float = 0.6, gamma_tau: float = 0.6,
                 target_percentile: float = 0.9, mask_radius: int = 2):
        super().__init__()
        assert relation_input in ("raw", "residual")
        self.relation_input = relation_input
        self.rho0 = rho0
        self.gamma_tau = gamma_tau
        self.target_percentile = target_percentile
        self.mask_radius = mask_radius
        self.fusion = TraditionalFusion(
            gf_radius=gf_radius, gf_eps=gf_eps,
            saliency_radius=saliency_radius, lambda_s=lambda_s,
            lambda_g=lambda_g, base_eps=base_eps, detail_eps=detail_eps,
        )
        self.relation = RelationPrior(
            lambdas=lambdas, kappa=kappa,
            resolution_scale=resolution_scale, tau_gate=tau_gate,
        )

    @staticmethod
    def _as_tensor(x, name: str) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(np.ascontiguousarray(x)).float()
        elif torch.is_tensor(x):
            x = x.float()
        else:
            raise TypeError(f"{name} must be a numpy array or torch tensor")
        if x.ndim == 2:
            x = x[None, None]
        elif x.ndim == 3:
            if x.shape[0] in (1, 3):
                x = x[None]
            elif x.shape[-1] in (1, 3):
                x = x.permute(2, 0, 1)[None]
            else:
                raise ValueError(f"{name} must be (H,W), (1,H,W) or (H,W,1)")
        if x.ndim != 4:
            raise ValueError(f"{name} must be 2D, 3D or 4D")
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)
        return x.clamp(0.0, 1.0)

    @torch.no_grad()
    def decompose(self, ir, vis) -> Dict[str, torch.Tensor]:
        """Return B_ir/B_vis/D_ir/D_vis/S_ir/B_f/D_f."""
        ir_t = self._as_tensor(ir, "ir")
        vis_t = self._as_tensor(vis, "vis")
        if ir_t.shape[-2:] != vis_t.shape[-2:]:
            raise ValueError("IR and VIS must have the same spatial size")
        _, aux = self.fusion(ir_t, vis_t)
        return aux

    @torch.no_grad()
    def relation_field(self, ir, vis,
                       details: Optional[Dict[str, torch.Tensor]] = None
                       ) -> Dict[str, torch.Tensor]:
        """Return Gamma, c, a, p, w, G."""
        ir_t = self._as_tensor(ir, "ir")
        vis_t = self._as_tensor(vis, "vis")
        if details is None:
            details = self.decompose(ir_t, vis_t)
        if self.relation_input == "residual":
            src_i, src_v = details["D_ir"], details["D_vis"]
        else:
            src_i, src_v = ir_t, vis_t
        return self.relation(src_i, src_v)

    @torch.no_grad()
    def fuse(self, ir, vis, return_aux: bool = False):
        """Fuse a registered pair.

        Returns:
            F: (B,1,H,W) fused image in [0,1].
            aux (optional): B_f, D_ir, D_vis, D_f, Gamma, c, a, w, G, mask, d.
        """
        ir_t = self._as_tensor(ir, "ir")
        vis_t = self._as_tensor(vis, "vis")
        if ir_t.shape[-2:] != vis_t.shape[-2:]:
            raise ValueError("IR and VIS must have the same spatial size")

        f0_classic, fac = self.fusion(ir_t, vis_t)
        rel = self.relation_field(ir_t, vis_t, details=fac)
        m = target_mask(ir_t, self.target_percentile, self.mask_radius)
        d = h6_detail(fac["D_ir"], fac["D_vis"], fac["D_f"],
                      rel["Gamma"], m, self.rho0, self.gamma_tau)
        F_out = torch.clamp(fac["B_f"] + d, 0.0, 1.0)

        if not return_aux:
            return F_out
        aux = {
            "F0_classic": f0_classic,
            "B_f": fac["B_f"],
            "D_ir": fac["D_ir"],
            "D_vis": fac["D_vis"],
            "D_f": fac["D_f"],
            "d": d,
            "mask": m,
            "Gamma": rel["Gamma"],
            "c": rel["c"],
            "a": rel["a"],
            "p": rel["p"],
            "w": rel["w"],
            "G": rel["G"],
        }
        return F_out, aux

    def forward(self, ir, vis):
        return self.fuse(ir, vis)


def fuse(ir, vis, **kwargs):
    """Convenience one-call interface."""
    return PARF(**kwargs).fuse(ir, vis)
