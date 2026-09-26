#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Difuzyon sureci
============================

TESHIS SONUCU YAPILAN DEGISIKLIKLER
------------------------------------
Belirti: egitim kaybi cok dusuk ama uretilen ornekler gurultu; ornekleme
sonunda z_std = 2.4-2.9 olcaldu (gercek z0'in std'si 1.0).

Kok neden: eps-parametrizasyonunda z0 tahmini

    z0_hat = (1/sqrt(ab)) z_t - sqrt(1/ab - 1) eps_hat

t=999'da 1/sqrt(ab) = 157. Yani z0_hat'in std'sinin 1 olmasi icin eps
hatasinin std'si 0.0064'un ALTINDA olmali. Modelin eps tahminindeki
kucucuk bir sapma bile z0_hat'i devasa yapar; clamp(-6,6) devreye girer
ve yoruge boyunca fazla enerji birikir -> varyans sismesi.

COZUM 1 - v-parametrizasyonu (Salimans & Ho):
    v := sqrt(ab) * eps - sqrt(1-ab) * z0
    z0_hat = sqrt(ab) z_t - sqrt(1-ab) v_hat
Katsayilar her t icin <= 1. Amplifikasyon YAPISAL OLARAK YOK.

COZUM 2 - Sifir terminal SNR (Lin ve ark.):
    Lineer zamanlamada alphabar_T = 4e-5, yani z_T'de hala sinyal sizintisi
    var. Egitim bunu ogrenirken ornekleme saf gurultuden basliyor ->
    egitim/ornekleme uyumsuzlugu. Beta zamanlamasi alphabar_T = 0 olacak
    sekilde yeniden olceklenir. (v-parametrizasyonu ile birlikte kullanilmali;
    eps-parametrizasyonu alphabar_T = 0'da tanimsiz olur.)

COZUM 3 - Kosinus zamanlamasi: dusuk/orta t bolgesine daha fazla kapasite.

Kayip
-----
    L = ||hedef - model||^2                          (hedef: v veya eps)
      + lambda_f * spektral sekil kaybi
      + lambda_i * InterLeadConsistency(x_hat)
      + lambda_p * PDE_residual(x_hat)
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import DiffusionConfig, N_GEN_LEADS
from .data import leads8_to_12


# ============================================================================
# ZAMANLAMA
# ============================================================================

def make_betas(cfg: DiffusionConfig) -> torch.Tensor:
    if cfg.schedule == "linear":
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, cfg.n_steps,
                               dtype=torch.float64)
    elif cfg.schedule == "cosine":
        s, T = 0.008, cfg.n_steps
        f = lambda u: math.cos((u + s) / (1 + s) * math.pi / 2) ** 2
        ab = torch.tensor([f(i / T) / f(0) for i in range(T + 1)],
                          dtype=torch.float64)
        betas = (1 - ab[1:] / ab[:-1]).clamp(1e-8, 0.999)
    else:
        raise ValueError(cfg.schedule)

    if getattr(cfg, "zero_terminal_snr", False):
        betas = enforce_zero_terminal_snr(betas)
    return betas


def enforce_zero_terminal_snr(betas: torch.Tensor) -> torch.Tensor:
    """
    Lin ve ark. (2024): sqrt(alphabar) dizisini, son degeri tam 0 olacak
    sekilde afin olarak yeniden olcekler. Boylece z_T saf gurultudur ve
    egitim ile ornekleme birbirine uyar.
    """
    alphas = 1.0 - betas
    ab = torch.cumprod(alphas, dim=0)
    s = ab.sqrt()
    s0, sT = s[0].clone(), s[-1].clone()
    s = (s - sT) * (s0 / (s0 - sT))          # s[0] korunur, s[-1] -> 0
    ab_new = s ** 2
    alphas_new = torch.cat([ab_new[:1], ab_new[1:] / ab_new[:-1]])
    return (1.0 - alphas_new).clamp(0.0, 0.999)


# ============================================================================
# EK KAYIP TERIMLERI
# ============================================================================

class InterLeadConsistency(nn.Module):
    """Einthoven + Goldberger kisiti (turetilen 4 uzuv derivasyonu)."""

    def forward(self, x8_hat: torch.Tensor, x8_true: torch.Tensor):
        p12 = leads8_to_12(x8_hat)
        t12 = leads8_to_12(x8_true)
        limb = [1, 2, 3, 4]                      # II, III, aVR, aVL
        return F.mse_loss(p12[:, limb], t12[:, limb])


class ChestSmoothness(nn.Module):
    """V1 -> V6 morfolojisi kademeli degisir; ani ziplamalari cezalandirir."""

    def forward(self, x8_hat: torch.Tensor):
        v = x8_hat[:, 0:6]
        d1 = v[:, 1:] - v[:, :-1]
        d2 = d1[:, 1:] - d1[:, :-1]
        return d2.pow(2).mean()


class LeadPDEResidual(nn.Module):
    """
    Ogrenilebilir tepki-difuzyon dinamigi (calisma1_libs.LeadPDENet'ten):
        du/dt = f(u; theta) + D * sum_j w_ij (u_j - u_i)
    Egitim sonrasi W matrisi gorsellestirilebilir.
    """

    def __init__(self, n_leads: int = 12, hidden: int = 64, fs: float = 100.0):
        super().__init__()
        self.n_leads = n_leads
        self.dt = 1.0 / fs
        self.f_net = nn.Sequential(
            nn.Linear(n_leads, hidden), nn.Tanh(), nn.Linear(hidden, n_leads))
        self.log_D = nn.Parameter(torch.tensor(-2.0))
        self.W = nn.Parameter(torch.randn(n_leads, n_leads) * 0.01)
        self.register_buffer("eye", torch.eye(n_leads))

    def dynamics(self, u):
        W = self.W * (1.0 - self.eye)
        diff = u.unsqueeze(1) - u.unsqueeze(2)
        spatial = (W.unsqueeze(0) * (-diff)).sum(dim=2)
        return self.f_net(u) + torch.exp(self.log_D) * spatial

    def forward(self, x8_hat: torch.Tensor):
        u = leads8_to_12(x8_hat).transpose(1, 2)
        B, T, L = u.shape
        lhs = (u[:, 1:] - u[:, :-1]) / self.dt
        rhs = self.dynamics(u[:, :-1].reshape(-1, L)).view(B, T - 1, L)
        return (lhs - rhs).pow(2).mean()


# ============================================================================
# ANA DIFUZYON SINIFI
# ============================================================================

class GaussianDiffusion(nn.Module):

    def __init__(self, cfg: DiffusionConfig | None = None,
                 autoencoder: nn.Module | None = None):
        super().__init__()
        cfg = cfg or DiffusionConfig()
        self.cfg = cfg
        self.param = getattr(cfg, "parameterization", "v")
        self.ae = autoencoder

        betas = make_betas(cfg)
        alphas = 1.0 - betas
        ab = torch.cumprod(alphas, dim=0)
        ab_prev = torch.cat([torch.ones(1, dtype=torch.float64), ab[:-1]])

        reg = lambda n, v: self.register_buffer(n, v.float())
        reg("betas", betas)
        reg("alphas", alphas)
        reg("alphas_bar", ab)
        reg("alphas_bar_prev", ab_prev)
        reg("sqrt_ab", ab.clamp_min(0).sqrt())
        reg("sqrt_1m_ab", (1.0 - ab).clamp_min(0).sqrt())

        ab_safe = ab.clamp_min(1e-12)          # sifir terminal SNR icin guvenli
        reg("sqrt_recip_ab", (1.0 / ab_safe).sqrt())
        reg("sqrt_recipm1_ab", (1.0 / ab_safe - 1).clamp_min(0).sqrt())

        post_var = betas * (1.0 - ab_prev) / (1.0 - ab).clamp_min(1e-12)
        reg("posterior_var", post_var)
        reg("posterior_logvar", post_var.clamp(min=1e-20).log())
        reg("posterior_mean_c0",
            betas * ab_prev.sqrt() / (1.0 - ab).clamp_min(1e-12))
        reg("posterior_mean_ct",
            (1.0 - ab_prev) * alphas.sqrt() / (1.0 - ab).clamp_min(1e-12))

        self.interlead = InterLeadConsistency()
        self.chest_smooth = ChestSmoothness()
        self.pde = LeadPDEResidual() if cfg.lambda_pde > 0 else None

    # ------------------------------------------------------------------
    @staticmethod
    def _extract(arr, t, shape):
        return arr.gather(0, t).view(-1, *([1] * (len(shape) - 1)))

    def q_sample(self, z0, t, noise=None):
        noise = torch.randn_like(z0) if noise is None else noise
        zt = (self._extract(self.sqrt_ab, t, z0.shape) * z0
              + self._extract(self.sqrt_1m_ab, t, z0.shape) * noise)
        return zt, noise

    # --- parametrizasyon donusumleri ---------------------------------
    def get_v(self, z0, noise, t):
        return (self._extract(self.sqrt_ab, t, z0.shape) * noise
                - self._extract(self.sqrt_1m_ab, t, z0.shape) * z0)

    def z0_from_v(self, z_t, t, v):
        return (self._extract(self.sqrt_ab, t, z_t.shape) * z_t
                - self._extract(self.sqrt_1m_ab, t, z_t.shape) * v)

    def eps_from_v(self, z_t, t, v):
        return (self._extract(self.sqrt_ab, t, z_t.shape) * v
                + self._extract(self.sqrt_1m_ab, t, z_t.shape) * z_t)

    def z0_from_eps(self, z_t, t, eps):
        return (self._extract(self.sqrt_recip_ab, t, z_t.shape) * z_t
                - self._extract(self.sqrt_recipm1_ab, t, z_t.shape) * eps)

    # geriye donuk ad
    def predict_z0(self, z_t, t, eps):
        return self.z0_from_eps(z_t, t, eps)

    def decode_output(self, z_t, t, out):
        """Model ciktisindan (v veya eps) hem z0 hem eps uretir."""
        if self.param == "v":
            return self.z0_from_v(z_t, t, out), self.eps_from_v(z_t, t, out)
        return self.z0_from_eps(z_t, t, out), out

    def training_target(self, z0, noise, t):
        return self.get_v(z0, noise, t) if self.param == "v" else noise

    # ------------------------------------------------------------------
    def loss(self, model, z0, batch, x8_true: torch.Tensor | None = None):
        B = z0.shape[0]
        t = torch.randint(0, self.cfg.n_steps, (B,), device=z0.device)
        z_t, noise = self.q_sample(z0, t)
        target = self.training_target(z0, noise, t)

        out = model(z_t, t, batch["disease"], batch["age_bits"], batch["sex"],
                    batch.get("d_null"), batch.get("a_null"), batch.get("g_null"))

        parts = {}
        loss = F.mse_loss(out, target)
        parts["main_mse" if self.param == "v" else "eps_mse"] = loss.detach()

        z0_hat, eps_hat = self.decode_output(z_t, t, out)
        # izleme icin: eps-uzayindaki hata (parametrizasyondan bagimsiz)
        with torch.no_grad():
            parts["eps_mse_eq"] = F.mse_loss(eps_hat, noise).detach()
            parts["z0_mse"] = F.mse_loss(z0_hat, z0).detach()

        # --- spektral sekil kaybi ---
        # Genlik farki kullanilir (karmasik fark Parseval geregi ana kayibin
        # yeniden agirliklandirilmis hali olurdu, yeni bilgi tasimazdi).
        if self.cfg.lambda_freq > 0:
            ab_t = self.alphas_bar.gather(0, t).view(-1, 1, 1)
            w_snr = (ab_t / (1.0 - ab_t).clamp_min(1e-8)).clamp(
                max=self.cfg.freq_snr_clip)
            Z = torch.fft.rfft(z0.float(), dim=-1, norm="ortho").abs()
            Zh = torch.fft.rfft(z0_hat.float(), dim=-1, norm="ortho").abs()
            nf = Z.shape[-1]
            f = torch.linspace(0.0, 1.0, nf, device=Z.device)
            hf_w = (1.0 + self.cfg.freq_hf_gain * f).view(1, 1, nf)
            fl = (w_snr * hf_w * (Z - Zh) ** 2).mean()
            loss = loss + self.cfg.lambda_freq * fl
            parts["freq"] = fl.detach()

        # --- x-uzayi terimleri ---
        need_x = (self.cfg.lambda_interlead > 0 or self.cfg.lambda_pde > 0)
        if need_x and self.ae is not None and x8_true is not None:
            x8_hat = self.ae.decode_scaled(z0_hat)
            if self.cfg.lambda_interlead > 0:
                il = self.interlead(x8_hat, x8_true)
                loss = loss + self.cfg.lambda_interlead * il
                parts["interlead"] = il.detach()
            if self.cfg.lambda_pde > 0 and self.pde is not None:
                pd = self.pde(x8_hat)
                loss = loss + self.cfg.lambda_pde * pd
                parts["pde"] = pd.detach()

        parts["total"] = loss.detach()
        return loss, parts

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _guided_out(self, model, z_t, t, cond, guidance: dict | None,
                    null_mask: dict | None = None):
        """
        Kosul basina ayri yonlendirme. v- ve eps-uzayinda ayni sekilde
        gecerlidir (her ikisi de model ciktisinin lineer kombinasyonu).

        null_mask : {"d": (B,), "a": (B,), "g": (B,)} — SENARYO maskeleri.
            Tablo 11 senaryolarinda (only_dl / only_psi) bazi kosullar
            KALICI olarak dusurulur. Bu maskeler yonlendirme dallarinin
            uzerine 'maximum' ile bindirilir: senaryo bir kosulu attiysa
            hicbir dalda geri gelmez.
        """
        B = z_t.shape[0]
        dev = z_t.device
        one = lambda: torch.ones(B, device=dev)
        zero = lambda: torch.zeros(B, device=dev)
        nm = null_mask or {}
        z0_ = torch.zeros(B, device=dev)
        d0 = nm.get("d", z0_); a0 = nm.get("a", z0_); g0 = nm.get("g", z0_)

        def call(dn, an, gn):
            return model(z_t, t, cond["disease"], cond["age_bits"], cond["sex"],
                         torch.maximum(dn, d0), torch.maximum(an, a0),
                         torch.maximum(gn, g0))

        if not guidance:
            return call(zero(), zero(), zero())

        # --- MOD 1: BUTUNSEL yonlendirme (standart CFG) ---------------
        #     eps = eps_NULL + w * (eps_TAM - eps_NULL)
        # w = 1.0 tam olarak kosullu modele denktir; w > 1 tum kosullari
        # (hastalik + yas + cinsiyet) birlikte keskinlestirir.
        # Yalnizca 2 model cagrisi gerektirir.
        w_all = float(guidance.get("all", 0.0))
        if w_all:
            o_null = call(one(), one(), one())
            o_full = call(zero(), zero(), zero())
            return o_null + w_all * (o_full - o_null)

        # --- MOD 2: KOSUL BASINA yonlendirme (bilesimsel) -------------
        # DIKKAT: burada her terim "yalnizca o kosul acik" tahminidir.
        # Yani {"disease": 1.5} demek "hastaligi guclendir" DEGIL,
        # "yas ve cinsiyeti tamamen at, hastaligi 1.5x abart" demektir.
        # Yas/cinsiyeti korumak icin onlarin agirliklarini da verin.
        w_d = float(guidance.get("disease", 0.0))
        w_a = float(guidance.get("age", 0.0))
        w_g = float(guidance.get("sex", 0.0))
        if w_d == 0 and w_a == 0 and w_g == 0:
            return call(zero(), zero(), zero())

        o_null = call(one(), one(), one())
        o = o_null
        if w_d != 0:
            o = o + w_d * (call(zero(), one(), one()) - o_null)
        if w_a != 0:
            o = o + w_a * (call(one(), zero(), one()) - o_null)
        if w_g != 0:
            o = o + w_g * (call(one(), one(), zero()) - o_null)
        return o

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, model, cond: dict, shape, device,
               guidance: dict | None = None,
               ddim_steps: int | None = None, eta: float = 0.0,
               z0_clip: float = 4.0,
               progress: bool = True,
               return_traj: bool = False,
               null_mask: dict | None = None):
        """
        cond  : {"disease","age_bits","sex"}
        shape : (B, C, T)
        ddim_steps=None -> tam DDPM
        return_traj=True -> her adimda z ve z0_hat std'lerini de dondurur
                            (teshis icin: varyans sismesini gorunur kilar)
        null_mask -> {"d","a","g"} senaryo maskeleri (Tablo 11)
        """
        model.eval()
        z = torch.randn(shape, device=device)

        if ddim_steps is None:
            steps = list(range(self.cfg.n_steps - 1, -1, -1))
        else:
            s = np.linspace(self.cfg.n_steps - 1, 0, ddim_steps)
            steps = sorted({int(round(v)) for v in s}, reverse=True)

        it = steps
        if progress:
            try:
                from tqdm import tqdm
                it = tqdm(steps, desc="  ornekleme", ncols=78, leave=False)
            except ImportError:
                pass

        traj = []
        for idx, i in enumerate(it):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            out = self._guided_out(model, z, t, cond, guidance, null_mask)
            z0 = self.decode_output(z, t, out)[0]
            if z0_clip is not None and z0_clip > 0:
                z0 = z0.clamp(-z0_clip, z0_clip)

            if return_traj:
                traj.append({"t": i, "z_std": float(z.std()),
                             "z0_std": float(z0.std())})

            if ddim_steps is None:
                mean = (self._extract(self.posterior_mean_c0, t, z.shape) * z0
                        + self._extract(self.posterior_mean_ct, t, z.shape) * z)
                if i > 0:
                    logvar = self._extract(self.posterior_logvar, t, z.shape)
                    z = mean + (0.5 * logvar).exp() * torch.randn_like(z)
                else:
                    z = mean
            else:
                i_prev = steps[idx + 1] if idx + 1 < len(steps) else -1
                ab_t = self.alphas_bar[i]
                ab_p = (self.alphas_bar[i_prev] if i_prev >= 0
                        else torch.tensor(1.0, device=device))
                eps = self.decode_output(z, t, out)[1]
                sigma = eta * (((1 - ab_p) / (1 - ab_t).clamp_min(1e-12))
                               * (1 - ab_t / ab_p.clamp_min(1e-12))
                               ).clamp_min(0).sqrt()
                dir_zt = (1 - ab_p - sigma ** 2).clamp_min(0).sqrt() * eps
                z = ab_p.clamp_min(0).sqrt() * z0 + dir_zt
                if i_prev >= 0 and eta > 0:
                    z = z + sigma * torch.randn_like(z)

        if return_traj:
            return z, traj
        return z


# ============================================================================
# EMA
# ============================================================================

class EMA:
    """
    Ustel hareketli ortalama.

    ISINMA: decay=0.9999 ile 2000 adim sonra golge agirliklarin %82'si
    HALA ilklendirme degeridir. Isinma bunu duzeltir:
        d_t = min(decay, (1 + t) / (10 + t))
    """

    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 warmup: bool = True):
        self.decay = decay
        self.warmup = warmup
        self.step = 0
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.backup = {}

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        d = self.current_decay()
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach(), alpha=1 - d)

    def apply_to(self, model: nn.Module):
        self.backup = {k: v.detach().clone()
                       for k, v in model.state_dict().items()
                       if k in self.shadow}
        model.load_state_dict(self.shadow, strict=False)

    def restore(self, model: nn.Module):
        if self.backup:
            model.load_state_dict(self.backup, strict=False)
            self.backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow,
                "step": self.step, "warmup": self.warmup}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]
        self.step = sd.get("step", 0)
        self.warmup = sd.get("warmup", True)
