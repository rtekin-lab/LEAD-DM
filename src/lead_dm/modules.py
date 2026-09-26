#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Yapi taslari
=========================
Bu dosya tum omurgalar (M1..M5) tarafindan paylasilir.

Ana bilesenler
--------------
ConditionEncoder : hastalik / yas / cinsiyet / difuzyon adimi -> gommeler
LeadSelector     : hastalik gommesi -> derivasyon ilgi agirliklari  w (B,L)
LTCM             : Lead- and Time-selective Condition Modulation   [ONERIMIZ]
MCFARN           : referans makalenin modulu (ablasyon icin)
AdaLNZero        : DiT'in standart modulu (taban)
CrossAttnMod     : capraz dikkat ile kosullama (ablasyon icin)
LeadTCNBlock     : derivasyon-yerel + karisim yollu TCN blogu
LeadHead         : asimetrik derinlikli derivasyon basligi
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import COND_DIM, AGE_BITS, N_GEN_LEADS, CHEST_POS, LIMB_POS


# ============================================================================
# TEMEL YARDIMCILAR
# ============================================================================

def zero_module(m: nn.Module) -> nn.Module:
    """Modulun tum parametrelerini sifirlar (adaLN-Zero stratejisi)."""
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class GroupNorm1d(nn.GroupNorm):
    """Difuzyon icin BatchNorm YERINE. Batch istatistiklerine bagimli degil."""
    def __init__(self, num_channels: int, num_groups: int = 8):
        super().__init__(num_groups=min(num_groups, num_channels),
                         num_channels=num_channels, eps=1e-6, affine=True)


def timestep_embedding(t: torch.Tensor, dim: int = COND_DIM,
                       max_period: float = 1e4) -> torch.Tensor:
    """
    Difuzyon adimi t icin sinus/kosinus gommesi.
    t: (B,) tamsayi -> (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(half, dtype=torch.float32,
                                             device=t.device) / half
    )
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=1)
    return emb


# ============================================================================
# KOSUL KODLAYICI
# ============================================================================

class ConditionEncoder(nn.Module):
    """
    Girdi:
        disease  (B, C)   cok-sicak
        age_bits (B, 7)   Gray kodu
        sex      (B,)     0/1/2
        t        (B,)     difuzyon adimi
        *_null   (B,)     1.0 -> o kosul maskeli (CFG icin)

    Cikti:
        e_d, e_a, e_g, t_emb : her biri (B, COND_DIM)
        c_glob               : (B, COND_DIM)  global ozet (t + yas + cinsiyet)
    """

    def __init__(self, n_classes: int, dim: int = COND_DIM):
        super().__init__()
        self.dim = dim

        # Hastalik: cok etiketli oldugu icin Linear (Embedding degil)
        self.disease_proj = nn.Sequential(
            nn.Linear(n_classes, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.age_proj = nn.Sequential(
            nn.Linear(AGE_BITS, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.sex_emb = nn.Embedding(3, dim)          # 0=E 1=K 2=bilinmiyor

        # Ogrenilebilir "yok" gommeleri  (maskeli kosul egitimi)
        self.null_d = nn.Parameter(torch.randn(dim) * 0.02)
        self.null_a = nn.Parameter(torch.randn(dim) * 0.02)
        self.null_g = nn.Parameter(torch.randn(dim) * 0.02)

        # Difuzyon adimi
        self.t_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim), nn.SiLU())

        # Global ozet: difuzyon adimi + yas + cinsiyet
        self.glob_mlp = nn.Sequential(
            nn.Linear(3 * dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, disease, age_bits, sex, t,
                d_null=None, a_null=None, g_null=None):
        B = disease.shape[0]
        z = lambda v: (torch.zeros(B, device=disease.device)
                       if v is None else v.float())
        d_null, a_null, g_null = z(d_null), z(a_null), z(g_null)

        e_d = self.disease_proj(disease)
        e_a = self.age_proj(age_bits)
        e_g = self.sex_emb(sex)

        m = lambda e, null, nullvec: (
            torch.where(null.view(-1, 1) > 0.5,
                        nullvec.unsqueeze(0).expand_as(e), e))
        e_d = m(e_d, d_null, self.null_d)
        e_a = m(e_a, a_null, self.null_a)
        e_g = m(e_g, g_null, self.null_g)

        t_emb = self.t_mlp(timestep_embedding(t, self.dim))
        c_glob = self.glob_mlp(torch.cat([t_emb, e_a, e_g], dim=1))
        return {"e_d": e_d, "e_a": e_a, "e_g": e_g,
                "t_emb": t_emb, "c_glob": c_glob}


# ============================================================================
# DAL B — DERIVASYON SECICILIGI  (tum bloklarda paylasilir)
# ============================================================================

class LeadSelector(nn.Module):
    """
    Ogrenilebilir derivasyon sorgulari, hastalik gommesine capraz bakar.

        s = Q . MLP(e_d)^T / sqrt(d)        (B, L)
        w = L * softmax(s)                  ortalamasi 1'e normalize

    Yorum: "bu hastalik hangi derivasyonlarda gorunur"
    Kaynak: PCLGRNet / LeadGraphCrossAttention fikri.

    Q sifirla baslatilir -> egitim basinda w = 1 (tum derivasyonlar esit).
    """

    def __init__(self, n_leads: int = N_GEN_LEADS, dim: int = COND_DIM):
        super().__init__()
        self.n_leads = n_leads
        self.dim = dim
        self.Q = nn.Parameter(torch.zeros(n_leads, dim))
        self.key = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(),
                                 nn.Linear(dim, dim))
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, e_d: torch.Tensor) -> torch.Tensor:
        k = self.key(e_d)                                # (B, d)
        s = torch.einsum("ld,bd->bl", self.Q, k) * self.scale
        w = self.n_leads * torch.softmax(s, dim=-1)      # (B, L), ort. = 1
        return w


# ============================================================================
# MODULASYON MODULLERI
# ============================================================================

class AdaLNZero(nn.Module):
    """
    Taban: tum kosullar birlestirilip tek bir (gamma, beta, alpha) uretilir.
    Cikti sekli: her biri (B, C, 1) -> zaman ve derivasyon boyunca sabit.
    """

    def __init__(self, channels: int, dim: int = COND_DIM):
        super().__init__()
        self.net = nn.Sequential(nn.SiLU(), nn.Linear(4 * dim, 3 * channels))
        zero_module(self.net[1])
        self.channels = channels

    def forward(self, z, ctx, w=None):
        c = torch.cat([ctx["t_emb"], ctx["e_d"], ctx["e_a"], ctx["e_g"]], dim=1)
        g, b, a = self.net(c).chunk(3, dim=1)
        s = (-1, self.channels, 1)
        return g.view(*s), b.view(*s), a.view(*s)


class MCFARN(nn.Module):
    """
    Referans makalenin modulu (ablasyon karsilastirmasi icin).

        [g_k, b_k, a_k] = Chunk3(MLP_k(e_k)),  k in {d, a, g}
        g_c = g_d * g_a * g_g     (eleman bazli carpim)

    Not: makale gamma/beta/alpha'yi 0'a ilklendiriyor; carpim durumunda
    bu sifir tuzagi yaratir. Bu yuzden (1+g) formunda kullaniyoruz ve
    MLP ciktilarini sifirliyoruz -> baslangicta g_c = 1 (etkisiz).
    """

    def __init__(self, channels: int, dim: int = COND_DIM):
        super().__init__()
        self.channels = channels
        self.mlp_d = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * channels))
        self.mlp_a = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * channels))
        self.mlp_g = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * channels))
        self.mlp_t = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3 * channels))
        for m in (self.mlp_d, self.mlp_a, self.mlp_g, self.mlp_t):
            zero_module(m[1])

    def forward(self, z, ctx, w=None):
        C = self.channels
        gd, bd, ad = self.mlp_d(ctx["e_d"]).chunk(3, 1)
        ga, ba, aa = self.mlp_a(ctx["e_a"]).chunk(3, 1)
        gg, bg, ag = self.mlp_g(ctx["e_g"]).chunk(3, 1)
        gt, bt, at = self.mlp_t(ctx["t_emb"]).chunk(3, 1)
        # (1+g) formunda carpim: baslangicta hepsi 1 -> etkisiz
        g = (1 + gd) * (1 + ga) * (1 + gg) * (1 + gt) - 1.0
        b = bd + ba + bg + bt
        a = (1 + ad) * (1 + aa) * (1 + ag) * (1 + at) - 1.0
        s = (-1, C, 1)
        return g.view(*s), b.view(*s), a.view(*s)


class LTCM(nn.Module):
    """
    Lead- and Time-selective Condition Modulation   [BU CALISMANIN ONERISI]

    Dal A (global) : t_emb + yas + cinsiyet -> (gamma_g, beta_g, alpha_g)
                     -> tum derivasyon ve zamana yayilir
    Dal B (lead)   : hastalik -> w (B, L)     [LeadSelector, disaridan gelir]
    Dal C (zaman)  : hastalik + yerel morfoloji -> g(t) (B, L, T)

    Birlesim:
        m     = w[:,:,None] * g                          (B, L, T)
        gamma = gamma_g + lam * m * gamma_d
        beta  = beta_g  + lam * m * beta_d

    lam ogrenilebilir ve 0'dan baslar -> egitimin basinda LTCM == adaLN-Zero.
    Bu, "en kotu ihtimalle kotulesmez" garantisi verir.

    lam_init = -1.0 verilirse dal B ve C tamamen kapatilir (ablasyon A8).
    """

    def __init__(self, channels: int, n_leads: int = N_GEN_LEADS,
                 dim: int = COND_DIM, gate_kernel: int = 9,
                 lam_init: float = 0.0):
        super().__init__()
        assert channels % n_leads == 0, "kanal sayisi lead sayisina bolunmeli"
        self.channels = channels
        self.n_leads = n_leads
        self.cpl = channels // n_leads          # lead basina kanal
        self.disabled_bc = (lam_init < 0)

        # --- Dal A: global ---
        # DIKKAT: c_glob yalnizca (t_emb, yas, cinsiyet) icerir; hastalik
        # normalde dal B/C uzerinden girer. Dal B/C kapatildiginda (A8
        # ablasyonu) hastalik gommesi dal A'ya EKLENMELIDIR, aksi halde
        # ablasyon "lead/zaman seciciligini kaldirmak" degil "hastalik
        # kosullamasini tamamen kaldirmak" olur (ilk surumdeki hata).
        glob_in = dim * (2 if self.disabled_bc else 1)
        self.glob = nn.Sequential(nn.SiLU(), nn.Linear(glob_in, 3 * channels))
        zero_module(self.glob[1])

        if self.disabled_bc:
            self.lam = None
            return

        # --- Dal B/C icin hastaliga ozel olcek/kaydirma ---
        self.dis = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * channels))
        zero_module(self.dis[1])

        # --- Dal C: zaman kapisi ---
        pad = gate_kernel // 2
        self.gate_dw = nn.Conv1d(channels, channels, gate_kernel,
                                 padding=pad, groups=channels, bias=False)
        self.gate_cond = nn.Linear(dim, channels)
        self.gate_out = nn.Conv1d(channels, n_leads, 1, groups=n_leads)
        nn.init.zeros_(self.gate_dw.weight)
        nn.init.zeros_(self.gate_cond.weight); nn.init.zeros_(self.gate_cond.bias)
        nn.init.zeros_(self.gate_out.weight);  nn.init.zeros_(self.gate_out.bias)

        self.lam = nn.Parameter(torch.tensor(float(lam_init)))

    def forward(self, z: torch.Tensor, ctx: dict, w: torch.Tensor | None = None):
        """
        z: (B, C, T)   C = n_leads * cpl
        w: (B, L)      LeadSelector ciktisi
        Doner: gamma, beta, alpha -> yayinlanabilir sekilde
        """
        B, C, T = z.shape
        gin = (torch.cat([ctx["c_glob"], ctx["e_d"]], dim=1)
               if self.disabled_bc else ctx["c_glob"])
        gg, bg, ag = self.glob(gin).chunk(3, dim=1)
        gamma = gg.view(B, C, 1)
        beta = bg.view(B, C, 1)
        alpha = ag.view(B, C, 1)

        if self.disabled_bc:
            return gamma, beta, alpha

        # --- Dal C: zamana bagli kapi ---
        h = self.gate_dw(z)                                   # (B,C,T)
        h = h + self.gate_cond(ctx["e_d"]).unsqueeze(-1)      # kosul enjeksiyonu
        g_t = torch.sigmoid(self.gate_out(F.silu(h)))         # (B,L,T)

        # --- Dal B: derivasyon agirliklari ---
        if w is None:
            w = torch.ones(B, self.n_leads, device=z.device, dtype=z.dtype)
        m = w.unsqueeze(-1) * g_t                             # (B,L,T)

        # (B,L,T) -> (B,C,T)  her lead kendi cpl kanalina yayilir
        m = m.unsqueeze(2).expand(B, self.n_leads, self.cpl, T)
        m = m.reshape(B, C, T)

        gd, bd = self.dis(ctx["e_d"]).chunk(2, dim=1)
        lam = self.lam
        gamma = gamma + lam * m * gd.view(B, C, 1)
        beta = beta + lam * m * bd.view(B, C, 1)
        return gamma, beta, alpha

    def lead_attention(self, ctx: dict, w: torch.Tensor) -> torch.Tensor:
        """Gorsellestime icin: derivasyon ilgi agirliklarini dondurur."""
        return w.detach()


class CrossAttnMod(nn.Module):
    """
    Ablasyon A7: kosul gommeleri anahtar/deger, gizil temsil sorgu.
    adaLN yerine dogrudan capraz dikkat ile kosullama.
    """

    def __init__(self, channels: int, dim: int = COND_DIM, heads: int = 4):
        super().__init__()
        self.channels = channels
        self.norm = GroupNorm1d(channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.kv = nn.Linear(dim, 2 * channels)
        self.proj = zero_module(nn.Conv1d(channels, channels, 1))
        self.heads = heads
        # Global alpha (rezidüel olcegi) yine adaLN tarzi
        self.a_mlp = nn.Sequential(nn.SiLU(), nn.Linear(4 * dim, channels))
        zero_module(self.a_mlp[1])

    def forward(self, z, ctx, w=None):
        B, C, T = z.shape
        conds = torch.stack([ctx["t_emb"], ctx["e_d"],
                             ctx["e_a"], ctx["e_g"]], dim=1)   # (B,4,dim)
        k, v = self.kv(conds).chunk(2, dim=-1)                  # (B,4,C)
        q = self.q(self.norm(z))                                # (B,C,T)
        H = self.heads
        q = q.view(B, H, C // H, T).permute(0, 1, 3, 2)         # (B,H,T,C/H)
        k = k.view(B, 4, H, C // H).permute(0, 2, 1, 3)         # (B,H,4,C/H)
        v = v.view(B, 4, H, C // H).permute(0, 2, 1, 3)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(C // H), -1)
        o = (att @ v).permute(0, 1, 3, 2).reshape(B, C, T)
        delta = self.proj(o)
        a = self.a_mlp(torch.cat([ctx["t_emb"], ctx["e_d"],
                                  ctx["e_a"], ctx["e_g"]], 1)).view(B, C, 1)
        # gamma/beta yerine dogrudan toplama; arayuzu ayni tutmak icin
        return torch.zeros_like(a), delta, a


def build_modulation(kind: str, channels: int, n_leads: int,
                     lam_init: float = 0.0) -> nn.Module:
    kind = kind.lower()
    if kind == "ltcm":
        return LTCM(channels, n_leads, lam_init=lam_init)
    if kind == "mcfarn":
        return MCFARN(channels)
    if kind == "adaln":
        return AdaLNZero(channels)
    if kind == "crossattn":
        return CrossAttnMod(channels)
    raise ValueError(f"bilinmeyen modulasyon: {kind}")


# ============================================================================
# TCN BLOKLARI
# ============================================================================

class LeadTCNBlock(nn.Module):
    """
    M1'in temel blogu. Iki yol:

      [1] Derivasyon-YEREL : groups=n_leads  -> her lead kendi morfolojisi
      [2] Derivasyon-KARISIM: groups=1       -> leadler arasi bilgi akisi

    Her yol kendi modulasyonunu alir (kosul enjeksiyonu iki kez).
    """

    def __init__(self, channels: int, n_leads: int, kernel: int = 7,
                 dilation: int = 1, dropout: float = 0.0,
                 modulation: str = "ltcm", lam_init: float = 0.0,
                 use_mix_path: bool = True):
        super().__init__()
        self.use_mix_path = use_mix_path
        pad = (kernel - 1) // 2 * dilation

        # --- Yol 1: derivasyon-yerel ---
        self.norm1 = GroupNorm1d(channels, n_leads)
        self.mod1 = build_modulation(modulation, channels, n_leads, lam_init)
        self.conv1 = nn.Conv1d(channels, channels, kernel, padding=pad,
                               dilation=dilation, groups=n_leads)
        self.conv1b = nn.Conv1d(channels, channels, 1, groups=n_leads)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # --- Yol 2: derivasyonlar arasi karisim ---
        if use_mix_path:
            self.norm2 = GroupNorm1d(channels, n_leads)
            self.mod2 = build_modulation(modulation, channels, n_leads, lam_init)
            self.conv2 = nn.Conv1d(channels, channels, 1, groups=1)
            self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Cikis conv'lari sifirdan baslar -> blok basta kimlik fonksiyonu
        zero_module(self.conv1b)
        if use_mix_path:
            zero_module(self.conv2)

    @staticmethod
    def _apply_mod(h, gamma, beta):
        return (1.0 + gamma) * h + beta

    def forward(self, z, ctx, w=None):
        # Yol 1
        g, b, a = self.mod1(z, ctx, w)
        h = self._apply_mod(self.norm1(z), g, b)
        h = self.conv1b(self.drop1(F.silu(self.conv1(h))))
        z = z + (1.0 + a) * h if a.shape[1] == z.shape[1] else z + h

        # Yol 2
        if self.use_mix_path:
            g, b, a = self.mod2(z, ctx, w)
            h = self._apply_mod(self.norm2(z), g, b)
            h = self.drop2(F.silu(self.conv2(h)))
            z = z + (1.0 + a) * h if a.shape[1] == z.shape[1] else z + h
        return z


class TCNResBlock(nn.Module):
    """Basliklarda kullanilan sade rezidüel TCN blogu (kosulsuz)."""

    def __init__(self, ch: int, kernel: int = 7, dilation: int = 1):
        super().__init__()
        pad = (kernel - 1) // 2 * dilation
        self.norm = GroupNorm1d(ch, min(8, ch))
        self.conv = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.proj = zero_module(nn.Conv1d(ch, ch, 1))

    def forward(self, x):
        return x + self.proj(F.silu(self.conv(self.norm(x))))


class LeadHead(nn.Module):
    """
    Tek bir derivasyonun cikis basligi.

    depth = 0 -> sadece 1x1 conv       (ablasyon A3)
    depth >= 1 -> depth adet TCNResBlock + cikis conv
    Gogus derivasyonlari icin depth=3, uzuv icin depth=1 (asimetrik tasarim).
    """

    def __init__(self, ch: int, depth: int = 3, kernel: int = 7,
                 dilations: Sequence[int] = (1, 2, 4)):
        super().__init__()
        blocks = []
        for i in range(depth):
            blocks.append(TCNResBlock(ch, kernel, dilations[i % len(dilations)]))
        self.blocks = nn.Sequential(*blocks) if blocks else nn.Identity()
        self.out = nn.Conv1d(ch, ch, 1)
        nn.init.zeros_(self.out.bias)

    def forward(self, x):
        return self.out(self.blocks(x))


class AsymmetricLeadHeads(nn.Module):
    """
    8 derivasyon icin ayri basliklar.
    Gogus (V1..V6) derin, uzuv (I, aVF) sig  -> fizyolojik gerekce:
    uzuv derivasyonlari dogrusal bagimli, morfolojik zenginlikleri dusuk.
    """

    def __init__(self, channels: int, n_leads: int = N_GEN_LEADS,
                 asymmetric: bool = True, chest_depth: int = 3,
                 limb_depth: int = 1, uniform_depth: int = 2):
        super().__init__()
        assert channels % n_leads == 0
        self.cpl = channels // n_leads
        self.n_leads = n_leads
        depths = []
        for i in range(n_leads):
            if not asymmetric:
                depths.append(uniform_depth)
            else:
                depths.append(chest_depth if i in CHEST_POS else limb_depth)
        self.depths = depths
        self.heads = nn.ModuleList([LeadHead(self.cpl, d) for d in depths])

    def forward(self, z):
        B, C, T = z.shape
        zl = z.view(B, self.n_leads, self.cpl, T)
        outs = [h(zl[:, i]) for i, h in enumerate(self.heads)]
        return torch.stack(outs, dim=1).view(B, C, T)
