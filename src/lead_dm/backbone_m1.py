#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM/T  (M1)  |  Derivasyon-yerel TCN gurultu tahmincisi
============================================================

Akis
----
    z_t (B, 256, 125) + kosullar
      |
      +-- giris projeksiyonu (gruplu 1x1)
      |
      +-- 12 x LeadTCNBlock
      |     [1] derivasyon-YEREL yol  (groups=8)  + LTCM
      |     [2] derivasyon-KARISIM yolu (groups=1) + LTCM
      |
      +-- 8 x LeadHead  (gogus derinlik 3, uzuv derinlik 1)
      |
      +-- cikis projeksiyonu -> eps_hat (B, 256, 125)

Ozgun tasarim kararlari
-----------------------
1. Yerel/karisim ayrimi : once her lead kendi morfolojisini isler, sonra
   leadler arasi bilgi akar. Standart difuzyon omurgalarinda bu ayrim yok.
2. Asimetrik baslik derinligi : uzuv derivasyonlari dogrusal bagimli
   (denetimde rekonstruksiyon hatasi ~0.002 mV), gogus derivasyonlari
   bagimsiz ve morfolojik olarak zengin.
3. LTCM : kosul modulasyonu (derivasyon x zaman) cozunurlugunde.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import M1Config, COND_DIM, N_GEN_LEADS
from .modules import (ConditionEncoder, LeadSelector, LeadTCNBlock,
                      AsymmetricLeadHeads, GroupNorm1d, zero_module)


class LeadDM_T(nn.Module):
    """M1 gurultu tahmincisi. eps-parametrizasyonu."""

    def __init__(self, n_classes: int, cfg: M1Config | None = None):
        super().__init__()
        cfg = cfg or M1Config()
        self.cfg = cfg
        C = cfg.channels                # 8 * 32 = 256
        L = cfg.n_leads
        self.channels, self.n_leads = C, L

        # ---- Kosullar ----
        self.cond = ConditionEncoder(n_classes, COND_DIM)
        self.use_lead_selector = (cfg.modulation == "ltcm"
                                  and cfg.ltcm_lambda_init >= 0)
        if self.use_lead_selector:
            self.lead_sel = LeadSelector(L, COND_DIM)

        # ---- Giris ----
        self.in_proj = nn.Sequential(
            nn.Conv1d(C, C, 1, groups=L), GroupNorm1d(C, L), nn.SiLU())

        # ---- Govde ----
        dil = list(cfg.dilations)
        if len(dil) < cfg.n_blocks:
            dil = (dil * ((cfg.n_blocks // len(dil)) + 1))[:cfg.n_blocks]
        self.blocks = nn.ModuleList([
            LeadTCNBlock(C, L, kernel=cfg.kernel, dilation=dil[i],
                         dropout=cfg.dropout, modulation=cfg.modulation,
                         lam_init=cfg.ltcm_lambda_init,
                         use_mix_path=cfg.use_mix_path)
            for i in range(cfg.n_blocks)
        ])

        # ---- Basliklar ----
        self.heads = AsymmetricLeadHeads(
            C, L,
            asymmetric=cfg.asymmetric_heads,
            chest_depth=cfg.chest_head_depth,
            limb_depth=cfg.limb_head_depth,
            uniform_depth=cfg.uniform_head_depth)

        # ---- Cikis ----
        self.out_norm = GroupNorm1d(C, L)
        self.out_proj = zero_module(nn.Conv1d(C, C, 1, groups=L))

        self._last_lead_w = None       # gorsellestime icin saklanir

    # ------------------------------------------------------------------
    def forward(self, z_t, t, disease, age_bits, sex,
                d_null=None, a_null=None, g_null=None):
        """
        z_t   : (B, C, T)  gurultulu gizil temsil
        t     : (B,)       difuzyon adimi
        Doner : (B, C, T)  tahmin edilen gurultu
        """
        ctx = self.cond(disease, age_bits, sex, t, d_null, a_null, g_null)
        w = self.lead_sel(ctx["e_d"]) if self.use_lead_selector else None
        self._last_lead_w = w

        h = self.in_proj(z_t)
        for blk in self.blocks:
            h = blk(h, ctx, w)
        h = self.heads(h)
        return self.out_proj(F.silu(self.out_norm(h)))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def lead_attention_map(self, disease, age_bits, sex, device=None):
        """
        Makaledeki 'derivasyon ilgi haritasi' sekli icin.
        Doner: (B, 8) — her hastalik kosulunun hangi derivasyonlari
        one cikardigini gosterir. Ortalama 1.0'dir; >1 = vurgulanan.
        """
        if not self.use_lead_selector:
            return None
        t = torch.zeros(disease.shape[0], dtype=torch.long,
                        device=disease.device)
        ctx = self.cond(disease, age_bits, sex, t)
        return self.lead_sel(ctx["e_d"])

    def param_summary(self) -> dict:
        s = lambda m: sum(p.numel() for p in m.parameters())
        d = {
            "condition": s(self.cond),
            "lead_selector": s(self.lead_sel) if self.use_lead_selector else 0,
            "in_proj": s(self.in_proj),
            "blocks": s(self.blocks),
            "heads": s(self.heads),
            "out": s(self.out_norm) + s(self.out_proj),
        }
        d["TOTAL"] = sum(d.values())
        return d


def build_m1(n_classes: int, cfg: M1Config | None = None) -> LeadDM_T:
    return LeadDM_T(n_classes, cfg)


if __name__ == "__main__":
    from .config import M1Config
    m = build_m1(n_classes=9)
    B, C, T = 2, m.channels, 125
    z = torch.randn(B, C, T)
    t = torch.randint(0, 1000, (B,))
    dis = torch.zeros(B, 9); dis[:, 1] = 1
    age = torch.randint(0, 2, (B, 7)).float()
    sex = torch.randint(0, 2, (B,))
    out = m(z, t, dis, age, sex)
    print(f"girdi : {tuple(z.shape)}")
    print(f"cikti : {tuple(out.shape)}")
    print("parametreler:")
    for k, v in m.param_summary().items():
        print(f"  {k:<15}: {v:>12,}")
    print("lead ilgi haritasi:", m.lead_attention_map(dis, age, sex))
