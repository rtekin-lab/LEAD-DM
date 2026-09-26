#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Derivasyon-yapili enkoder-dekoder  (KL-duzenlileştirilmiş)
=======================================================================

Referans makaleden iki yapisal ayrilik:

 1) DERIVASYON-YAPILI GIZIL UZAY
      CDM-DL-PSI :  (8, 1000) -> (256, 125)     derivasyon kimligi KAYBOLUR
      LEAD-DM    :  (8, 1000) -> (8, 32, 125)   her derivasyon kanallarini korur
    Ayni sikistirma orani (32.000 sayi), farkli yapi. LTCM'in "hangi hastalik
    hangi derivasyonda gorunur" bilgisini kullanabilmesi icin zorunlu.

 2) KL DUZENLILESTIRME  (VAE)
    Duzenlileştirilmemis bir otokodlayici, rekonstruksiyonu iyilestirmek icin
    enerjiyi az sayida boyuta sikistirir. Olculen sonuc: gizil uzayin
    esdeger izotropik varyansi ~0.022 (olmasi gereken 1.0). Difuzyon
    N(0,I)'dan basladigi icin dekoderin hic gormedigi yonlere enerji koyar
    -> manifold disi -> uretilen sinyal gurultu olur.
    Belirti: eps_mse teorik alt sinirin (mean(alphabar)=0.2755) cok altinda.

    Cozum: kucuk agirlikli KL terimi (Stable Diffusion ~1e-6) + kanal basina
    normalizasyon. KL gizil dagilimi N(0,I)'ya yaklastirir; kanal basina
    normalizasyon kalan olcek farkini kapatir.

Gizil uzay duzeni
-----------------
    z[:, c, :] kanali, lead = c // ch_per_lead derivasyonuna aittir.
    z.view(B, 8, 32, 125) dogru gruplamayi verir.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AEConfig, N_GEN_LEADS, SIG_LEN
from .modules import GroupNorm1d


# ============================================================================
# BLOKLAR
# ============================================================================

class GConvBlock(nn.Module):
    """Gruplu Conv -> GroupNorm -> SiLU, rezidüel."""

    def __init__(self, in_ch, out_ch, groups, kernel=7, stride=1):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                               padding=pad, groups=groups)
        self.norm1 = GroupNorm1d(out_ch, groups)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel, padding=pad, groups=groups)
        self.norm2 = GroupNorm1d(out_ch, groups)
        self.skip = (nn.Identity() if (in_ch == out_ch and stride == 1)
                     else nn.Conv1d(in_ch, out_ch, 1, stride=stride, groups=groups))

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class GUpBlock(nn.Module):
    """Lineer upsample -> gruplu conv."""

    def __init__(self, in_ch, out_ch, groups, kernel=7):
        super().__init__()
        self.block = GConvBlock(in_ch, out_ch, groups, kernel, stride=1)

    def forward(self, x, target_len=None):
        if target_len is None:
            target_len = x.shape[-1] * 2
        x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)
        return self.block(x)


# ============================================================================
# ANA MODEL
# ============================================================================

class LeadStructuredAE(nn.Module):
    """
    Enkoder : (B, 8, 1000) -> z (B, 256, 125)     [VAE ise mu, logvar]
    Dekoder : z (B, 256, 125) -> (B, 8, 1000)
    """

    def __init__(self, cfg: AEConfig | None = None):
        super().__init__()
        cfg = cfg or AEConfig()
        self.cfg = cfg
        L = cfg.n_leads
        g = L if cfg.lead_independent else 1
        self.groups = g

        c1 = L * cfg.stem_ch_per_lead      # 128
        c2 = L * cfg.mid_ch_per_lead       # 192
        c3 = L * cfg.ch_per_lead           # 256
        self.cpl = cfg.ch_per_lead
        self.n_leads = L

        out_mult = 2 if cfg.variational else 1
        self.out_mult = out_mult

        # ---------------- Enkoder ----------------
        self.enc_stem = nn.Sequential(
            nn.Conv1d(L, c1, cfg.kernel, padding=cfg.kernel // 2, groups=g),
            GroupNorm1d(c1, g), nn.SiLU())
        self.enc1 = GConvBlock(c1, c1, g, cfg.kernel, stride=2)   # 1000 -> 500
        self.enc2 = GConvBlock(c1, c2, g, cfg.kernel, stride=2)   #  500 -> 250
        self.enc3 = GConvBlock(c2, c3, g, cfg.kernel, stride=2)   #  250 -> 125
        self.enc_out = nn.Sequential(
            GroupNorm1d(c3, g), nn.SiLU(),
            nn.Conv1d(c3, c3 * out_mult, 1, groups=g))

        # ---------------- Dekoder ----------------
        self.dec_in = nn.Sequential(
            nn.Conv1d(c3, c3, 1, groups=g), GroupNorm1d(c3, g), nn.SiLU())
        self.dec3 = GUpBlock(c3, c2, g, cfg.kernel)               #  125 -> 250
        self.dec2 = GUpBlock(c2, c1, g, cfg.kernel)               #  250 -> 500
        self.dec1 = GUpBlock(c1, c1, g, cfg.kernel)               #  500 -> 1000
        self.dec_out = nn.Sequential(
            GroupNorm1d(c1, g), nn.SiLU(),
            nn.Conv1d(c1, L, cfg.kernel, padding=cfg.kernel // 2, groups=g))

        # ---------------- Gizil normalizasyon ----------------
        self.register_buffer("latent_scale", torch.tensor(1.0))
        self.register_buffer("latent_mean_c", torch.zeros(c3, 1))
        self.register_buffer("latent_std_c", torch.ones(c3, 1))

    # ------------------------------------------------------------------
    def _split_mu_logvar(self, h):
        """
        Gruplu duzende chunk(2, dim=1) YANLIS olur (leadleri boler).
        Dogru yol: (B, L, 2*cpl, T) olarak yeniden sekillendirip dim=2'de bol.
        """
        B, _, T = h.shape
        h = h.view(B, self.n_leads, 2 * self.cpl, T)
        mu = h[:, :, :self.cpl]
        logvar = h[:, :, self.cpl:]
        C = self.n_leads * self.cpl
        return (mu.reshape(B, C, T),
                logvar.reshape(B, C, T).clamp(-30.0, 20.0))

    def encode_dist(self, x):
        """Ham enkoder cikisi. Doner: (mu, logvar) veya (z, None)."""
        h = self.enc_stem(x)
        h = self.enc1(h)
        h = self.enc2(h)
        h = self.enc3(h)
        h = self.enc_out(h)
        if self.cfg.variational:
            return self._split_mu_logvar(h)
        return h, None

    def encode(self, x, sample: bool | None = None):
        """
        Ham (normalize edilmemis) gizil temsil.
        sample=None -> egitimde ornekle, degerlendirmede mu kullan.
        """
        mu, logvar = self.encode_dist(x)
        if logvar is None:
            return mu
        if sample is None:
            sample = self.training
        if not sample:
            return mu
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        h = self.dec_in(z)
        h = self.dec3(h, target_len=z.shape[-1] * 2)
        h = self.dec2(h, target_len=z.shape[-1] * 4)
        h = self.dec1(h, target_len=z.shape[-1] * 8)
        return self.dec_out(h)

    def forward(self, x):
        mu, logvar = self.encode_dist(x)
        if logvar is None:
            z = mu
        elif self.training:
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)
        else:
            z = mu
        return self.decode(z), z, mu, logvar

    # ------------------------------------------------------------------
    # Difuzyonun gordugu normalize gizil uzay
    # ------------------------------------------------------------------
    def normalize_latent(self, z):
        if self.cfg.latent_norm == "per_channel":
            return (z - self.latent_mean_c) / self.latent_std_c
        return z * self.latent_scale

    def denormalize_latent(self, z):
        if self.cfg.latent_norm == "per_channel":
            return z * self.latent_std_c + self.latent_mean_c
        return z / self.latent_scale

    def encode_scaled(self, x, sample: bool | None = None):
        return self.normalize_latent(self.encode(x, sample=sample))

    def decode_scaled(self, z):
        return self.decode(self.denormalize_latent(z))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def fit_latent_stats(self, loader, device, max_batches: int = 200):
        """
        Egitim setinden gizil istatistikleri olcup buffer'lara yazar.
        Hem skaler hem kanal basina degerler hesaplanir; hangisinin
        kullanilacagini cfg.latent_norm belirler.
        """
        was_training = self.training
        self.eval()
        s1 = s2 = None
        n_elem = 0
        sq_sum = 0.0
        cnt = 0
        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            z = self.encode(batch["x"].to(device), sample=False).float()
            B, C, T = z.shape
            if s1 is None:
                s1 = torch.zeros(C, device=z.device, dtype=torch.float64)
                s2 = torch.zeros(C, device=z.device, dtype=torch.float64)
            s1 += z.double().sum(dim=(0, 2))
            s2 += z.double().pow(2).sum(dim=(0, 2))
            n_elem += B * T
            sq_sum += float(z.double().pow(2).sum())
            cnt += B * C * T

        mean = (s1 / n_elem).float()
        var = (s2 / n_elem).float() - mean ** 2
        raw_std = var.clamp_min(0.0).sqrt()
        # 'Olu' kanallari asiri buyutmemek icin taban
        floor = max(0.05 * float(raw_std.median()), 1e-4)
        std = raw_std.clamp_min(floor)

        self.latent_mean_c.copy_(mean.view(-1, 1))
        self.latent_std_c.copy_(std.view(-1, 1))
        rms = (sq_sum / max(cnt, 1)) ** 0.5
        self.latent_scale.fill_(1.0 / max(rms, 1e-6))

        # Difuzyonun GERCEKTE gordugu (normalize) temsilin istatistikleri
        if self.cfg.latent_norm == "per_channel":
            eff_std = raw_std / std                 # tanim geregi ~1.0
        else:
            eff_std = raw_std * float(self.latent_scale)
        eff_ratio = float(eff_std.max() / eff_std.clamp_min(1e-12).min())

        if was_training:
            self.train()
        return {
            "scalar_scale": float(self.latent_scale),
            "latent_rms": rms,
            "ch_std_min": float(raw_std.min()),
            "ch_std_med": float(raw_std.median()),
            "ch_std_max": float(raw_std.max()),
            "ch_std_ratio_raw": float(raw_std.max() /
                                      raw_std.clamp_min(1e-12).min()),
            # ASIL ONEMLI OLAN: normalizasyon SONRASI oran.
            # per_channel modunda tanim geregi 1.0'dir; difuzyon bunu gorur.
            "ch_std_ratio_effective": eff_ratio,
            "latent_norm": self.cfg.latent_norm,
            "n_dead": int((raw_std < floor).sum()),
            "n_channels": int(raw_std.numel()),
        }

    # Geriye donuk uyum
    @torch.no_grad()
    def fit_latent_scale(self, loader, device, max_batches: int = 200):
        return self.fit_latent_stats(loader, device, max_batches)["scalar_scale"]

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ============================================================================
# KAYIP
# ============================================================================

class AELoss(nn.Module):
    """
    L = MSE + w_l1*L1 + w_spec*spektral + kl_weight*KL

    MSE cekirdegi referans makale Denklem 11 ile ayni.
    KL agirligi cok kucuktur (~1e-6): amac rekonstruksiyonu bozmak degil,
    gizil uzayin difuzyona uygun (izotropik, surekli) olmasini saglamaktir.
    """

    def __init__(self, w_l1: float = 0.1, w_spec: float = 0.05,
                 kl_weight: float = 1e-6):
        super().__init__()
        self.w_l1 = w_l1
        self.w_spec = w_spec
        self.kl_weight = kl_weight

    def forward(self, x_hat, x, mu=None, logvar=None):
        mse = F.mse_loss(x_hat, x)
        out = {"mse": mse.detach()}
        loss = mse
        if self.w_l1 > 0:
            l1 = F.l1_loss(x_hat, x)
            loss = loss + self.w_l1 * l1
            out["l1"] = l1.detach()
        if self.w_spec > 0:
            X = torch.fft.rfft(x.float(), dim=-1, norm="ortho").abs()
            Xh = torch.fft.rfft(x_hat.float(), dim=-1, norm="ortho").abs()
            spec = F.mse_loss(Xh, X)
            loss = loss + self.w_spec * spec
            out["spec"] = spec.detach()
        if self.kl_weight > 0 and logvar is not None:
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).mean()
            loss = loss + self.kl_weight * kl
            out["kl"] = kl.detach()
        out["total"] = loss.detach()
        return loss, out


# ============================================================================
# CHECKPOINT YUKLEME  (geriye donuk uyumlu)
# ============================================================================

def load_ae_checkpoint(path, device=None, verbose: bool = True):
    """
    best.pt dosyasindan enkoder-dekoderi yukler.

    ESKI checkpoint'ler (VAE eklenmeden once kaydedilenler) cfg sozlugunde
    'variational' / 'kl_weight' / 'latent_norm' alanlarini icermez. Bu
    durumda dataclass varsayilanlari devreye girer (variational=True) ve
    enc_out kanal sayisi uyusmaz. Asagidaki mantik eski dosyalari duz
    otokodlayici olarak dogru sekilde yukler.

    Doner: (ae, cfg, ckpt)
    """
    import dataclasses
    import torch as _t

    device = device or _t.device("cpu")
    ck = _t.load(path, map_location=device, weights_only=False)
    raw = dict(ck.get("cfg", {}))

    valid = {f.name for f in dataclasses.fields(AEConfig)}
    dropped = [k for k in raw if k not in valid]
    for k in dropped:
        raw.pop(k)

    legacy = "variational" not in ck.get("cfg", {})
    if legacy:
        raw["variational"] = False
        raw["kl_weight"] = 0.0
        raw["latent_norm"] = "scalar"

    cfg = AEConfig(**raw)
    ae = LeadStructuredAE(cfg).to(device)
    sd = ck["model"]
    missing, unexpected = ae.load_state_dict(sd, strict=False)

    # Eski dosyalarda latent_mean_c / latent_std_c yok; scalar norm
    # kullanildigi icin varsayilan (0, 1) degerleri zararsizdir.
    ignorable = {"latent_mean_c", "latent_std_c"}
    real_missing = [k for k in missing if k not in ignorable]
    if real_missing:
        raise RuntimeError(f"Checkpoint eksik anahtar iceriyor: {real_missing}")
    ae.eval()

    fitted = ck.get("latent_stats_fitted", None)
    if verbose:
        kind = "ESKI (duz otokodlayici)" if legacy else \
               ("VAE" if cfg.variational else "duz otokodlayici")
        print(f"  AE tipi       : {kind}   gizil norm={cfg.latent_norm}")
        print(f"  AE egitim     : epoch {ck.get('epoch','?')}  "
              f"val_mse {ck.get('val_mse', float('nan')):.5f}")
        if fitted is False:
            print("  " + "!" * 62)
            print("  [!!] BU CHECKPOINT YARIDA KESILMIS BIR EGITIMDEN GELIYOR.")
            print("       Gizil normalizasyon istatistikleri hesaplanmamis.")
            print("       AE'yi yeniden egitin veya --refit calistirin.")
            print("  " + "!" * 62)
        if legacy:
            print("  [!] Bu checkpoint KL duzenlileştirmesi ONCESINDE egitildi.")
            print("      Gizil uzay anizotropik olabilir; teshis bunu olcecek.")
        if dropped:
            print(f"  [i] Taninmayan cfg alanlari atlandi: {dropped}")
    return ae, cfg, ck


if __name__ == "__main__":
    for var in (False, True):
        cfg = AEConfig(variational=var)
        ae = LeadStructuredAE(cfg)
        x = torch.randn(2, N_GEN_LEADS, SIG_LEN)
        xh, z, mu, lv = ae(x)
        print(f"variational={var}: gizil={tuple(z.shape)} "
              f"cikti={tuple(xh.shape)} param={ae.n_params()/1e6:.3f} M "
              f"logvar={'var' if lv is not None else 'yok'}")
