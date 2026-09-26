#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Gizil uzay ve ornekleme teshisi
============================================
Belirti: egitim kaybi cok dusuk (eps_mse 0.061) ama uretilen sinyal gurultu.

Hipotez: gizil uzay IZOTROPIK DEGIL. Duzenlileştirilmemis bir otokodlayici
enerjiyi az sayida boyuta sikistirir. Difuzyon ise N(0,I)'dan baslar, yani
dekoderin hic gormedigi yonlere enerji koyar -> manifold disi -> gurultu.

Teorik kontrol:
    Gizil uzay ~N(0,1) olsaydi, EN IYI ortalama eps MSE = mean(alphabar)
    = 0.2755. Gozlenen 0.0613 bunun 4 KATI ALTINDA -> izotropik degil.

Bu betik 6 test yapar ve Excel + 400 DPI sekil uretir:
  T1  Gizil uzay istatistikleri (kanal basina std, basiklik)
  T2  Etkin boyut (PCA spektrumu, katilim orani)
  T3  Kirilganlik: decode(z + sigma*gurultu) kalitesi
  T4  Prior uyumsuzlugu: gercek z0 vs N(0,I)
  T5  Kismi denoising: gercek z0'dan t=400'e gurultu ekle, geri kazan
  T6  Ornekleme karsilastirmasi: DDPM-1000 vs DDIM-50 vs DDIM-250

Kullanim:
    python diagnose_latent.py --dataset ptbxl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lead_dm.config import (AEConfig, RUNS_DIR, RESULTS_DIR, DATASET_CLASSES,
                            GEN8_LEADS, FS, SIGNAL_SCALE)
from lead_dm.data import make_loader, leads8_to_12, denormalize
from lead_dm.autoencoder import LeadStructuredAE, load_ae_checkpoint
from lead_dm.backbone_m1 import build_m1
from lead_dm.diffusion import GaussianDiffusion, EMA
from lead_dm.config import M1Config, DiffusionConfig
from lead_dm.reporting import (ExcelReport, setup_style, save_figure,
                               plot_ecg_12lead, PALETTE)


# ============================================================================
def load_all(dataset, device, dm_name="M1_leadtcn"):
    dm_ck = torch.load(RUNS_DIR / f"{dataset}__{dm_name}" / "best.pt",
                       map_location=device, weights_only=False)
    ae_dir = dm_ck.get("ae_dir", f"{dataset}__ae")
    ae, ae_cfg, ae_ck = load_ae_checkpoint(
        RUNS_DIR / ae_dir / "best.pt", device)
    n_classes = dm_ck.get("n_classes", len(DATASET_CLASSES[dataset]))
    model = build_m1(n_classes, M1Config()).to(device)
    model.load_state_dict(dm_ck["model"])
    if "ema" in dm_ck:                      # EMA agirliklarini kullan
        ema = EMA(model); ema.load_state_dict(dm_ck["ema"])
        ema.apply_to(model)
    model.eval()
    diff = GaussianDiffusion(DiffusionConfig(), autoencoder=ae).to(device)
    print(f"  DM   : step={dm_ck.get('step')}  val={dm_ck.get('val'):.5f}")
    return ae, model, diff, n_classes


@torch.no_grad()
def collect_latents(ae, loader, device, max_batches=40):
    Z = []
    for i, b in enumerate(loader):
        if i >= max_batches:
            break
        Z.append(ae.encode_scaled(b["x"].to(device)).float().cpu())
    return torch.cat(Z, 0)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl", choices=list(DATASET_CLASSES))
    ap.add_argument("--dm-name", default="M1_leadtcn")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", type=int, default=40)
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    print("=" * 78)
    print("LEAD-DM | Gizil Uzay ve Ornekleme Teshisi")
    print("=" * 78)
    print(f"Veri seti: {args.dataset}")
    ae, model, diff, n_classes = load_all(args.dataset, device, args.dm_name)

    tr = make_loader(args.dataset, "train", 64, shuffle=False, num_workers=2)
    te = make_loader(args.dataset, "test", 64, shuffle=False, num_workers=2)

    figdir = RESULTS_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tag = args.dataset
    plt = setup_style()
    rep = ExcelReport(RESULTS_DIR / f"DIAG_{tag}.xlsx",
                      meta={"Icerik": "Gizil uzay / ornekleme teshisi",
                            "Veri seti": args.dataset,
                            "Model": args.dm_name})

    # ---------------------------------------------------------------- T1
    print("\n[T1] Gizil uzay istatistikleri")
    Z = collect_latents(ae, tr, device, args.batches)          # (N,C,T)
    N, C, T = Z.shape
    ch_std = Z.std(dim=(0, 2)).numpy()                          # (C,)
    flat = Z.permute(1, 0, 2).reshape(C, -1)
    kurt = ((flat - flat.mean(1, keepdim=True)) ** 4).mean(1) / \
           (flat.var(1) ** 2 + 1e-12) - 3.0
    print(f"  z sekli        : {tuple(Z.shape)}   ornek={N}")
    print(f"  genel RMS      : {float(Z.pow(2).mean().sqrt()):.4f}  (1.0 hedef)")
    print(f"  kanal std      : min={ch_std.min():.4f} medyan={np.median(ch_std):.4f} "
          f"max={ch_std.max():.4f}  oran={ch_std.max()/max(ch_std.min(),1e-9):.1f}x")
    print(f"  basiklik (ort) : {float(kurt.mean()):.3f}  (0 = Gauss)")
    dead = int((ch_std < 0.1 * np.median(ch_std)).sum())
    print(f"  'olu' kanal    : {dead}/{C}  (std < medyanin %10'u)")

    # ---------------------------------------------------------------- T2
    print("\n[T2] Etkin boyut")
    X = Z.reshape(N, -1)
    X = X - X.mean(0, keepdim=True)
    n_use = min(N, 512)
    s = torch.linalg.svdvals(X[:n_use].double())
    ev = (s ** 2 / (n_use - 1)).numpy()
    ev = ev / ev.sum()
    pr = 1.0 / (ev ** 2).sum()                    # katilim orani
    c90 = int(np.searchsorted(np.cumsum(ev), 0.90) + 1)
    c99 = int(np.searchsorted(np.cumsum(ev), 0.99) + 1)
    print(f"  toplam boyut   : {C*T}   (ornek sayisi {n_use})")
    print(f"  katilim orani  : {pr:.1f}   [UYARI: ornek sayisiyla sinirli,")
    print(f"                            ust sinir {n_use-1}; mutlak yorum yapmayin]")
    print(f"  varyansin %90'i: {c90} bilesende")
    print(f"  varyansin %99'u: {c99} bilesende")

    # ---------------------------------------------------------------- T3
    print("\n[T3] Gizil uzayin kirilganligi")
    b = next(iter(te))
    x = b["x"][:32].to(device)
    with torch.no_grad():
        z0 = ae.encode_scaled(x)
        base = ae.decode_scaled(z0)
        base_rmse = float((base - x).pow(2).mean().sqrt()) / SIGNAL_SCALE * 1000
    rows_t3 = [{"sigma": 0.0, "RMSE_uV": base_rmse, "Aciklama": "saf rekonstruksiyon"}]
    for sg in [0.05, 0.1, 0.2, 0.5, 1.0]:
        with torch.no_grad():
            xr = ae.decode_scaled(z0 + sg * torch.randn_like(z0))
        r = float((xr - x).pow(2).mean().sqrt()) / SIGNAL_SCALE * 1000
        rows_t3.append({"sigma": sg, "RMSE_uV": r,
                        "Aciklama": f"{r/base_rmse:.1f}x taban"})
        print(f"  sigma={sg:<4}: RMSE={r:8.1f} uV   ({r/base_rmse:5.1f}x taban)")
    df3 = pd.DataFrame(rows_t3)

    # ---------------------------------------------------------------- T4
    print("\n[T4] Prior uyumsuzlugu (gercek z0 vs N(0,I))")
    zr = torch.randn(256, C, T)
    with torch.no_grad():
        x_rand = ae.decode_scaled(zr.to(device))
    real_amp = float(denormalize(x).abs().max())
    rand_amp = float(denormalize(x_rand).abs().max())
    # gercek z0'in N(0,I) altindaki log-olabilirligi vs rastgele
    ll_real = float(-(Z[:256] ** 2).mean())
    ll_rand = float(-(zr ** 2).mean())
    print(f"  gercek z0 : E[z^2]={-ll_real:.4f}   kanal std orani={ch_std.max()/ch_std.min():.1f}x")
    print(f"  N(0,I)    : E[z^2]={-ll_rand:.4f}   kanal std orani=1.0x")
    print(f"  N(0,I) dekoded genlik : {rand_amp:.3f} mV  (gercek {real_amp:.3f} mV)")
    df4 = pd.DataFrame([
        {"Kaynak": "gercek z0", "E[z^2]": -ll_real,
         "kanal_std_orani": float(ch_std.max() / ch_std.min()),
         "dekode_max_mV": real_amp},
        {"Kaynak": "N(0,I)", "E[z^2]": -ll_rand, "kanal_std_orani": 1.0,
         "dekode_max_mV": rand_amp}])

    # ---------------------------------------------------------------- T5
    print("\n[T5] Kismi denoising (ters surec calisiyor mu?)")
    cond = {"disease": b["disease"][:4].to(device),
            "age_bits": b["age_bits"][:4].to(device),
            "sex": b["sex"][:4].to(device)}
    z0_4 = z0[:4]
    rows_t5 = []
    for t_start in [200, 400, 600, 800, 999]:
        tt = torch.full((4,), t_start, device=device, dtype=torch.long)
        with torch.no_grad():
            zt, _ = diff.q_sample(z0_4, tt)
            z = zt.clone()
            for i in range(t_start, -1, -1):
                ti = torch.full((4,), i, device=device, dtype=torch.long)
                eps = model(z, ti, cond["disease"], cond["age_bits"], cond["sex"])
                zz0 = diff.predict_z0(z, ti, eps).clamp(-6, 6)
                mean = (diff._extract(diff.posterior_mean_c0, ti, z.shape) * zz0
                        + diff._extract(diff.posterior_mean_ct, ti, z.shape) * z)
                if i > 0:
                    lv = diff._extract(diff.posterior_logvar, ti, z.shape)
                    z = mean + (0.5 * lv).exp() * torch.randn_like(z)
                else:
                    z = mean
            xr = ae.decode_scaled(z)
        zerr = float((z - z0_4).pow(2).mean().sqrt())
        xerr = float((xr - x[:4]).pow(2).mean().sqrt()) / SIGNAL_SCALE * 1000
        rows_t5.append({"t_start": t_start, "z_RMSE": zerr, "x_RMSE_uV": xerr})
        print(f"  t={t_start:<4}: z_RMSE={zerr:.4f}  x_RMSE={xerr:8.1f} uV")
    df5 = pd.DataFrame(rows_t5)
    print("  -> t=999 satiri kotu ama kucuk t iyi ise: ters surec SAGLAM,")
    print("     sorun yalnizca t=T'deki prior uyumsuzlugudur.")

    # ---------------------------------------------------------------- T6
    print("\n[T6] Ornekleme yontemi karsilastirmasi")
    rows_t6 = []
    samples = {}
    for name, kw in [("DDIM-50", dict(ddim_steps=50)),
                     ("DDIM-250", dict(ddim_steps=250)),
                     ("DDPM-1000", dict(ddim_steps=None))]:
        zs = diff.sample(model, cond, (4, C, T), device, progress=False, **kw)
        with torch.no_grad():
            xs = ae.decode_scaled(zs)
        samples[name] = xs
        rows_t6.append({
            "Yontem": name,
            "z_std": float(zs.std()),
            "z_kanal_std_orani": float(zs.std(dim=(0, 2)).max() /
                                       zs.std(dim=(0, 2)).min()),
            "x_std_mV": float(denormalize(xs).std()),
            "x_max_mV": float(denormalize(xs).abs().max()),
        })
        print(f"  {name:<10}: z_std={rows_t6[-1]['z_std']:.3f}  "
              f"x_std={rows_t6[-1]['x_std_mV']:.4f} mV  "
              f"x_max={rows_t6[-1]['x_max_mV']:.3f} mV")
    real_std = float(denormalize(x).std())
    print(f"  {'GERCEK':<10}: z_std={float(z0.std()):.3f}  x_std={real_std:.4f} mV  "
          f"x_max={real_amp:.3f} mV")
    rows_t6.append({"Yontem": "GERCEK", "z_std": float(z0.std()),
                    "z_kanal_std_orani": float(ch_std.max() / ch_std.min()),
                    "x_std_mV": real_std, "x_max_mV": real_amp})
    df6 = pd.DataFrame(rows_t6)

    # ---------------------------------------------------------------- Excel
    df1 = pd.DataFrame({"Kanal": np.arange(C), "std": ch_std,
                        "basiklik": kurt.numpy(),
                        "Lead": np.repeat(list(GEN8_LEADS), C // 8)})
    df2 = pd.DataFrame([{"Metrik": "Toplam boyut", "Deger": C * T},
                        {"Metrik": "Katilim orani", "Deger": pr},
                        {"Metrik": "Varyans %90 icin bilesen", "Deger": c90},
                        {"Metrik": "Varyans %99 icin bilesen", "Deger": c99},
                        {"Metrik": "Etkin/toplam orani", "Deger": pr / (C * T)},
                        {"Metrik": "Gozlenen eps_mse", "Deger": 0.0613},
                        {"Metrik": "Teorik alt sinir (izotropik)", "Deger": 0.2755},
                        {"Metrik": "Esdeger izotropik sigma^2", "Deger": 0.0221}])
    rep.add_table("T1_gizil_istatistik", df1,
                  caption="T1 - Kanal basina gizil std ve basiklik.")
    rep.add_table("T2_etkin_boyut", df2,
                  caption="T2 - Etkin boyut analizi. Katilim orani toplam boyuttan cok kucukse gizil uzay anizotropiktir.")
    rep.add_table("T3_kirilganlik", df3,
                  caption="T3 - Gizil uzaya gurultu eklendiginde rekonstruksiyon bozulmasi.")
    rep.add_table("T4_prior_uyumsuzlugu", df4,
                  caption="T4 - Gercek gizil dagilim ile N(0,I) karsilastirmasi.")
    rep.add_table("T5_kismi_denoising", df5,
                  caption="T5 - Farkli t'den baslatilan geri kazanim. Ters surecin saglamligini olcer.")
    rep.add_table("T6_ornekleme", df6,
                  caption="T6 - Ornekleme yontemleri ve gercek veriyle karsilastirma.")
    print(f"\n[Excel] {rep.save()}")

    # ---------------------------------------------------------------- Sekiller
    fig, ax = plt.subplots(1, 3, figsize=(9.6, 2.7))
    ax[0].hist(ch_std, bins=50, color=PALETTE[0])
    ax[0].axvline(1.0, color=PALETTE[1], ls="--", label="izotropik hedef")
    ax[0].set_xlabel("Kanal std"); ax[0].set_ylabel("Adet")
    ax[0].legend(); ax[0].set_title("(a) Kanal std dagilimi", loc="left")

    ax[1].semilogy(np.cumsum(ev), color=PALETTE[0])
    ax[1].axhline(0.90, color=PALETTE[1], ls="--", lw=.8)
    ax[1].axhline(0.99, color=PALETTE[2], ls="--", lw=.8)
    ax[1].set_xlabel("Bilesen"); ax[1].set_ylabel("Kumulatif varyans")
    ax[1].set_xlim(0, min(400, len(ev))); ax[1].grid(True)
    ax[1].set_title("(b) PCA spektrumu", loc="left")

    ax[2].plot(df3["sigma"], df3["RMSE_uV"], marker="o", color=PALETTE[0])
    ax[2].set_xlabel("Eklenen gurultu $\\sigma$")
    ax[2].set_ylabel("RMSE ($\\mu$V)"); ax[2].set_yscale("log"); ax[2].grid(True)
    ax[2].set_title("(c) Gizil kirilganlik", loc="left")
    fig.tight_layout()
    save_figure(fig, figdir / f"diag_{tag}_latent")

    for name, xs in samples.items():
        x12 = leads8_to_12(denormalize(xs))[0].cpu().numpy()
        plot_ecg_12lead(x12, figdir / f"diag_{tag}_sample_{name.replace('-','')}",
                        fs=FS, title=f"Ornekleme: {name}")
    x12r = leads8_to_12(denormalize(x))[0].cpu().numpy()
    plot_ecg_12lead(x12r, figdir / f"diag_{tag}_real", fs=FS,
                    title="Gercek EKG (referans)")
    print(f"[Sekil] {figdir}")

    # ---------------------------------------------------------------- Karar
    print("\n" + "=" * 78)
    print("TESHIS")
    print("=" * 78)
    aniso = float(ch_std.max() / max(ch_std.min(), 1e-9))
    samp_std = float(df6.loc[df6.Yontem == "DDPM-1000", "z_std"].iloc[0])
    real_std = float(df6.loc[df6.Yontem == "GERCEK", "z_std"].iloc[0])
    infl = samp_std / max(real_std, 1e-9)

    print(f"  1) Gizil izotropi      : kanal std orani = {aniso:.2f}x")
    print("     " + ("[OK] saglikli (<1.5x)" if aniso < 1.5 else
                     "[!!] anizotropik -> latent_norm=per_channel / --refit"))
    print(f"  2) Ornekleme varyansi  : sampled/gercek = {infl:.2f}x")
    print("     " + ("[OK] saglikli (0.7-1.5x)" if 0.7 < infl < 1.5 else
                     "[!!] sisme/cokme -> parameterization='v' kullanildi mi?"))
    print(f"  3) Ters surec          : t=200'den x_RMSE = "
          f"{df5['x_RMSE_uV'].iloc[0]:.1f} uV  (AE tabani {base_rmse:.1f} uV)")
    print("     " + ("[OK] saglikli" if df5["x_RMSE_uV"].iloc[0] < 5 * base_rmse
                     else "[!!] geri kazanim zayif"))
    if base_rmse > 40:
        print("\n  [!!] AE TABAN HATASI COK YUKSEK ({:.1f} uV).".format(base_rmse))
        print("       Enkoder-dekoder yeterince egitilmemis olabilir;")
        print("       difuzyon sonuclari bu tavanla sinirlanir.")


if __name__ == "__main__":
    main()
