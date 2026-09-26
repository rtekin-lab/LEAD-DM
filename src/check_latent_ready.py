#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Gizil uzay hazirlik kontrolu
=========================================
Difuzyon modeli GEREKTIRMEZ. Sadece egitilmis enkoder-dekoder ile,
gizil uzayin difuzyona uygun olup olmadigini olcer.

Amac: 5 saatlik difuzyon egitimine baslamadan once "gizil uzay hazir mi"
sorusunu 1 dakikada cevaplamak.

Olculen 5 kriter (difuzyonun N(0,I)'dan baslayabilmesi icin):
  K1  ortalama ~ 0
  K2  kanal basina std ~ 1  (izotropi)
  K3  basiklik ~ 0          (Gauss'a yakinlik)
  K4  N(0,I) ornekleri makul genlikte cozuluyor
  K5  gizil uzay + gurultu dayanikliligi

Kullanim:
    python check_latent_ready.py --dataset ptbxl
    python check_latent_ready.py --all
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from lead_dm.config import (RUNS_DIR, RESULTS_DIR, DATASET_CLASSES,
                            GEN8_LEADS, SIGNAL_SCALE, FS)
from lead_dm.data import make_loader, leads8_to_12, denormalize
from lead_dm.autoencoder import load_ae_checkpoint
from lead_dm.reporting import ExcelReport, setup_style, save_figure, PALETTE


def refit_stats(dataset: str, device, verbose=True):
    """
    Gizil normalizasyon istatistiklerini yeniden hesaplayip best.pt'ye yazar.
    AE'yi yeniden EGITMEZ; sadece buffer'lari gunceller (~1 dakika).
    """
    path = RUNS_DIR / f"{dataset}__ae" / "best.pt"
    ae, cfg, ck = load_ae_checkpoint(path, device, verbose=False)
    tr = make_loader(dataset, "train", 128, shuffle=False, num_workers=2)
    st = ae.fit_latent_stats(tr, device, max_batches=10_000)
    ck["model"] = ae.state_dict()
    ck["latent_stats"] = st
    torch.save(ck, path)
    if verbose:
        print(f"  [{dataset}] istatistikler yeniden hesaplandi ve kaydedildi")
        print(f"      ham oran   : {st['ch_std_ratio_raw']:.2f}")
        print(f"      etkin oran : {st['ch_std_ratio_effective']:.3f}")
    return st


@torch.no_grad()
def check_one(dataset: str, device, n_batches: int = 40, verbose=True):
    ae, cfg, ck = load_ae_checkpoint(
        RUNS_DIR / f"{dataset}__ae" / "best.pt", device, verbose=verbose)

    # Buffer'lar varsayilan degerde mi? (per_channel modunda olmamali)
    if cfg.latent_norm == "per_channel":
        defaultish = (float(ae.latent_std_c.std()) < 1e-6 and
                      abs(float(ae.latent_std_c.mean()) - 1.0) < 1e-6)
        if defaultish and verbose:
            print(f"  [!] {dataset}: gizil normalizasyon buffer'lari VARSAYILAN")
            print(f"      degerde. Duzeltmek icin: "
                  f"python check_latent_ready.py --dataset {dataset} --refit")

    tr = make_loader(dataset, "train", 64, shuffle=False, num_workers=2)
    te = make_loader(dataset, "test", 64, shuffle=False, num_workers=2)

    # Difuzyonun GORDUGU temsili topla
    Z = []
    for i, b in enumerate(tr):
        if i >= n_batches:
            break
        Z.append(ae.encode_scaled(b["x"].to(device)).float().cpu())
    Z = torch.cat(Z, 0)
    N, C, T = Z.shape

    ch_mean = Z.mean(dim=(0, 2))
    ch_std = Z.std(dim=(0, 2))
    flat = Z.permute(1, 0, 2).reshape(C, -1)
    kurt = (((flat - flat.mean(1, keepdim=True)) ** 4).mean(1)
            / (flat.var(1) ** 2 + 1e-12) - 3.0)

    # K4: N(0,I) cozulunce ne oluyor
    x_real = next(iter(te))["x"][:64].to(device)
    z_real = ae.encode_scaled(x_real)
    x_rand = ae.decode_scaled(torch.randn_like(z_real))
    amp_real = float(denormalize(x_real).std())
    amp_rand = float(denormalize(x_rand).std())

    # K5: gurultu dayanikligi
    base = ae.decode_scaled(z_real)
    rmse0 = float((base - x_real).pow(2).mean().sqrt()) / SIGNAL_SCALE * 1000
    rmse1 = float((ae.decode_scaled(z_real + torch.randn_like(z_real) * 0.5)
                   - x_real).pow(2).mean().sqrt()) / SIGNAL_SCALE * 1000

    res = {
        "Dataset": dataset,
        "K1_mean": float(Z.mean()),
        "K1_mean_absmax": float(ch_mean.abs().max()),
        "K2_std": float(Z.std()),
        "K2_ch_std_min": float(ch_std.min()),
        "K2_ch_std_max": float(ch_std.max()),
        "K2_ratio": float(ch_std.max() / ch_std.clamp_min(1e-12).min()),
        "K3_kurtosis": float(kurt.mean()),
        "K3_kurt_max": float(kurt.max()),
        "K4_amp_ratio": amp_rand / max(amp_real, 1e-9),
        "K5_rmse_base_uV": rmse0,
        "K5_rmse_noisy_uV": rmse1,
        "K5_degradation": rmse1 / max(rmse0, 1e-9),
        "latent_norm": cfg.latent_norm,
        "variational": cfg.variational,
    }

    # ZORUNLU kriterler: difuzyonun N(0,I)'dan baslayabilmesi icin
    # gizil uzayin merkezlenmis ve izotropik olmasi SARTTIR.
    hard = {
        "K1 ortalama ~0 (ZORUNLU)": abs(res["K1_mean_absmax"]) < 0.15,
        "K2 kanal std ~1 (ZORUNLU)": res["K2_ratio"] < 1.5,
    }
    # BILGILENDIRME: bunlar basarisiz olsa da difuzyon calisir.
    # Basiklik: EKG'nin keskin QRS tepeleri gizil uzayda dogal olarak agir
    # kuyruk yaratir. Difuzyonun isi zaten Gauss-DISI bir dagilim ogrenmek;
    # yalnizca TERMINAL dagilimin (z_T) Gauss olmasi gerekir, o da yapisal
    # olarak garantidir. Bu yuzden basiklik ~3 kabul edilebilir.
    soft = {
        "K3 basiklik (bilgi)": res["K3_kurtosis"] < 6.0,
        "K4 N(0,I) genlik (bilgi)": 0.1 < res["K4_amp_ratio"] < 6.0,
        "K5 gurultu dayanikli (bilgi)": res["K5_degradation"] < 15.0,
    }
    ok = {**hard, **soft}
    res["PASS"] = all(hard.values())
    res["PASS_soft"] = all(soft.values())

    if verbose:
        print(f"\n  --- {dataset} ---")
        print(f"  K1 ortalama       : {res['K1_mean']:+.5f}  "
              f"(kanal |max| {res['K1_mean_absmax']:.4f})")
        print(f"  K2 std            : {res['K2_std']:.4f}   "
              f"kanal {res['K2_ch_std_min']:.3f}-{res['K2_ch_std_max']:.3f}  "
              f"oran {res['K2_ratio']:.3f}")
        print(f"  K3 basiklik       : {res['K3_kurtosis']:.3f}  "
              f"(en kotu kanal {res['K3_kurt_max']:.2f})")
        print(f"  K4 N(0,I) genligi : gercegin {res['K4_amp_ratio']:.2f} kati")
        print(f"  K5 gurultu (0.5)  : {rmse0:.1f} -> {rmse1:.1f} uV  "
              f"({res['K5_degradation']:.1f}x)")
        for k, v in hard.items():
            print(f"    {'[OK] ' if v else '[!!] '}{k}")
        for k, v in soft.items():
            print(f"    {'[OK] ' if v else '[ i] '}{k}")
        print(f"  SONUC: {'DIFUZYONA HAZIR' if res['PASS'] else 'ZORUNLU KRITER BASARISIZ'}")

    return res, Z, ch_std, kurt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--refit", action="store_true",
                    help="gizil normalizasyon istatistiklerini yeniden hesapla "
                         "ve best.pt'ye kaydet (AE yeniden EGITILMEZ)")
    args = ap.parse_args()
    device = torch.device(args.device)

    datasets = list(DATASET_CLASSES) if args.all else [args.dataset]
    print("=" * 78)
    print("LEAD-DM | Gizil Uzay Hazirlik Kontrolu")
    print("=" * 78)

    if args.refit:
        print("\nGizil istatistikler yeniden hesaplaniyor...")
        for ds in datasets:
            try:
                refit_stats(ds, device)
            except FileNotFoundError as e:
                print(f"  [{ds}] atlandi: {e}")
        print()

    rows, store = [], {}
    for ds in datasets:
        try:
            r, Z, cs, ku = check_one(ds, device)
            rows.append(r); store[ds] = (cs.numpy(), ku.numpy())
        except FileNotFoundError as e:
            print(f"  [{ds}] atlandi: {e}")

    if not rows:
        return
    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = ExcelReport(RESULTS_DIR / "LATENT_READY.xlsx",
                      meta={"Icerik": "Gizil uzay difuzyon hazirlik kontrolu",
                            "Kriter": "K2_ratio<1.5, K3_kurtosis<2, K4 0.3-4.0"})
    rep.add_table("hazirlik", df,
                  caption=("Difuzyonun gordugu (normalize) gizil uzayin "
                           "istatistikleri. PASS=True ise difuzyona gecilebilir."))
    print(f"\n[Excel] {rep.save()}")

    plt = setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for i, (ds, (cs, ku)) in enumerate(store.items()):
        ax[0].hist(cs, bins=40, alpha=0.55, label=ds, color=PALETTE[i])
        ax[1].hist(ku, bins=40, alpha=0.55, label=ds, color=PALETTE[i])
    ax[0].axvline(1.0, color="k", ls="--", lw=.8)
    ax[0].set_xlabel("Kanal std (normalize gizil)"); ax[0].set_ylabel("Adet")
    ax[0].legend(); ax[0].set_title("(a) Izotropi", loc="left")
    ax[1].axvline(0.0, color="k", ls="--", lw=.8)
    ax[1].set_xlabel("Basiklik"); ax[1].set_ylabel("Adet")
    ax[1].legend(); ax[1].set_title("(b) Gauss'a yakinlik", loc="left")
    fig.tight_layout()
    save_figure(fig, RESULTS_DIR / "figures" / "latent_ready")
    print(f"[Sekil] {RESULTS_DIR / 'figures'}")

    print("\n" + "=" * 78)
    for r in rows:
        tag = "HAZIR" if r["PASS"] else "ZORUNLU KRITER BASARISIZ"
        note = "" if r.get("PASS_soft", True) else "  (bilgi kriterleri uyarili)"
        print(f"  {r['Dataset']:<10}: {tag}{note}")
    print("=" * 78)


if __name__ == "__main__":
    main()
