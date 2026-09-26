#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Asama 1 : Derivasyon-yapili enkoder-dekoder egitimi
================================================================
Difuzyon bu agin gizil uzayinda calisacak, o yuzden once bu egitilir.

Cikti:
    runs/<dataset>__ae/best.pt          en iyi checkpoint (+ latent_scale)
    runs/<dataset>__ae/train_log.csv    iterasyon gunlugu
    results/AE_<dataset>.xlsx           Excel raporu (3 tablo)
    results/figures/ae_<dataset>_*.pdf/png   300+ DPI sekiller

Kullanim:
    python train_autoencoder.py --dataset cpsc2018
    python train_autoencoder.py --dataset ptbxl --epochs 150
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from lead_dm.config import (AEConfig, TrainConfig, RUNS_DIR, RESULTS_DIR,
                            DATASET_CLASSES, CANON_12, SIGNAL_SCALE, FS)
from lead_dm.data import make_loader, leads8_to_12, denormalize
from lead_dm.autoencoder import LeadStructuredAE, AELoss
from lead_dm.reporting import (ExcelReport, plot_ecg_12lead, save_figure,
                               new_figure, setup_style, PALETTE, LEAD_COLORS)


# ============================================================================
# DEGERLENDIRME
# ============================================================================

@torch.no_grad()
def evaluate(model, loader, device, max_batches: int | None = None):
    """Rekonstrüksiyon kalitesini sinyal ve derivasyon duzeyinde olcer."""
    model.eval()
    n = 0
    agg = {"mse": 0.0, "mae": 0.0}
    per_lead_se = np.zeros(8)
    per_lead_n = 0
    corr_sum = np.zeros(8)

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        x = batch["x"].to(device, non_blocking=True)
        xh, _, _, _ = model(x)
        b = x.shape[0]
        agg["mse"] += float(F.mse_loss(xh, x)) * b
        agg["mae"] += float(F.l1_loss(xh, x)) * b
        n += b

        xn = x.detach().float().cpu().numpy()
        hn = xh.detach().float().cpu().numpy()
        per_lead_se += ((hn - xn) ** 2).mean(axis=(0, 2)) * b
        # Pearson korelasyonu (kanal basina)
        a = xn - xn.mean(axis=2, keepdims=True)
        c = hn - hn.mean(axis=2, keepdims=True)
        num = (a * c).sum(axis=2)
        den = np.sqrt((a ** 2).sum(axis=2) * (c ** 2).sum(axis=2)) + 1e-8
        corr_sum += (num / den).sum(axis=0)
        per_lead_n += b

    out = {k: v / max(n, 1) for k, v in agg.items()}
    out["rmse"] = out["mse"] ** 0.5
    # mV cinsine cevir
    out["rmse_mV"] = out["rmse"] / SIGNAL_SCALE
    out["mae_mV"] = out["mae"] / SIGNAL_SCALE
    out["per_lead_rmse_mV"] = np.sqrt(per_lead_se / max(per_lead_n, 1)) / SIGNAL_SCALE
    out["per_lead_corr"] = corr_sum / max(per_lead_n, 1)
    return out


@torch.no_grad()
def latent_stats(model, loader, device, max_batches: int = 50):
    """Gizil uzayin istatistikleri — difuzyon icin kritik."""
    model.eval()
    zs = []
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        z = model.encode(batch["x"].to(device), sample=False)
        zs.append(z.float().cpu())
    Z = torch.cat(zs, 0)                        # (N, C, T)
    B, C, T = Z.shape
    L = model.cfg.n_leads
    Zl = Z.view(B, L, C // L, T)
    ch_std = Z.std(dim=(0, 2))
    flat = Z.permute(1, 0, 2).reshape(C, -1)
    kurt = float((((flat - flat.mean(1, keepdim=True)) ** 4).mean(1) /
                  (flat.var(1) ** 2 + 1e-12) - 3.0).mean())
    return {
        "shape": tuple(Z.shape[1:]),
        "mean": float(Z.mean()),
        "std": float(Z.std()),
        "rms": float(Z.pow(2).mean().sqrt()),
        "min": float(Z.min()),
        "max": float(Z.max()),
        "per_lead_std": Zl.std(dim=(0, 2, 3)).numpy(),
        "ch_std_ratio": float(ch_std.max() / ch_std.clamp_min(1e-12).min()),
        "kurtosis": kurt,
    }


# ============================================================================
# EGITIM
# ============================================================================

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)

    ae_cfg = AEConfig(lead_independent=not args.flat_latent,
                      variational=not args.no_vae,
                      kl_weight=args.kl_weight,
                      latent_norm=args.latent_norm)
    run = RUNS_DIR / f"{args.dataset}__ae{'_flat' if args.flat_latent else ''}"
    run.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("LEAD-DM | Asama 1: Enkoder-Dekoder Egitimi")
    print("=" * 78)
    print(f"Veri seti      : {args.dataset}")
    print(f"Gizil uzay     : ({ae_cfg.latent_ch}, {ae_cfg.latent_len})"
          f"  = ({ae_cfg.n_leads} lead x {ae_cfg.ch_per_lead} kanal, "
          f"{ae_cfg.latent_len} adim)")
    print(f"Lead-bagimsiz  : {ae_cfg.lead_independent}")
    print(f"VAE (KL)       : {ae_cfg.variational}  kl_weight={ae_cfg.kl_weight}")
    print(f"Gizil norm     : {ae_cfg.latent_norm}")
    print(f"Cihaz          : {device}")

    tr = make_loader(args.dataset, "train", args.batch, shuffle=True,
                     num_workers=args.workers, seed=args.seed)
    va = make_loader(args.dataset, "val", args.batch, shuffle=False,
                     num_workers=max(1, args.workers // 2))
    te = make_loader(args.dataset, "test", args.batch, shuffle=False,
                     num_workers=max(1, args.workers // 2))
    print(f"Ornekler       : train={len(tr.dataset)}  "
          f"val={len(va.dataset)}  test={len(te.dataset)}")

    model = LeadStructuredAE(ae_cfg).to(device)
    print(f"Parametre      : {model.n_params()/1e6:.3f} M")

    crit = AELoss(w_l1=args.w_l1, w_spec=args.w_spec, kl_weight=args.kl_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-6)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs * len(tr), eta_min=args.lr * 0.02)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    log_rows = []
    best = float("inf")
    step = 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        ep_loss, nb = 0.0, 0
        for batch in tr:
            x = batch["x"].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda",
                                    enabled=args.amp and device.type == "cuda"):
                xh, _, mu, logvar = model(x)
                loss, parts = crit(xh, x, mu, logvar)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            ep_loss += float(loss); nb += 1; step += 1

        vm = evaluate(model, va, device)
        row = {"epoch": ep, "step": step,
               "train_loss": ep_loss / max(nb, 1),
               "val_mse": vm["mse"], "val_rmse_mV": vm["rmse_mV"],
               "val_mae_mV": vm["mae_mV"],
               "val_corr_mean": float(vm["per_lead_corr"].mean()),
               "lr": sched.get_last_lr()[0],
               "time_s": time.time() - t0}
        log_rows.append(row)

        if ep % args.print_every == 0 or ep == 1:
            print(f"  ep {ep:>3}/{args.epochs}  "
                  f"train={row['train_loss']:.5f}  "
                  f"val_rmse={row['val_rmse_mV']*1000:.2f} uV  "
                  f"corr={row['val_corr_mean']:.4f}  "
                  f"({row['time_s']/60:.1f} dk)")

        if vm["mse"] < best:
            best = vm["mse"]
            # DIKKAT: bu ara kayit fit_latent_stats CAGRILMADAN yapilir,
            # yani gizil normalizasyon buffer'lari varsayilan degerdedir.
            # Bayrak, yarida kesilmis egitimin sessizce kullanilmasini onler.
            torch.save({"model": model.state_dict(),
                        "cfg": ae_cfg.__dict__, "epoch": ep,
                        "val_mse": best,
                        "latent_stats_fitted": False}, run / "best.pt")

    # ---- en iyi checkpoint'i yukle, gizil olcegi hesapla ----
    ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    lstat = model.fit_latent_stats(tr, device, max_batches=200)
    torch.save({"model": model.state_dict(), "cfg": ae_cfg.__dict__,
                "epoch": ck["epoch"], "val_mse": ck["val_mse"],
                "latent_stats": lstat,
                "latent_stats_fitted": True}, run / "best.pt")
    print("\nGizil uzay istatistikleri:")
    for k, v in lstat.items():
        print(f"  {k:<16}: {v}")
    eff = lstat.get("ch_std_ratio_effective", lstat.get("ch_std_ratio_raw", 1.0))
    print(f"\n  Difuzyonun gordugu (normalize) kanal std orani: {eff:.3f}")
    if eff > 3.0:
        print("  [!] UYARI: normalizasyon SONRASI anizotropi > 3x.")
        print("      latent_norm=per_channel kullanin veya kl_weight artirin.")
    else:
        print("  [OK] Normalize gizil uzay izotropik. Difuzyona hazir.")
        if lstat.get("ch_std_ratio_raw", 1) > 5:
            print("       (Ham oran yuksek gorunuyor ama kanal basina")
            print("        normalizasyon bunu tanim geregi 1.0'a indiriyor.)")

    # ---- final degerlendirme ----
    tm = evaluate(model, te, device)
    ls = latent_stats(model, tr, device)
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(run / "train_log.csv", index=False)

    print(f"\nTest RMSE : {tm['rmse_mV']*1000:.2f} uV")
    print(f"Test MAE  : {tm['mae_mV']*1000:.2f} uV")
    print(f"Test korr : {tm['per_lead_corr'].mean():.4f}")

    write_report(args, ae_cfg, model, tm, ls, log_df, te, device, run)
    return model


# ============================================================================
# RAPOR
# ============================================================================

def write_report(args, ae_cfg, model, test_metrics, lstats, log_df,
                 test_loader, device, run):
    from lead_dm.config import GEN8_LEADS
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figdir = RESULTS_DIR / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}{'_flat' if args.flat_latent else ''}"

    # ---------------- Excel ----------------
    rep = ExcelReport(
        RESULTS_DIR / f"AE_{tag}.xlsx",
        meta={
            "Asama": "1 - Derivasyon-yapili enkoder-dekoder",
            "Veri seti": args.dataset,
            "Gizil uzay": f"({ae_cfg.latent_ch}, {ae_cfg.latent_len})",
            "Lead-bagimsiz": ae_cfg.lead_independent,
            "Parametre (M)": round(model.n_params() / 1e6, 4),
            "Epoch": args.epochs,
            "Batch": args.batch,
            "Ogrenme orani": args.lr,
            "VAE (KL)": ae_cfg.variational,
            "kl_weight": ae_cfg.kl_weight,
            "Gizil norm": ae_cfg.latent_norm,
            "Gizil olcek (skaler)": round(float(model.latent_scale), 5),
            "Olusturulma": time.strftime("%Y-%m-%d %H:%M"),
        })

    # Tablo A: genel rekonstrüksiyon
    dfA = pd.DataFrame([{
        "Metric": "RMSE (uV)", "Value": test_metrics["rmse_mV"] * 1000},
        {"Metric": "MAE (uV)", "Value": test_metrics["mae_mV"] * 1000},
        {"Metric": "MSE (olcekli)", "Value": test_metrics["mse"]},
        {"Metric": "Pearson r (ort.)",
         "Value": float(test_metrics["per_lead_corr"].mean())},
        {"Metric": "Gizil RMS", "Value": lstats["rms"]},
        {"Metric": "Gizil std", "Value": lstats["std"]},
        {"Metric": "Kanal std orani (hedef <3)", "Value": lstats["ch_std_ratio"]},
        {"Metric": "Basiklik (hedef ~0)", "Value": lstats["kurtosis"]},
        {"Metric": "Parametre (M)", "Value": model.n_params() / 1e6},
    ])
    rep.add_table("A_rekonstruksiyon", dfA,
                  caption="Tablo A - Test setinde rekonstrüksiyon kalitesi.")

    # Tablo B: derivasyon basina
    dfB = pd.DataFrame({
        "Lead": list(GEN8_LEADS),
        "RMSE(uV)": test_metrics["per_lead_rmse_mV"] * 1000,
        "Pearson_r": test_metrics["per_lead_corr"],
        "Latent_std": lstats["per_lead_std"],
        "Group": ["Chest"] * 6 + ["Limb"] * 2,
    })
    rep.add_table("B_lead_bazinda", dfB,
                  highlight={"RMSE(uV)": "min", "Pearson_r": "max"},
                  caption=("Tablo B - Derivasyon basina rekonstrüksiyon. "
                           "Gogus/uzuv farki asimetrik baslik tasariminin "
                           "gerekcesini destekler."))

    # Tablo C: egitim gunlugu
    rep.add_table("C_egitim_gunlugu", log_df,
                  caption="Tablo C - Epoch bazli egitim gunlugu.")

    path = rep.save()
    print(f"\n[Excel] {path}")

    # ---------------- Sekiller ----------------
    plt = setup_style()

    # Sekil 1: egitim egrileri
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    axes[0].plot(log_df["epoch"], log_df["train_loss"], color=PALETTE[0],
                 label="egitim")
    axes[0].plot(log_df["epoch"], log_df["val_mse"], color=PALETTE[1],
                 label="dogrulama")
    axes[0].set_yscale("log"); axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Kayip"); axes[0].legend(); axes[0].grid(True)
    axes[0].set_title("(a) Kayip", loc="left")

    axes[1].plot(log_df["epoch"], log_df["val_rmse_mV"] * 1000,
                 color=PALETTE[2])
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("RMSE ($\\mu$V)")
    axes[1].grid(True); axes[1].set_title("(b) Rekonstrüksiyon hatasi",
                                          loc="left")

    axes[2].plot(log_df["epoch"], log_df["val_corr_mean"], color=PALETTE[3])
    axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Pearson $r$")
    axes[2].grid(True); axes[2].set_title("(c) Korelasyon", loc="left")
    fig.tight_layout()
    save_figure(fig, figdir / f"ae_{tag}_training")

    # Sekil 2: derivasyon basina hata
    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    xs = np.arange(8)
    cols = [LEAD_COLORS[l] for l in GEN8_LEADS]
    ax.bar(xs, test_metrics["per_lead_rmse_mV"] * 1000, color=cols, width=0.65)
    ax.set_xticks(xs); ax.set_xticklabels(GEN8_LEADS)
    ax.set_ylabel("RMSE ($\\mu$V)"); ax.grid(True, axis="y")
    ax.set_title("Derivasyon basina rekonstrüksiyon hatasi", loc="left")
    fig.tight_layout()
    save_figure(fig, figdir / f"ae_{tag}_per_lead")

    # Sekil 3: ornek rekonstrüksiyonlar (12 derivasyon, klinik duzen)
    model.eval()
    batch = next(iter(test_loader))
    with torch.no_grad():
        x = batch["x"][:1].to(device)
        xh, _, _, _ = model(x)
    x12_true = leads8_to_12(denormalize(x))[0].cpu().numpy()
    x12_rec = leads8_to_12(denormalize(xh))[0].cpu().numpy()
    plot_ecg_12lead(x12_rec, figdir / f"ae_{tag}_reconstruction", fs=FS,
                    second=x12_true, labels=("Rekonstrüksiyon", "Gercek"),
                    title="Enkoder-dekoder rekonstrüksiyonu (12 derivasyon)")

    # Sekil 4: gizil uzay yapisi
    with torch.no_grad():
        z = model.encode(x)[0].cpu().numpy()
    L, cpl = model.cfg.n_leads, model.cfg.ch_per_lead
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    im = ax.imshow(z, aspect="auto", cmap="RdBu_r",
                   vmin=-np.abs(z).max(), vmax=np.abs(z).max())
    for l in range(1, L):
        ax.axhline(l * cpl - 0.5, color="k", lw=0.6)
    ax.set_yticks([(l + 0.5) * cpl for l in range(L)])
    ax.set_yticklabels(GEN8_LEADS)
    ax.set_xlabel("Gizil zaman adimi"); ax.set_ylabel("Derivasyon blogu")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    ax.set_title("Derivasyon-yapili gizil uzay $z_0$ "
                 f"({model.cfg.latent_ch}$\\times${model.cfg.latent_len})",
                 loc="left")
    fig.tight_layout()
    save_figure(fig, figdir / f"ae_{tag}_latent")

    print(f"[Sekil] {figdir}  (PDF + PNG, {400} DPI)")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cpsc2018",
                   choices=list(DATASET_CLASSES))
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--w-l1", type=float, default=0.1)
    p.add_argument("--w-spec", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--print-every", type=int, default=5)
    p.add_argument("--kl-weight", type=float, default=1e-6,
                   help="KL duzenlileştirme agirligi (0 -> duz otokodlayici)")
    p.add_argument("--no-vae", action="store_true",
                   help="VAE yerine duz otokodlayici (ablasyon)")
    p.add_argument("--latent-norm", default="per_channel",
                   choices=["scalar", "per_channel"])
    p.add_argument("--flat-latent", action="store_true",
                   help="Ablasyon A2: derivasyon yapisi olmayan duz gizil uzay")
    train(p.parse_args())


if __name__ == "__main__":
    main()
