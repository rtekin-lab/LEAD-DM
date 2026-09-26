#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Asama 2 : M1 difuzyon modeli egitimi
=================================================
Onkosul: train_autoencoder.py calistirilmis olmali.

Cikti:
    runs/<dataset>__<name>/best.pt         model + EMA + config
    runs/<dataset>__<name>/train_log.csv
    results/DM_<dataset>_<name>.xlsx
    results/figures/dm_<...>_*.pdf/png

Kullanim:
    python train_diffusion.py --dataset cpsc2018
    python train_diffusion.py --dataset ptbxl --iters 100000
    python train_diffusion.py --dataset cpsc2018 --ablation A5_mcfarn
    python train_diffusion.py --dataset cpsc2018 --cond-dropout 0.1   # CFG
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lead_dm.config import (ExperimentConfig, AEConfig, M1Config,
                            DiffusionConfig, ablation,
                            RUNS_DIR, RESULTS_DIR, DATASET_CLASSES,
                            GEN8_LEADS, FS)
from lead_dm.data import make_loader, leads8_to_12, denormalize
from lead_dm.autoencoder import LeadStructuredAE, load_ae_checkpoint
from lead_dm.backbone_m1 import build_m1
from lead_dm.diffusion import GaussianDiffusion, EMA
from lead_dm.reporting import (ExcelReport, plot_ecg_12lead, plot_lead_attention,
                               save_figure, setup_style, PALETTE)

DEFAULT_NAME = "M1_leadtcn"


# ============================================================================
def load_autoencoder(dataset: str, device, flat: bool = False):
    run = RUNS_DIR / f"{dataset}__ae{'_flat' if flat else ''}"
    ckpt = run / "best.pt"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Enkoder-dekoder bulunamadi: {ckpt}\n"
            f"Once calistirin: python train_autoencoder.py --dataset {dataset}")
    ae, cfg, ck = load_ae_checkpoint(ckpt, device)
    for p in ae.parameters():
        p.requires_grad_(False)
    ae.source_path = str(ckpt)      # hangi AE kullanildigi kayit altina alinir
    ls = ck.get("latent_stats", {})
    print(f"  AE yuklendi   : epoch {ck['epoch']}, val_mse {ck['val_mse']:.5f}")
    print(f"  Gizil norm    : {cfg.latent_norm}  VAE={cfg.variational}")
    if ls:
        eff = ls.get("ch_std_ratio_effective",
                     ls.get("ch_std_ratio_raw", ls.get("ch_std_ratio", 1.0)))
        print(f"  kanal std     : min={ls.get('ch_std_min',0):.4f} "
              f"med={ls.get('ch_std_med',0):.4f} "
              f"max={ls.get('ch_std_max',0):.4f}")
        print(f"  etkin oran    : {eff:.3f}x  (difuzyonun gordugu; ~1.0 olmali)")
        if eff > 3.0:
            print("  [!] UYARI: gizil uzay anizotropik. Uretilen ornekler")
            print("      gurultu cikabilir. AE'yi KL ile yeniden egitin.")
    return ae, cfg


@torch.no_grad()
def validation_loss(model, diff, ae, loader, device, max_batches=20):
    model.eval()
    tot, n = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        x = batch["x"].to(device)
        b = {k: (v.to(device) if torch.is_tensor(v) else v)
             for k, v in batch.items()}
        z0 = ae.encode_scaled(x)
        loss, _ = diff.loss(model, z0, b, x8_true=x)
        tot += float(loss) * x.shape[0]; n += x.shape[0]
    return tot / max(n, 1)


# ============================================================================
def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    # --- konfigurasyon ---
    if args.ablation:
        cfg = ablation(args.ablation)
        # --name verilmisse ablasyonun otomatik adini EZER. Aksi halde
        # --ablation X --name Y komutu sessizce runs/<ds>__M1_X klasorune
        # yazar ve varsa oradaki checkpoint'i EZER (bir kez yasandi).
        if args.name and args.name != DEFAULT_NAME:
            cfg.name = args.name
    else:
        cfg = ExperimentConfig(name=args.name)
    cfg.train.dataset = args.dataset

    # Var olan bir kosunun uzerine yazilacaksa uyar
    _rd = cfg.run_dir()
    if (_rd / "best.pt").exists() and not args.overwrite:
        _old = torch.load(_rd / "best.pt", map_location="cpu",
                          weights_only=False)
        print(f"\n  [!] {_rd} zaten bir checkpoint iceriyor "
              f"(step={_old.get('step')}, val={_old.get('val')})")
        print("      Egitim bunun UZERINE yazacak. Farkli bir --name verin")
        print("      ya da bilerek yapiyorsaniz --overwrite ekleyin.")
        raise SystemExit(1)
    cfg.train.dm_iters = args.iters
    cfg.train.dm_batch = args.batch
    cfg.train.dm_lr = args.lr
    cfg.diffusion.cond_dropout = args.cond_dropout
    cfg.diffusion.joint_dropout = args.joint_dropout
    cfg.diffusion.lambda_freq = args.lambda_freq
    cfg.diffusion.lambda_pde = args.lambda_pde
    cfg.diffusion.lambda_interlead = args.lambda_interlead
    cfg.diffusion.parameterization = args.param
    cfg.diffusion.schedule = args.schedule
    cfg.diffusion.zero_terminal_snr = not args.no_zero_snr

    run = cfg.run_dir(); run.mkdir(parents=True, exist_ok=True)
    cfg.save(run / "config.json")

    n_classes = len(DATASET_CLASSES[args.dataset])

    print("=" * 78)
    print(f"LEAD-DM | Asama 2: {cfg.name} difuzyon egitimi")
    print("=" * 78)
    print(f"Veri seti     : {args.dataset}  ({n_classes} sinif)")
    print(f"Modulasyon    : {cfg.model.modulation}")
    print(f"Asimetrik bas : {cfg.model.asymmetric_heads} "
          f"(gogus {cfg.model.chest_head_depth}, uzuv {cfg.model.limb_head_depth})")
    print(f"Karisim yolu  : {cfg.model.use_mix_path}")
    print(f"Kosul dusurme : {cfg.diffusion.cond_dropout} "
          f"(ortak {cfg.diffusion.joint_dropout})")
    print(f"Parametrizasyon: {cfg.diffusion.parameterization}   "
          f"zamanlama={cfg.diffusion.schedule}   "
          f"sifir-terminal-SNR={cfg.diffusion.zero_terminal_snr}")
    print(f"Cihaz         : {device}")

    # --- veri ---
    flat = (args.ablation == "A2_flat_latent")
    tr = make_loader(args.dataset, "train", args.batch, shuffle=True,
                     num_workers=args.workers,
                     cond_dropout=cfg.diffusion.cond_dropout,
                     joint_dropout=cfg.diffusion.joint_dropout,
                     seed=args.seed)
    va = make_loader(args.dataset, "val", args.batch, shuffle=False,
                     num_workers=max(1, args.workers // 2))
    print(f"Ornekler      : train={len(tr.dataset)} val={len(va.dataset)}")

    # --- modeller ---
    ae, ae_cfg = load_autoencoder(args.dataset, device, flat=flat)
    ae_rel = f"{args.dataset}__ae{'_flat' if flat else ''}"
    cfg.model.ch_per_lead = ae_cfg.ch_per_lead
    cfg.model.n_leads = ae_cfg.n_leads

    model = build_m1(n_classes, cfg.model).to(device)
    diff = GaussianDiffusion(cfg.diffusion, autoencoder=ae).to(device)

    print("\nParametreler:")
    for k, v in model.param_summary().items():
        print(f"  {k:<15}: {v:>12,}")
    print(f"  {'AE (donuk)':<15}: {ae.n_params():>12,}")

    ema = EMA(model, cfg.train.ema_decay)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            betas=(0.9, 0.999), weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda",
                                  enabled=args.amp and device.type == "cuda")

    def lr_at(step):
        if step < cfg.train.dm_warmup:
            return args.lr * step / max(cfg.train.dm_warmup, 1)
        p = (step - cfg.train.dm_warmup) / max(
            args.iters - cfg.train.dm_warmup, 1)
        return args.lr * (0.02 + 0.98 * 0.5 * (1 + np.cos(np.pi * p)))

    # --- egitim dongusu ---
    log_rows, best_val = [], float("inf")
    step, t0 = 0, time.time()
    it = iter(tr)
    print(f"\nEgitim basliyor: {args.iters} iterasyon\n")

    while step < args.iters:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tr); batch = next(it)

        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        x = batch["x"].to(device, non_blocking=True)
        b = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
             for k, v in batch.items()}

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda",
                                enabled=args.amp and device.type == "cuda"):
            with torch.no_grad():
                z0 = ae.encode_scaled(x)
            loss, parts = diff.loss(model, z0, b, x8_true=x)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               cfg.train.grad_clip)
        scaler.step(opt); scaler.update()
        ema.update(model)
        step += 1

        if step % cfg.train.log_every == 0:
            row = {"step": step, "loss": float(loss),
                   "lr": opt.param_groups[0]["lr"],
                   "grad_norm": float(gnorm),
                   "time_s": time.time() - t0}
            for k, v in parts.items():
                row[k] = float(v)
            log_rows.append(row)

        if step % cfg.train.val_every == 0 or step == args.iters:
            ema.apply_to(model)
            vl = validation_loss(model, diff, ae, va, device)
            ema.restore(model)
            if log_rows:
                log_rows[-1]["val_loss"] = vl
            lam = getattr(model.blocks[0].mod1, "lam", None)
            lam_s = f"  lam={float(lam):.4f}" if lam is not None else ""
            print(f"  it {step:>7}/{args.iters}  "
                  f"loss={float(loss):.4f}  val={vl:.4f}  "
                  f"lr={opt.param_groups[0]['lr']:.2e}{lam_s}  "
                  f"({(time.time()-t0)/60:.1f} dk)")
            if vl < best_val:
                best_val = vl
                torch.save({"model": model.state_dict(),
                            "ema": ema.state_dict(),
                            "cfg": cfg.name, "step": step, "val": vl,
                            "n_classes": n_classes,
                            # KRITIK: uretim ayni AE'yi kullanmali. Bu alan
                            # olmadan generate.py varsayilan AE'yi yukler ve
                            # gizil uzaylar uyusmaz (A2 ablasyonunda oldu).
                            "ae_dir": ae_rel},
                           run / "best.pt")

        if step % cfg.train.ckpt_every == 0:
            torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                        "step": step}, run / "last.pt")

    # --- kaydet ve raporla ---
    log_df = pd.DataFrame(log_rows)
    log_df.to_csv(run / "train_log.csv", index=False)
    torch.save({"model": model.state_dict(), "ema": ema.state_dict(),
                "step": step, "n_classes": n_classes,
                "ae_dir": ae_rel}, run / "last.pt")

    print(f"\nEgitim bitti. En iyi dogrulama kaybi: {best_val:.5f}")
    write_report(args, cfg, model, ema, ae, diff, log_df, va, device, run,
                 n_classes, time.time() - t0)


# ============================================================================
def write_report(args, cfg, model, ema, ae, diff, log_df, val_loader,
                 device, run, n_classes, elapsed_s):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figdir = RESULTS_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_{cfg.name}"
    classes = DATASET_CLASSES[args.dataset]

    ema.apply_to(model)
    model.eval()

    # ---------------- Excel ----------------
    ps = model.param_summary()
    mem = (torch.cuda.max_memory_allocated() / 1e6
           if device.type == "cuda" else float("nan"))

    rep = ExcelReport(
        RESULTS_DIR / f"DM_{tag}.xlsx",
        meta={
            "Asama": "2 - Difuzyon egitimi",
            "Model": cfg.name,
            "Veri seti": args.dataset,
            "Modulasyon": cfg.model.modulation,
            "Asimetrik baslik": cfg.model.asymmetric_heads,
            "Karisim yolu": cfg.model.use_mix_path,
            "Blok sayisi": cfg.model.n_blocks,
            "Difuzyon adimi": cfg.diffusion.n_steps,
            "lambda_freq": cfg.diffusion.lambda_freq,
            "lambda_pde": cfg.diffusion.lambda_pde,
            "lambda_interlead": cfg.diffusion.lambda_interlead,
            "Kosul dusurme": cfg.diffusion.cond_dropout,
            "Iterasyon": args.iters,
            "Egitim suresi (h)": round(elapsed_s / 3600, 3),
            "Olusturulma": time.strftime("%Y-%m-%d %H:%M"),
        })

    # Tablo A: parametre dagilimi
    dfA = pd.DataFrame([{"Component": k, "Params": v,
                         "Params(M)": v / 1e6} for k, v in ps.items()])
    rep.add_table("A_parametreler", dfA,
                  caption="Tablo A - Bilesen bazinda parametre dagilimi.")

    # Tablo B: verimlilik (referans Tablo 5 formati)
    dfB = pd.DataFrame([{
        "Model": cfg.name,
        "Params(M)": ps["TOTAL"] / 1e6,
        "Memory(MB)": mem,
        "TrainTime(h)": elapsed_s / 3600,
        "Inference(s)": float("nan"),   # generate.py dolduracak
    }])
    rep.add_table("B_verimlilik", dfB,
                  caption=("Tablo B - Verimlilik (referans makale Tablo 5 "
                           "formati). Cikarim suresi generate.py ile olculur."))

    # Tablo C: egitim gunlugu
    rep.add_table("C_egitim_gunlugu", log_df,
                  caption="Tablo C - Iterasyon bazli egitim gunlugu.")

    # Tablo D: LTCM lambda degerleri (blok basina ogrenilen guc)
    lams = []
    for i, blk in enumerate(model.blocks):
        for path, mod in (("local", blk.mod1),
                          ("mix", getattr(blk, "mod2", None))):
            lam = getattr(mod, "lam", None) if mod is not None else None
            if lam is not None:
                lams.append({"Block": i, "Path": path,
                             "lambda": float(lam)})
    if lams:
        rep.add_table("D_LTCM_lambda", pd.DataFrame(lams),
                      caption=("Tablo D - Blok basina ogrenilen LTCM lambda. "
                               "0'dan basladi; buyudukce derivasyon/zaman "
                               "secicilikinin katkisi artmistir."))

    # Tablo E: derivasyon ilgi haritasi (LTCM)
    W = None
    if model.use_lead_selector:
        eye = torch.eye(n_classes, device=device)
        age = torch.from_numpy(
            np.tile(np.array([0, 1, 1, 0, 1, 1, 0], dtype=np.float32),
                    (n_classes, 1))).to(device)      # ~60 yas civari
        sex = torch.zeros(n_classes, dtype=torch.long, device=device)
        with torch.no_grad():
            W = model.lead_attention_map(eye, age, sex).cpu().numpy()
        dfE = pd.DataFrame(W, columns=list(GEN8_LEADS))
        dfE.insert(0, "Disease", classes)
        rep.add_table("E_lead_ilgi", dfE, highlight="max",
                      caption=("Tablo E - LTCM derivasyon ilgi agirliklari. "
                               "1.0 notr; >1 model o derivasyonu vurguluyor."))

    path = rep.save()
    print(f"[Excel] {path}")

    # ---------------- Sekiller ----------------
    plt = setup_style()

    # Sekil 1: egitim egrileri
    comp = [c for c in ["main_mse", "eps_mse", "eps_mse_eq", "z0_mse",
                        "freq", "interlead", "pde"]
            if c in log_df.columns]
    fig, axes = plt.subplots(1, 2 + (1 if lams else 0),
                             figsize=(7.2 if not lams else 9.6, 2.6))
    ax = axes[0]
    ax.plot(log_df["step"], log_df["loss"], color=PALETTE[0], lw=0.8,
            label="egitim")
    if "val_loss" in log_df.columns:
        v = log_df.dropna(subset=["val_loss"])
        ax.plot(v["step"], v["val_loss"], color=PALETTE[1], lw=1.2,
                marker="o", ms=2.5, label="dogrulama")
    ax.set_xlabel("Iterasyon"); ax.set_ylabel("Toplam kayip")
    ax.set_yscale("log"); ax.grid(True); ax.legend()
    ax.set_title("(a) Kayip", loc="left")

    ax = axes[1]
    for i, c in enumerate(comp):
        ax.plot(log_df["step"], log_df[c], color=PALETTE[i], lw=0.8, label=c)
    ax.set_xlabel("Iterasyon"); ax.set_ylabel("Bilesen")
    ax.set_yscale("log"); ax.grid(True); ax.legend()
    ax.set_title("(b) Kayip bilesenleri", loc="left")

    if lams:
        ax = axes[2]
        dl = pd.DataFrame(lams)
        for path_name, g in dl.groupby("Path"):
            ax.plot(g["Block"], g["lambda"], marker="o", ms=3,
                    label=path_name)
        ax.set_xlabel("Blok"); ax.set_ylabel("$\\lambda_{LTCM}$")
        ax.grid(True); ax.legend()
        ax.set_title("(c) Ogrenilen LTCM gucu", loc="left")
    fig.tight_layout()
    save_figure(fig, figdir / f"dm_{tag}_training")

    # Sekil 2: derivasyon ilgi haritasi
    if W is not None:
        plot_lead_attention(
            W, classes, list(GEN8_LEADS),
            figdir / f"dm_{tag}_lead_attention",
            title=("LTCM derivasyon ilgi haritasi "
                   f"({args.dataset.upper()})"))

    # Sekil 3: hizli ornek (DDIM 50 adim — sadece gorsel kontrol)
    try:
        batch = next(iter(val_loader))
        k = min(1, batch["x"].shape[0])
        cond = {"disease": batch["disease"][:k].to(device),
                "age_bits": batch["age_bits"][:k].to(device),
                "sex": batch["sex"][:k].to(device)}
        C = model.channels
        T = ae.cfg.latent_len
        z = diff.sample(model, cond, (k, C, T), device,
                        ddim_steps=50, progress=False)
        with torch.no_grad():
            x8 = ae.decode_scaled(z)
        x12 = leads8_to_12(denormalize(x8))[0].cpu().numpy()
        lbl = [c for c, v in zip(classes, batch["disease"][0].tolist()) if v > 0]
        plot_ecg_12lead(
            x12, figdir / f"dm_{tag}_sample",
            fs=FS,
            title=("Uretilen 12 derivasyon (DDIM-50, on izleme)  |  "
                   f"kosul: {', '.join(lbl) or 'NORM'}"))
    except Exception as e:
        print(f"  [i] on izleme ornegi uretilemedi: {e}")

    ema.restore(model)
    print(f"[Sekil] {figdir}  (PDF + PNG, 400 DPI)")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="cpsc2018", choices=list(DATASET_CLASSES))
    p.add_argument("--name", default=DEFAULT_NAME)
    p.add_argument("--overwrite", action="store_true",
                   help="var olan checkpoint'in uzerine yazmaya izin ver")
    p.add_argument("--ablation", default="",
                   help="A1_full|A2_flat_latent|A3_single_head|A4a_symmetric_heads|"
                        "A4b_chest_deep|A4c_limb_deep|A5_mcfarn|A6_adaln|"
                        "A7_crossattn|A8_ltcm_global_only")
    p.add_argument("--iters", type=int, default=100_000)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lambda-freq", type=float, default=0.1)
    p.add_argument("--lambda-pde", type=float, default=0.0)
    p.add_argument("--lambda-interlead", type=float, default=0.0)
    p.add_argument("--param", default="v", choices=["v", "eps"],
                   help="v onerilir; eps yuksek t'de varyans sismesine yol acar")
    p.add_argument("--schedule", default="cosine", choices=["cosine", "linear"])
    p.add_argument("--no-zero-snr", action="store_true",
                   help="sifir terminal SNR'i kapat (ablasyon)")
    p.add_argument("--cond-dropout", type=float, default=0.10)
    p.add_argument("--joint-dropout", type=float, default=0.05)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--amp", action="store_true", default=True)
    train(p.parse_args())


if __name__ == "__main__":
    main()
