#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Ablasyon koşturucu
===============================
Her ablasyon icin uctan uca zinciri calistirir:
    train_diffusion -> generate -> metrics -> evaluate

SURDURULEBILIR: cikti dosyasi zaten varsa o adim ATLANIR. Kesilirse ayni
komutla kaldigi yerden devam eder.

Adil karsilastirma icin ablasyonlar 100k iterasyonda calisir; referans
noktasi M1_leadtcn (100k, tam model) olmalidir. 300k'lik M1_long ile
KIYASLAMAYIN.

Kullanim:
    python run_ablations.py --list                 # plani goster
    python run_ablations.py --only A5_mcfarn A6_adaln
    python run_ablations.py --priority 1           # yalnizca oncelik 1
    python run_ablations.py --dry-run              # komutlari yaz, calistirma
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

# lead_dm/__init__.py torch'u ceker; --list gibi hafif komutlar torch
# gerektirmemeli. Bu yuzden config modulu DOGRUDAN yuklenir.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_ldm_cfg", Path(__file__).parent / "lead_dm" / "config.py")
_cfg = _ilu.module_from_spec(_spec)
sys.modules["_ldm_cfg"] = _cfg          # dataclass modul kaydi ister
_spec.loader.exec_module(_cfg)
RUNS_DIR, RESULTS_DIR, GENERATED_DIR = (_cfg.RUNS_DIR, _cfg.RESULTS_DIR,
                                        _cfg.GENERATED_DIR)


# ============================================================================
# ABLASYON PLANI
#   oncelik 1 : LTCM'in degeri  (makalenin ana iddiasi)
#   oncelik 2 : yapisal kararlar
#   oncelik 3 : ikincil varyantlar
# ============================================================================

ABLATIONS = [
    # (kod, oncelik, aciklama, AE yeniden egitim gerekir mi)
    ("A5_mcfarn", 1, "LTCM yerine MCFARN (referansin modulu)", False),
    ("A6_adaln", 1, "LTCM yerine adaLN-Zero (standart taban)", False),
    ("A8_ltcm_global_only", 1, "LTCM dal B/C kapali (yalniz global)", False),
    ("A4a_symmetric_heads", 2, "Tum basliklar derinlik 2", False),
    ("A4b_chest_deep", 2, "Gogus derin, uzuv sig (ilk sezgi)", False),
    ("A2_flat_latent", 2, "Duz gizil uzay (derivasyon yapisi yok)", True),
    ("A3_single_head", 3, "Tek cikis konvolusyonu", False),
    ("A7_crossattn", 3, "LTCM yerine capraz dikkat", False),
]

# Kayip ablasyonlari (B serisi) - ayri bayraklarla
LOSS_ABLATIONS = [
    ("B1_mse_only", 1, ["--lambda-freq", "0"], "Yalniz MSE"),
    ("B3_pde", 2, ["--lambda-pde", "0.01"], "+ PDE artigi"),
    ("B4_interlead", 2, ["--lambda-interlead", "0.05"],
     "+ derivasyonlar arasi tutarlilik"),
]


def run(cmd: list[str], dry: bool, log: list) -> bool:
    print("    $ " + " ".join(cmd))
    if dry:
        return True
    t0 = time.time()
    r = subprocess.run(cmd)
    dt = (time.time() - t0) / 60
    ok = r.returncode == 0
    log.append({"cmd": " ".join(cmd), "ok": ok, "min": round(dt, 1)})
    if not ok:
        print(f"    [!!] HATA (kod {r.returncode}) - bu ablasyon atlaniyor")
    else:
        print(f"    [OK] {dt:.1f} dk")
    return ok


def chain(name: str, dataset: str, args, extra_train: list | None = None,
          ablation_flag: str | None = None, needs_ae: bool = False,
          log: list | None = None):
    """Tek bir ablasyon icin train -> generate -> metrics -> evaluate."""
    log = log if log is not None else []
    py = sys.executable
    run_dir = RUNS_DIR / f"{dataset}__M1_{name}"
    gen_tag = f"{dataset}__M1_{name}__n{args.gen_limit}"

    print(f"\n{'='*78}\n[{name}]  {dataset}\n{'='*78}")

    # --- 0) gerekiyorsa AE ---
    if needs_ae:
        ae_dir = RUNS_DIR / f"{dataset}__ae_flat"
        if (ae_dir / "best.pt").exists() and not args.force:
            print("  [atla] duz gizil AE zaten var")
        else:
            if not run([py, "train_autoencoder.py", "--dataset", dataset,
                        "--epochs", str(args.ae_epochs), "--flat-latent"],
                       args.dry_run, log):
                return

    # --- 1) egitim ---
    if (run_dir / "best.pt").exists() and not args.force:
        print("  [atla] difuzyon checkpoint'i zaten var")
    else:
        cmd = [py, "train_diffusion.py", "--dataset", dataset,
               "--iters", str(args.iters), "--name", f"M1_{name}"]
        if ablation_flag:
            cmd += ["--ablation", ablation_flag]
        if extra_train:
            cmd += extra_train
        if not run(cmd, args.dry_run, log):
            return

    # --- 2) uretim ---
    if (GENERATED_DIR / f"{gen_tag}.h5").exists() and not args.force:
        print("  [atla] uretilmis veri zaten var")
    else:
        if not run([py, "generate.py", "--dataset", dataset,
                    "--dm-name", f"M1_{name}", "--ddim", str(args.ddim),
                    "--batch", "64", "--limit", str(args.gen_limit)],
                   args.dry_run, log):
            return

    # --- 3) metrikler ---
    if (RESULTS_DIR / f"METRICS_{gen_tag}.xlsx").exists() and not args.force:
        print("  [atla] metrikler zaten var")
    else:
        run([py, "metrics.py", "--dataset", dataset, "--gen", gen_tag],
            args.dry_run, log)

    # --- 4) degerlendirme ---
    if (RESULTS_DIR / f"EVAL_{gen_tag}.xlsx").exists() and not args.force:
        print("  [atla] degerlendirme zaten var")
    else:
        run([py, "evaluate.py", "--dataset", dataset, "--gen", gen_tag,
             "--classifiers", "xresnet1d50",
             "--epochs", str(args.eval_epochs)], args.dry_run, log)


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl")
    ap.add_argument("--iters", type=int, default=100_000,
                    help="ablasyonlar icin; referans M1_leadtcn ile AYNI olmali")
    ap.add_argument("--ddim", type=int, default=250)
    ap.add_argument("--gen-limit", type=int, default=6000)
    ap.add_argument("--eval-epochs", type=int, default=15)
    ap.add_argument("--ae-epochs", type=int, default=120)
    ap.add_argument("--only", nargs="*", default=[])
    ap.add_argument("--priority", type=int, default=0,
                    help="0 = hepsi, 1/2/3 = yalnizca o oncelik")
    ap.add_argument("--losses", action="store_true",
                    help="mimari yerine KAYIP ablasyonlarini calistir")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="mevcut ciktilari yoksay ve yeniden uret")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    items = LOSS_ABLATIONS if args.losses else ABLATIONS

    if args.list:
        print(f"{'kod':<24}{'onc':>4}  aciklama")
        print("-" * 78)
        for it in items:
            code, pri = it[0], it[1]
            desc = it[3] if args.losses else it[2]
            print(f"{code:<24}{pri:>4}  {desc}")
        print("\nTahmini sure (PTB-XL, 100k iterasyon):")
        print("  egitim ~4.4 sa + uretim ~0.5 sa + metrik/degerlendirme ~0.6 sa")
        print("  = ablasyon basina ~5.5 saat")
        n1 = sum(1 for it in items if it[1] == 1)
        print(f"  oncelik 1 ({n1} adet) = ~{n1*5.5:.0f} saat")
        print(f"  tumu ({len(items)} adet) = ~{len(items)*5.5:.0f} saat")
        return

    sel = [it for it in items
           if (not args.only or it[0] in args.only)
           and (args.priority == 0 or it[1] == args.priority)]
    if not sel:
        print("Secilen ablasyon yok."); return

    print("=" * 78)
    print("LEAD-DM | Ablasyon Kosturucu")
    print("=" * 78)
    print(f"Veri seti     : {args.dataset}")
    print(f"Iterasyon     : {args.iters}  "
          f"(referans: M1_leadtcn ayni iterasyonda olmali)")
    print(f"Uretim ornegi : {args.gen_limit}")
    print(f"Calisacak     : {[s[0] for s in sel]}")
    if args.dry_run:
        print("*** DRY-RUN: komutlar yalnizca yazdirilacak ***")

    log, t0 = [], time.time()
    for it in sel:
        if args.losses:
            code, _, flags, _ = it
            chain(code, args.dataset, args, extra_train=flags, log=log)
        else:
            code, _, _, needs_ae = it
            chain(code, args.dataset, args, ablation_flag=code,
                  needs_ae=needs_ae, log=log)

    print("\n" + "=" * 78)
    print(f"BITTI - toplam {(time.time()-t0)/3600:.1f} saat")
    bad = [l for l in log if not l["ok"]]
    if bad:
        print(f"[!!] {len(bad)} komut hata verdi:")
        for b in bad:
            print(f"   {b['cmd']}")
    print("\nSonraki adim:  python collect_results.py")


if __name__ == "__main__":
    main()
