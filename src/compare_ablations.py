#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Ablasyon karsilastirmasi (eslestirilmis bootstrap)
===============================================================
evaluate.py'nin kaydettigi tahminler (results/predictions/*.npz) uzerinde
ISTATISTIKSEL karsilastirma yapar. Model YENIDEN EGITILMEZ.

NEDEN ESLESTIRILMIS BOOTSTRAP
------------------------------
Bagimsiz %95 guven araliklarinin ortusmesi "fark yok" DEMEK DEGILDIR;
zayif bir testtir. Tum modeller AYNI test seti uzerinde olculduguÌˆ icin
FARKIN dagilimi dogrudan bootstraplanabilir:

    her tekrar : ayni ornek indeksleri ile iki modelin metrigi hesaplanir
                 d = metrik(A) - metrik(B) kaydedilir
    sonuc      : d'nin %95 araligi. 0'i icermiyorsa fark ANLAMLI.

Bu yontem ornek-duzeyi korelasyonu hesaba kattigi icin bagimsiz GA
karsilastirmasindan belirgin olarak daha gucludur.

DIKKAT: bu test yalnizca TEST SETI belirsizligini olcer. Egitim tohumu
degiskenligini olcmez; onun icin farkli tohumlarla tekrar gerekir.

Kullanim:
    python compare_ablations.py --ref ptbxl__M1_leadtcn__n6000 \\
        --others ptbxl__M1_A5_mcfarn__n6000 ptbxl__M1_A6_adaln__n6000 \\
                 ptbxl__M1_A8_ltcm_global_only__n6000
    python compare_ablations.py --auto ptbxl      # otomatik esle
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from lead_dm.config import RESULTS_DIR
from lead_dm.reporting import ExcelReport, setup_style, save_figure, PALETTE

PRED_DIR = RESULTS_DIR / "predictions"


# ============================================================================
def metric_fn(y_true, y_prob, which="F1", thr=0.5):
    if which == "F1":
        return f1_score(y_true, (y_prob >= thr).astype(int),
                        average="macro", zero_division=0)
    if which == "AUC":
        a = [roc_auc_score(y_true[:, j], y_prob[:, j])
             for j in range(y_true.shape[1])
             if 0 < y_true[:, j].sum() < len(y_true)]
        return float(np.mean(a)) if a else np.nan
    if which == "Acc":
        return float(((y_prob >= thr).astype(int) == y_true).all(1).mean())
    raise ValueError(which)


def paired_bootstrap(yt, pa, pb, which="F1", n_boot=2000, seed=0):
    """
    A ve B'nin AYNI test seti uzerindeki farkini bootstraplar.
    Doner: (fark, lo, hi, p_iki_yonlu)
    """
    rng = np.random.default_rng(seed)
    n = len(yt)
    obs = metric_fn(yt, pa, which) - metric_fn(yt, pb, which)
    diffs = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if yt[i].sum() == 0:
            continue
        d = metric_fn(yt[i], pa[i], which) - metric_fn(yt[i], pb[i], which)
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.asarray(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # iki yonlu p: farkin isaretini degistirme orani (bootstrap yaklasimi)
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return obs * 100, lo * 100, hi * 100, min(p, 1.0)


def available_tags(cls: str) -> list[str]:
    """Tahminleri kayitli olan etiketleri listeler."""
    if not PRED_DIR.exists():
        return []
    return sorted({re.sub(rf"__{cls}__(base|fid|div)\.npz$", "", f.name)
                   for f in PRED_DIR.glob(f"*__{cls}__*.npz")})


def load_pred(tag: str, cls: str, split: str):
    f = PRED_DIR / f"{tag}__{cls}__{split}.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return d["y_true"], d["y_prob"]


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="", help="referans (A1_full) etiketi")
    ap.add_argument("--others", nargs="*", default=[])
    ap.add_argument("--auto", default="", help="veri seti adi; otomatik esle")
    ap.add_argument("--classifier", default="xresnet1d50")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if not PRED_DIR.exists():
        print(f"[!] {PRED_DIR} yok. Once evaluate.py'yi GUNCEL surumle "
              f"calistirin (tahminleri kaydeder).")
        return

    if args.auto:
        tags = sorted({re.sub(rf"__{args.classifier}__(base|fid|div)\.npz$", "",
                              f.name)
                       for f in PRED_DIR.glob(f"*__{args.classifier}__*.npz")
                       if f.name.startswith(args.auto)})
        ref = next((t for t in tags if "leadtcn" in t or "A1" in t), None)
        if ref is None:
            print("Referans bulunamadi (adinda 'leadtcn' veya 'A1' gecmeli).")
            print("Bulunan etiketler:", tags); return
        others = [t for t in tags if t != ref]
    else:
        ref, others = args.ref, args.others
    if not ref or not others:
        print("--ref ve --others verin ya da --auto kullanin."); return

    print("=" * 78)
    print("LEAD-DM | Ablasyon Karsilastirmasi (eslestirilmis bootstrap)")
    print("=" * 78)
    print(f"Referans      : {ref}")
    print(f"Karsilastirma : {len(others)} model")
    print(f"Bootstrap     : {args.n_boot} tekrar\n")

    rows = []
    for split, label in (("fid", "Sadakat"), ("div", "Cesitlilik")):
        r = load_pred(ref, args.classifier, split)
        if r is None:
            print(f"[!] {ref} icin {split} tahminleri yok.")
            print("    Tahminler yalnizca evaluate.py'nin GUNCEL surumuyle")
            print("    yapilan kosularda kaydedilir. Cozum:")
            print(f"      python evaluate.py --dataset <ds> --gen {ref} \\")
            print(f"          --classifiers {args.classifier}")
            av = available_tags(args.classifier)
            if av:
                print("    Tahminleri MEVCUT olan etiketler:")
                for a in av:
                    print(f"      {a}")
            continue
        yt, pr = r
        print(f"--- {label} (R->G / G->R) ---")
        for o in others:
            q = load_pred(o, args.classifier, split)
            if q is None:
                print(f"  {o:<42} tahmin yok, atlandi"); continue
            yt2, po = q
            if not np.array_equal(yt, yt2):
                print(f"  {o:<42} [!] farkli test etiketleri, atlandi"); continue
            for met in ("F1", "AUC"):
                d, lo, hi, p = paired_bootstrap(yt, po, pr, met,
                                                args.n_boot, args.seed)
                sig = "ANLAMLI" if (lo > 0 or hi < 0) else "anlamli degil"
                arrow = "+" if d > 0 else ""
                print(f"  {o.split('__')[1]:<26} {met:<4} "
                      f"{arrow}{d:>6.2f} pt  [{lo:+.2f}, {hi:+.2f}]  "
                      f"p={p:.3f}  {sig}")
                rows.append({"Split": label, "Metric": met,
                             "Model": o, "Referans": ref,
                             "Fark(pt)": d, "GA_alt": lo, "GA_ust": hi,
                             "p": p, "Anlamli": lo > 0 or hi < 0})
        print()

    if not rows:
        print("Karsilastirilacak veri bulunamadi."); return
    df = pd.DataFrame(rows)

    n_sig = int(df["Anlamli"].sum())
    print("=" * 78)
    print(f"SONUC: {len(df)} karsilastirmanin {n_sig} tanesi anlamli")
    if n_sig == 0:
        print("  -> Hicbir mekanizma digerinden ISTATISTIKSEL OLARAK farkli degil.")
        print("     Makalede 'karsilastirilabilir' yazilmali; ustunluk iddiasi")
        print("     bu veriyle savunulamaz. Ayirt edici olarak parametre")
        print("     verimliligi ve yorumlanabilirlik one cikarilabilir.")
    else:
        print("  Anlamli farklar:")
        for _, r in df[df["Anlamli"]].iterrows():
            print(f"    {r['Split']:<11} {r['Metric']:<4} "
                  f"{r['Model'].split('__')[1]:<26} {r['Fark(pt)']:+.2f} pt")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = ExcelReport(RESULTS_DIR / "ABLATION_SIGNIFICANCE.xlsx",
                      meta={"Icerik": "Eslestirilmis bootstrap karsilastirmasi",
                            "Referans": ref,
                            "Siniflandirici": args.classifier,
                            "Bootstrap": args.n_boot,
                            "Not": ("Yalnizca TEST SETI belirsizligi. Egitim "
                                    "tohumu degiskenligi icin coklu tohum "
                                    "gerekir.")})
    rep.add_table("fark_testleri", df,
                  caption=("Pozitif fark = model referanstan IYI. GA 0'i "
                           "icermiyorsa fark anlamlidir."))
    print(f"\n[Excel] {rep.save()}")

    # sekil: fark + GA
    plt = setup_style()
    sub = df[df.Metric == "F1"]
    if len(sub):
        fig, ax = plt.subplots(figsize=(6.6, 0.45 * len(sub) + 1.6))
        y = np.arange(len(sub))
        lab = [f"{r['Model'].split('__')[1]}\n({r['Split']})"
               for _, r in sub.iterrows()]
        ax.errorbar(sub["Fark(pt)"], y,
                    xerr=[sub["Fark(pt)"] - sub["GA_alt"],
                          sub["GA_ust"] - sub["Fark(pt)"]],
                    fmt="o", color=PALETTE[0], capsize=3, ms=4)
        ax.axvline(0, color="k", ls="--", lw=0.9)
        ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=7)
        ax.set_xlabel("Makro F1 farki (pt) — referansa gore")
        ax.grid(True, axis="x")
        ax.set_title("Ablasyon farklari ve %95 guven araliklari", loc="left")
        fig.tight_layout()
        save_figure(fig, RESULTS_DIR / "figures" / "ablation_significance")
        print(f"[Sekil] {RESULTS_DIR/'figures'}")


if __name__ == "__main__":
    main()
