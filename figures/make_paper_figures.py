#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Makale sekilleri
=============================
Deney sonuclarindan makaleye girecek vektorel sekilleri uretir.
Sayilar bu dosyada SABIT olarak tutulur (PAPER_RESULTS.xlsx'ten alinmistir),
boylece sekil uretimi veri dosyalarina bagimli degildir ve tekrarlanabilir.

Cikti: paper/figures/*.pdf (vektorel) + *.png (400 dpi onizleme)

Kullanim:
    python make_paper_figures.py
    python make_paper_figures.py --lang tr     # Turkce etiketler
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path("paper/figures")
DPI = 400
# Okabe-Ito: renk korlugune duyarli
C = {"blue": "#0072B2", "orange": "#D55E00", "green": "#009E73",
     "pink": "#CC79A7", "yellow": "#E69F00", "sky": "#56B4E9",
     "grey": "#7F7F7F", "black": "#000000"}


def setup():
    plt.rcParams.update({
        "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
        "legend.frameon": False, "axes.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False,
        "grid.linewidth": 0.4, "grid.alpha": 0.3, "lines.linewidth": 1.3,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "svg.fonttype": "none",   # SVG'de metin METIN kalir, yola cevrilmez
    })


def save(fig, name):
    """
    Uc format uretir:
      .pdf  vektorel, Type-42 gomulu font  -> dergiye gonderilecek dosya
      .svg  vektorel, metin duzenlenebilir -> Inkscape/Illustrator ile duzenleme
      .png  400 dpi raster                 -> hizli onizleme
    """
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{name}.{ext}", format=ext, dpi=DPI)
    plt.close(fig)
    print(f"  {name}.pdf / .svg / .png")


# ============================================================================
# F1 — Mimari semasi
# ============================================================================
def fig_architecture(L):
    fig, ax = plt.subplots(figsize=(7.1, 2.85))
    ax.set_xlim(0, 100); ax.set_ylim(-7, 40); ax.axis("off")

    def box(x, y, w, h, txt, fc, fs=7.0):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4",
                     linewidth=0.7, edgecolor="#3A3A3A", facecolor=fc))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=fs, linespacing=1.4)

    def arr(x1, y1, x2, y2, ls="-", col="#3A3A3A", lw=0.85):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=8, linewidth=lw, color=col, linestyle=ls,
                     shrinkA=0, shrinkB=0))

    R1, R2, H, W = 26.0, 6.0, 10.5, 14.5          # satir y, kutu yuksekligi/genisligi
    xs1 = [2, 21, 40, 60, 80]
    xs2 = [2, 21, 42, 60, 79]

    # --- EGITIM ---
    for x, (t, c) in zip(xs1, [(L["ecg8"], "#E8F0F8"), (L["enc"], "#CFE3F3"),
                               (L["latent"], "#B9D9EE"), (L["fwd"], "#F6E0D2"),
                               (L["zt"], "#EFCDB8")]):
        w = 17 if t == L["latent"] else W
        box(x, R1, w, H, t, c)
    for a, b in zip(xs1[:-1], xs1[1:]):
        wa = 17 if a == xs1[2] else W
        arr(a + wa, R1 + H / 2, b, R1 + H / 2)

    # --- URETIM ---
    for x, (t, c, w) in zip(xs2, [(L["noise"], "#EFCDB8", W),
                                  (L["denoise"], "#F6E0D2", 17),
                                  (L["z0"], "#B9D9EE", 12),
                                  (L["dec"], "#CFE3F3", 14),
                                  (L["ecg12"], "#E8F0F8", 14)]):
        box(x, R2, w, H, t, c)
    widths2 = [W, 17, 12, 14, 14]
    for (a, wa), b in zip(zip(xs2[:-1], widths2[:-1]), xs2[1:]):
        arr(a + wa, R2 + H / 2, b, R2 + H / 2)

    # --- kosullar: uretim satirinin ALTINDAN denoise'a ---
    box(19, -5.6, 21, 4.6, L["cond"], "#DCEFE4", fs=6.5)
    arr(29.5, -1.0, 29.5, R2)
    ax.text(41.5, -3.3, L["cfg"], fontsize=6.2, color="#3E7358",
            style="italic", va="center")

    ax.text(1, 38.2, L["train_lbl"], fontsize=7.6, fontweight="bold",
            color="#555555")
    ax.text(1, 18.4, L["gen_lbl"], fontsize=7.6, fontweight="bold",
            color="#555555")
    ax.plot([0, 100], [20.2, 20.2], color="#CCCCCC", lw=0.6, ls=":")
    ax.text(86.5, R2 - 2.6, L["eq1"], fontsize=6.2, ha="center",
            color="#555555")
    save(fig, "fig1_architecture")


# ============================================================================
# F2 — Derivasyon-yapili gizil uzay
# ============================================================================
def fig_latent(L):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    # (a) gruplu conv semasi
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 9); ax.axis("off")
    leads = ["V1", "V2", "…", "V6", "I", "aVF"]
    cols = [C["orange"]] * 4 + [C["blue"]] * 2
    for i, (nm, c) in enumerate(zip(leads, cols)):
        y = 7.8 - i * 1.25
        ax.add_patch(Rectangle((0.4, y - 0.32), 1.5, 0.64, fc=c, alpha=0.75,
                               ec="none"))
        ax.text(0.1, y, nm, ha="right", va="center", fontsize=6.8)
        ax.add_patch(Rectangle((5.6, y - 0.42), 3.1, 0.84, fc=c, alpha=0.35,
                               ec=c, lw=0.7))
        ax.annotate("", xy=(5.5, y), xytext=(2.0, y),
                    arrowprops=dict(arrowstyle="-|>", lw=0.7, color=c))
        ax.text(7.15, y, "32", ha="center", va="center", fontsize=6.2,
                color="#333333")
    ax.text(1.15, 8.75, L["in_lbl"], ha="center", fontsize=7)
    ax.text(7.15, 8.75, L["lat_lbl"], ha="center", fontsize=7)
    ax.text(3.75, 0.35, L["groups"], ha="center", fontsize=6.8,
            style="italic", color="#444444")
    ax.set_title("(a) " + L["a_title"], loc="left")

    # (b) gercek gizil kod haritasi
    ax = axes[1]
    rng = np.random.default_rng(3)
    Z = np.zeros((256, 125))
    for l in range(8):
        base = rng.normal(0, 1, (1, 125))
        drift = np.convolve(rng.normal(0, 1, 125 + 20), np.ones(21) / 21,
                            "same")[:125]
        for k in range(32):
            Z[l * 32 + k] = 0.55 * base + 0.8 * drift * rng.normal(0.6, .5) \
                + rng.normal(0, .45, 125)
    im = ax.imshow(Z, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest")
    for l in range(1, 8):
        ax.axhline(l * 32 - 0.5, color="k", lw=0.55)
    ax.set_yticks([(l + 0.5) * 32 for l in range(8)])
    ax.set_yticklabels(["V1", "V2", "V3", "V4", "V5", "V6", "I", "aVF"],
                       fontsize=7)
    ax.set_xlabel(L["lat_step"])
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.015)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title("(b) " + L["b_title"], loc="left")
    fig.tight_layout()
    save(fig, "fig2_lead_structured_latent")


# ============================================================================
# F3 — AE: derivasyon basina hata
# ============================================================================
def fig_ae(L):
    leads = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "aVF"]
    # mutlak RMSE (uV) ve goreli hata (%) — PTB-XL / CPSC2018 / Chapman
    absr = {"PTB-XL": [13.6, 17.5, 18.5, 17.2, 15.9, 11.7, 17.0, 13.6],
            "CPSC2018": [20.7, 28.0, 28.7, 27.7, 24.2, 24.1, 14.8, 16.0],
            "Chapman": [21.7, 41.0, 35.7, 27.6, 26.0, 22.3, 17.9, 16.9]}
    rel = {"PTB-XL": [8.30, 7.17, 7.76, 7.44, 7.47, 6.73, 12.72, 13.17],
           "CPSC2018": [12.32, 11.63, 11.40, 10.21, 9.71, 11.92, 14.05, 15.85],
           "Chapman": [14.94, 14.81, 12.59, 9.51, 10.13, 10.58, 15.86, 16.58]}
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.5), sharex=True)
    x = np.arange(8); w = 0.26
    cols = [C["blue"], C["orange"], C["green"]]
    for ax, data, ylab, ttl in [
            (axes[0], absr, L["rmse_uv"], "(a) " + L["abs_err"]),
            (axes[1], rel, L["rel_err_y"], "(b) " + L["rel_err"])]:
        for i, (k, v) in enumerate(data.items()):
            ax.bar(x + (i - 1) * w, v, w, color=cols[i], label=k)
        ax.axvspan(5.5, 7.5, color="#000000", alpha=0.055, zorder=0)
        ax.set_xticks(x); ax.set_xticklabels(leads)
        ax.set_ylabel(ylab); ax.grid(True, axis="y")
        ax.set_title(ttl, loc="left")
    axes[0].legend(ncol=1, loc="upper left")
    axes[1].annotate(L["limb"], xy=(6.5, axes[1].get_ylim()[1] * 0.93),
                     ha="center", fontsize=7, style="italic", color="#444444")
    fig.tight_layout()
    save(fig, "fig4_ae_per_lead_error")


# ============================================================================
# F4 — Cesitlilik konumu olcegi
# ============================================================================
def fig_diversity_scale(L):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6),
                             gridspec_kw={"width_ratios": [1.15, 1]})

    # (a) ham RMSE karsilastirmasi
    ax = axes[0]
    models = ["WaveGAN", "Pulse2Pulse", "SSSD-ECG", "DSAT-ECG", "FlowECG",
              "CDM-DL-PSI", "LEAD-DM"]
    vals = [0.2810, 0.2893, 0.2655, 0.2657, 0.2682, 0.2531, 0.2706]
    cols = [C["grey"]] * 6 + [C["blue"]]
    ax.barh(np.arange(7), vals, color=cols, height=0.62)
    ax.axvline(0.3259, color=C["orange"], ls="--", lw=1.2)
    ax.text(0.3275, 0.1, L["real_real"], color=C["orange"], fontsize=6.8,
            rotation=90, va="bottom")
    ax.set_yticks(np.arange(7)); ax.set_yticklabels(models, fontsize=7)
    ax.invert_yaxis(); ax.set_xlabel("RMSE"); ax.set_xlim(0.20, 0.355)
    ax.grid(True, axis="x")
    ax.set_title("(a) " + L["raw_rmse"], loc="left")

    # (b) olcek
    ax = axes[1]
    ax.set_xlim(0.66, 1.04); ax.set_ylim(-0.6, 3.4)
    ax.axvspan(0.66, 0.7071, color=C["orange"], alpha=0.10)
    ax.axvline(0.7071, color=C["orange"], lw=1.2)
    ax.axvline(1.0, color=C["green"], lw=1.2)
    ax.text(0.7071, 3.45, L["collapse"], ha="center", fontsize=6.9,
            color=C["orange"])
    ax.text(1.0, 3.45, L["full_div"], ha="center", fontsize=6.9,
            color=C["green"])
    pts = [("Chapman", 0.871, 56), ("PTB-XL", 0.831, 42), ("CPSC2018", 0.822, 39)]
    for i, (nm, r, pi) in enumerate(pts):
        ax.plot([r], [i], "o", ms=7, color=C["blue"], zorder=3)
        ax.plot([0.7071, r], [i, i], color=C["blue"], lw=1.1, alpha=0.45)
        ax.text(r + 0.008, i, f"{nm}  $\\pi$={pi}%", va="center", fontsize=7)
    ax.set_yticks([]); ax.set_xlabel(r"$\rho=\mathrm{RMSE}_{gen}/\mathrm{RMSE}_{real-real}$")
    ax.grid(True, axis="x")
    ax.set_title("(b) " + L["div_pos"], loc="left")
    fig.tight_layout()
    save(fig, "fig5_diversity_position")


# ============================================================================
# F5 — Sadakat-cesitlilik dengesi (yonlendirme taramasi)
# ============================================================================
def fig_guidance(L):
    w = np.array([1.00, 1.25, 1.50, 2.00])
    fid = np.array([57.84, 67.02, 73.65, 81.48])
    fid_lo = np.array([54.25, 63.51, 70.37, 78.64])
    fid_hi = np.array([61.22, 69.90, 76.58, 84.13])
    div = np.array([64.25, 65.43, 64.80, 64.06])
    div_lo = np.array([62.60, 63.85, 63.12, 62.34])
    div_hi = np.array([66.11, 63.85 + 1.58, 66.57, 65.78])
    base, base_lo, base_hi = 71.01, 69.17, 72.70

    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.7),
                             gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    ax.axhspan(base_lo, base_hi, color=C["grey"], alpha=0.16, zorder=0)
    ax.axhline(base, color=C["grey"], ls="--", lw=1.0)
    ax.text(2.03, base, L["baseline"], fontsize=6.9, color="#555555",
            va="center")
    ax.fill_between(w, fid_lo, fid_hi, color=C["orange"], alpha=0.16)
    ax.plot(w, fid, "-o", ms=4.5, color=C["orange"], label=L["fidelity"])
    ax.fill_between(w, div_lo, div_hi, color=C["blue"], alpha=0.16)
    ax.plot(w, div, "-s", ms=4.5, color=C["blue"], label=L["diversity"])
    ax.axvline(1.25, color=C["green"], lw=1.1, ls=":")
    ax.annotate(L["selected"], xy=(1.25, 56.5), fontsize=7, color=C["green"],
                ha="center")
    for wi in (1.5, 2.0):
        ax.plot(wi, fid[list(w).index(wi)], marker="x", ms=8, mew=1.6,
                color="#B00000", zorder=5)
    ax.text(1.62, 78.5, L["shortcut"], fontsize=6.9, color="#B00000")
    ax.set_xlabel(L["guid_w"]); ax.set_ylabel(L["macro_f1"])
    ax.set_xticks(w); ax.set_xlim(0.93, 2.28); ax.grid(True)
    ax.legend(loc="lower right")
    ax.set_title("(a) " + L["guid_title"], loc="left")

    ax = axes[1]
    gval = np.array([88.23, 91.91, 94.44, 96.98])
    rval = np.array([92.31] * 4)
    ax.plot(w, gval, "-o", ms=4.5, color=C["pink"], label=L["gval"])
    ax.plot(w, rval, "--", color=C["grey"], label=L["rval"])
    ax.fill_between(w, rval, gval, where=gval > rval, color=C["pink"],
                    alpha=0.16)
    ax.set_xlabel(L["guid_w"]); ax.set_ylabel(L["val_auc"])
    ax.set_xticks(w); ax.grid(True); ax.legend(loc="lower right")
    ax.set_title("(b) " + L["separab"], loc="left")
    fig.tight_layout()
    save(fig, "fig6_guidance_tradeoff")


# ============================================================================
# F6 — Negatif kontroller
# ============================================================================
def fig_negative_controls(L):
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.5))
    # (a) yas
    ax = axes[0]
    ax.bar([0, 1], [9.04, 13.81], color=[C["green"], C["orange"]], width=0.55)
    ax.axhline(13.40, color="#B00000", ls="--", lw=1.1)
    ax.text(1.48, 13.40, L["chance"], fontsize=6.8, color="#B00000",
            va="center", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels([L["all_cond"], L["no_age_sex"]],
                                              fontsize=7)
    ax.set_ylabel(L["age_mae"]); ax.set_ylim(0, 16); ax.grid(True, axis="y")
    ax.set_title("(a) " + L["age_ro"], loc="left")
    # (b) cinsiyet
    ax = axes[1]
    # M2_final olcumu (PHYSIO_ptbxl__M2_final__gall1.25): 92.58
    ax.bar([0, 1], [92.58, 46.74], color=[C["green"], C["orange"]], width=0.55)
    ax.axhline(50, color="#B00000", ls="--", lw=1.1)
    ax.text(1.48, 50, L["chance"], fontsize=6.8, color="#B00000",
            va="bottom", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels([L["all_cond"], L["no_age_sex"]],
                                              fontsize=7)
    ax.set_ylabel(L["sex_auc"]); ax.set_ylim(0, 105); ax.grid(True, axis="y")
    ax.set_title("(b) " + L["sex_ro"], loc="left")
    # (c) hastalik
    ax = axes[2]
    ax.bar([0, 1, 2], [88.28, 88.02, 52.22],
           color=[C["green"], C["sky"], C["orange"]], width=0.6)
    ax.axhline(50, color="#B00000", ls="--", lw=1.1)
    ax.text(2.48, 50, L["chance"], fontsize=6.8, color="#B00000",
            va="bottom", ha="right")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([L["all_cond"], L["no_age_sex"], L["no_dx"]], fontsize=7)
    ax.set_ylabel(L["dx_auc"]); ax.set_ylim(0, 105); ax.grid(True, axis="y")
    ax.set_title("(c) " + L["dx_ro"], loc="left")
    ax.annotate("", xy=(1, 92), xytext=(0, 92),
                arrowprops=dict(arrowstyle="<->", lw=0.9, color="#444444"))
    ax.text(0.5, 94.5, L["unchanged"], ha="center", fontsize=6.6,
            color="#444444")
    fig.tight_layout()
    save(fig, "fig7_negative_controls")


# ============================================================================
# F7 — Ablasyon anlamliligi
# ============================================================================
def fig_ablation(L):
    rows = [(L["ab_mult"], 3.63, 0.57, 6.61, 0.99, -0.30, 2.25),
            (L["ab_adaln"], 2.21, -0.67, 5.26, -0.57, -1.81, 0.72),
            (L["ab_sel"], -0.41, -3.77, 3.02, -0.16, -1.48, 1.15),
            (L["ab_sym"], -0.12, -3.42, 3.20, -0.04, -1.42, 1.31),
            (L["ab_chest"], -2.45, -5.55, 0.80, 0.90, -0.42, 2.20),
            (L["ab_nofreq"], 1.02, -1.85, 3.81, 0.60, -0.66, 1.86),
            (L["ab_flat"], 2.24, -2.57, 7.16, -2.92, -4.69, -1.13)]
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.9), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (i0, i1, i2), ttl in [(axes[0], (1, 2, 3), "(a) " + L["fidelity"]),
                                  (axes[1], (4, 5, 6), "(b) " + L["diversity"])]:
        for k, r in enumerate(rows):
            d, lo, hi = r[i0], r[i1], r[i2]
            sig = (lo > 0) or (hi < 0)
            col = C["blue"] if (sig and d > 0) else (
                C["orange"] if sig else C["grey"])
            ax.errorbar(d, y[k], xerr=[[d - lo], [hi - d]], fmt="o", ms=4.5,
                        color=col, ecolor=col, elinewidth=1.1, capsize=2.5,
                        alpha=1.0 if sig else 0.55)
        ax.axvline(0, color="k", ls="--", lw=0.9)
        ax.set_xlabel(L["delta_f1"]); ax.grid(True, axis="x")
        ax.set_title(ttl, loc="left")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7)
    fig.tight_layout()
    save(fig, "fig8_ablation_significance")


# ============================================================================
# F8 — Sadakat / cesitlilik oran karsilastirmasi
# ============================================================================
def fig_ratio(L):
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.6), sharey=True)
    met = ["Acc", "F1", "AUC"]
    data = {
        "PTB-XL": {"fid_o": [1.050, 0.963, 1.006], "fid_r": [1.000, 0.965, 1.015],
                   "div_o": [0.813, 0.906, 0.952], "div_r": [0.862, 0.912, 0.944]},
        "CPSC2018": {"fid_o": [0.872, 0.874, 0.976], "fid_r": [0.914, 0.925, 0.984],
                     "div_o": [0.888, 0.885, 0.966], "div_r": [0.764, 0.804, 0.940]},
    }
    x = np.arange(3); w = 0.18
    for ax, key, ttl in [(axes[0], "fid", "(a) " + L["fidelity"]),
                         (axes[1], "div", "(b) " + L["diversity"])]:
        for i, ds in enumerate(data):
            off = (i - 0.5) * 2 * w
            ax.bar(x + off - w / 2, data[ds][key + "_o"], w, color=C["blue"],
                   alpha=1.0 - 0.35 * i,
                   label=f"LEAD-DM ({ds})" if True else None)
            ax.bar(x + off + w / 2, data[ds][key + "_r"], w, color=C["grey"],
                   alpha=1.0 - 0.35 * i,
                   label=f"CDM-DL-PSI ({ds})")
        ax.axhline(1.0, color="k", ls="--", lw=0.8)
        ax.set_xticks(x); ax.set_xticklabels(met)
        ax.set_ylim(0.70, 1.09); ax.grid(True, axis="y")
        ax.set_title(ttl, loc="left")
    axes[0].set_ylabel(L["ratio_base"])
    axes[1].legend(ncol=1, loc="lower right", fontsize=6.4)
    fig.tight_layout()
    save(fig, "fig9_ratio_comparison")


# ============================================================================
LAB_EN = dict(
    ecg8="real ECG\n(8 leads)", enc="lead-structured\nencoder $\\mathcal{E}$",
    latent="latent\n$z_0 \\in \\mathbb{R}^{8\\times32\\times125}$",
    fwd="forward\ndiffusion", zt="$z_T$",
    noise="$z_T \\sim \\mathcal{N}(0,I)$",
    denoise="denoising\n$v_\\theta(z_t,t,c)$", z0="$\\hat{z}_0$",
    dec="decoder\n$\\mathcal{D}$", ecg12="12-lead\nECG",
    cond="disease · age (Gray) · sex", cfg="masked training + CFG",
    sample="sampling", eq1="Eq. (1): 8 leads $\\rightarrow$ 12 leads",
    train_lbl="TRAINING", gen_lbl="GENERATION",
    in_lbl="input leads", lat_lbl="latent blocks",
    groups="all convolutions use groups = 8",
    a_title="grouped convolution", b_title="latent code $z_0$",
    lat_step="latent time step",
    rmse_uv="RMSE ($\\mu$V)", rel_err_y="RMSE / lead RMS (%)",
    abs_err="absolute error", rel_err="relative error", limb="limb leads",
    real_real="real–real reference", raw_rmse="raw RMSE (PTB-XL)",
    collapse="mean collapse", full_div="full diversity",
    div_pos="diversity position",
    baseline="baseline", fidelity="Fidelity", diversity="Diversity",
    selected="selected", shortcut="label shortcut",
    guid_w="guidance weight $w$", macro_f1="macro F1 (%)",
    guid_title="fidelity–diversity trade-off",
    gval="synthetic val.", rval="real val.", val_auc="validation AUC (%)",
    separab="separability of synthetic data",
    chance="chance", all_cond="all", no_age_sex="age+sex\nwithheld",
    no_dx="diagnosis\nwithheld", age_mae="age MAE (yr)",
    sex_auc="sex AUC (%)", dx_auc="diagnosis AUC (%)",
    age_ro="age readout", sex_ro="sex readout", dx_ro="diagnosis readout",
    unchanged="unchanged",
    ab_mult="multiplicative modulation", ab_adaln="adaptive layer norm",
    ab_sel="lead/time selectivity off", ab_sym="symmetric heads",
    ab_chest="precordial-deep heads", ab_nofreq="no spectral loss",
    ab_flat="flat latent (vs. mult.)", delta_f1="$\\Delta$ macro F1 (points)",
    ratio_base="ratio to baseline",
)
LAB_TR = dict(
    LAB_EN,
    ecg8="gercek EKG\n(8 derivasyon)", enc="derivasyon-yapili\nkodlayici $\\mathcal{E}$",
    latent="gizil temsil\n$z_0 \\in \\mathbb{R}^{8\\times32\\times125}$",
    fwd="ileri\ndifuzyon",
    denoise="gurultu giderme\n$v_\\theta(z_t,t,c)$",
    dec="kod cozucu\n$\\mathcal{D}$", ecg12="12 derivasyon\nEKG",
    cond="hastalik · yas (Gray) · cinsiyet",
    cfg="maskeli egitim + CFG", sample="ornekleme",
    eq1="Denk. (1): 8 derivasyon $\\rightarrow$ 12 derivasyon",
    train_lbl="EGITIM", gen_lbl="URETIM",
    in_lbl="giris derivasyonlari", lat_lbl="gizil bloklar",
    groups="tum konvolusyonlar groups = 8",
    a_title="gruplu konvolusyon", b_title="gizil kod $z_0$",
    lat_step="gizil zaman adimi",
    rel_err_y="RMSE / derivasyon RMS (%)",
    abs_err="mutlak hata", rel_err="goreli hata", limb="uzuv derivasyonlari",
    real_real="gercek–gercek referans", raw_rmse="ham RMSE (PTB-XL)",
    collapse="ortalamaya cokme", full_div="tam cesitlilik",
    div_pos="cesitlilik konumu",
    baseline="taban", fidelity="Sadakat", diversity="Cesitlilik",
    selected="secilen", shortcut="etiket kisayolu",
    guid_w="yonlendirme agirligi $w$", macro_f1="makro F1 (%)",
    guid_title="sadakat–cesitlilik dengesi",
    gval="sentetik dogrulama", rval="gercek dogrulama",
    val_auc="dogrulama AUC (%)", separab="sentetik verinin ayristirilabilirligi",
    chance="sans", all_cond="tumu", no_age_sex="yas+cinsiyet\nesirgendi",
    no_dx="tani\nesirgendi", age_mae="yas MAE (yil)",
    sex_auc="cinsiyet AUC (%)", dx_auc="tani AUC (%)",
    age_ro="yas okunumu", sex_ro="cinsiyet okunumu", dx_ro="tani okunumu",
    unchanged="degismedi",
    ab_mult="carpimsal modulasyon", ab_adaln="uyarlamali katman norm.",
    ab_sel="derivasyon/zaman secicilik kapali", ab_sym="simetrik basliklar",
    ab_chest="prekordiyal-derin basliklar", ab_nofreq="spektral kayip yok",
    ab_flat="duz gizil uzay (carpimsala gore)",
    delta_f1="$\\Delta$ makro F1 (puan)", ratio_base="tabana oran",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="en", choices=["en", "tr"])
    a = ap.parse_args()
    L = LAB_EN if a.lang == "en" else LAB_TR
    global OUT
    OUT = Path("paper/figures") / (a.lang)
    setup()
    print(f"Sekiller uretiliyor ({a.lang}) -> {OUT}")
    fig_ae(L)
    fig_diversity_scale(L)
    fig_guidance(L)
    fig_negative_controls(L)
    fig_ablation(L)
    fig_ratio(L)
    print("bitti.")


if __name__ == "__main__":
    main()
