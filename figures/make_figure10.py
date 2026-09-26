#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Sekil 10 — nitel karsilastirma
===========================================
Uretilen ve GERCEK (koşul-eslesmis) EKG kayitlarini yan yana gosterir.

Tasarim kararlari
-----------------
* 10 saniyelik bir kaydi 4 cm'lik panele sikistirmak her EKG'yi dikenli
  gosterir. Bu nedenle varsayilan olarak 3 saniyelik bir pencere cizilir
  (--seconds ile degistirilebilir); literaturde yaygin uygulamadir.
* Gercek kayit GRI renkte ARKADA, uretilen kayit renkli ONDE cizilir.
  Ayni kosul ucusune (hastalik, yas, cinsiyet) sahip gercek kayit
  kullanilir; boylece karsilastirma adildir.
* Genlik olcegi HER IKI sinyal icin AYNIDIR. Uretilen sinyallerin genligi
  gercegin ~%70'i oldugundan, bu fark sekilde GORUNUR kalir. Bu bilincli
  bir tercihtir: farki gizleyen ayri olcekleme kullanilmaz.
* --lowpass verilirse HER IKI sinyale de ayni 40 Hz alcak geciren filtre
  uygulanir (klinik EKG'de rutin). Kullanilirsa makale metninde
  BELIRTILMELIDIR.

Kullanim
--------
    python make_figure10.py --dataset ptbxl --gen ptbxl__M2_final__gall1.25
    python make_figure10.py --dataset ptbxl --gen ... --seconds 10
    python make_figure10.py --dataset ptbxl --gen ... --lowpass 40
    python make_figure10.py --dataset ptbxl --gen ... --lang tr
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PREP = Path("prepared")
GEN = Path("generated")
OUT = Path("paper/figures")
DPI = 400
FS = 100
CANON = ("I", "II", "III", "aVR", "aVL", "aVF",
         "V1", "V2", "V3", "V4", "V5", "V6")
CLASSES = {"ptbxl": ["NORM", "MI", "STTC", "CD", "HYP"],
           "cpsc2018": ["NORM", "AF", "I-AVB", "LBBB", "RBBB",
                        "PAC", "PVC", "STD", "STE"],
           "chapman": ["AFIB", "GSVT", "SB", "SR"]}
COL_GEN = "#0072B2"
COL_REAL = "#9A9A9A"


def setup():
    plt.rcParams.update({
        "savefig.dpi": DPI, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 7.5, "axes.linewidth": 0.6,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def lowpass(x, fc=40.0, fs=FS):
    """Sifir fazli Butterworth alcak geciren (klinik EKG rutini)."""
    from scipy.signal import butter, filtfilt
    b, a = butter(4, fc / (fs / 2), btype="low")
    return filtfilt(b, a, x, axis=-1)


def load(kind, dataset, gen_tag):
    import h5py
    d, stem = (PREP, dataset) if kind == "real" else (GEN, gen_tag)
    meta = pd.read_csv(d / f"{stem}_meta.csv")
    labels = np.load(d / f"{stem}_labels.npy")
    return h5py.File(d / f"{stem}.h5", "r"), meta, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl")
    ap.add_argument("--gen", required=True)
    ap.add_argument("--leads", nargs="+", default=["II", "V1", "V4"])
    ap.add_argument("--layout", default="sidebyside",
                    choices=["sidebyside", "overlay"],
                    help="sidebyside: gercek ve uretilen AYRI panellerde "
                         "(onerilen). overlay: ust uste — vuru zamanlamasi "
                         "ortusmedigi icin yaniltici olabilir.")
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--start", type=float, default=0.5)
    ap.add_argument("--lowpass", type=float, default=0.0,
                    help="0 = filtre yok; 40 = her iki sinyale de 40 Hz LPF")
    ap.add_argument("--lang", default="en", choices=["en", "tr"])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    setup()
    rng = np.random.default_rng(args.seed)

    classes = CLASSES[args.dataset]
    fr, mr, lr = load("real", args.dataset, args.gen)
    fg, mg, lg = load("gen", args.dataset, args.gen)

    # gercek kayit indeksini hizli bulmak icin
    rec2row = {r: i for i, r in enumerate(mr["record"].tolist())}

    i0 = int(args.start * FS)
    i1 = i0 + int(args.seconds * FS)
    t = np.arange(i1 - i0) / FS

    nrow = len(classes)
    nL = len(args.leads)
    ncol = nL * 2 if args.layout == "sidebyside" else nL
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.16, 0.98 * nrow),
                             sharex=True,
                             gridspec_kw={"wspace": 0.12} if
                             args.layout == "overlay" else
                             {"width_ratios": [1] * ncol, "wspace": 0.10})
    if nrow == 1:
        axes = axes[None, :]

    picked = []
    for r, cls in enumerate(classes):
        j = classes.index(cls)
        cand = np.where(lg[:, j] > 0)[0]
        if len(cand) == 0:
            picked.append(None)
            continue
        gi = int(rng.choice(cand))
        src = mg.iloc[gi]["source_record"]
        ri = rec2row.get(src)
        xg = np.asarray(fg["signals"][gi], dtype=np.float32)
        xr = (np.asarray(fr["signals"][ri], dtype=np.float32)
              if ri is not None else None)
        if args.lowpass:
            xg = lowpass(xg, args.lowpass)
            if xr is not None:
                xr = lowpass(xr, args.lowpass)
        picked.append((gi, mg.iloc[gi], xg, xr))

        idx = [CANON.index(l) for l in args.leads]
        vals = [xg[idx, i0:i1]]
        if xr is not None:
            vals.append(xr[idx, i0:i1])
        # Tek bir uc tepe paneli kucultmesin diye %99.5 yuzdelik;
        # olcek HER IKI sinyal icin ORTAKTIR, genlik farki gorunur kalir.
        lim = float(np.percentile(np.abs(np.concatenate(vals)), 99.5)) * 1.30

        def style(ax, title=None, ylab=None):
            ax.set_ylim(-lim, lim)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.grid(True, axis="x", lw=0.35, alpha=0.3)
            if title:
                ax.set_title(title, fontsize=7.2, pad=3)
            if ylab:
                ax.set_ylabel(ylab, fontsize=6.6, rotation=0, ha="right",
                              va="center", labelpad=14)

        age = mg.iloc[gi]["age"]
        sex = "M" if int(mg.iloc[gi]["sex"]) == 0 else "F"
        rowlab = f"{cls}\n{sex}, {age:.0f}"

        if args.layout == "sidebyside":
            for c, lead in enumerate(args.leads):
                k = CANON.index(lead)
                axr = axes[r, c]
                if xr is not None:
                    axr.plot(t, xr[k, i0:i1], color=COL_REAL, lw=0.8, zorder=3)
                style(axr, lead if r == 0 else None, rowlab if c == 0 else None)
                axg = axes[r, nL + c]
                axg.plot(t, xg[k, i0:i1], color=COL_GEN, lw=0.85, zorder=3)
                style(axg, lead if r == 0 else None)
        else:
            for c, lead in enumerate(args.leads):
                k = CANON.index(lead)
                ax = axes[r, c]
                if xr is not None:
                    ax.plot(t, xr[k, i0:i1], color=COL_REAL, lw=0.75,
                            label="real" if (r == 0 and c == 0) else None,
                            zorder=2)
                ax.plot(t, xg[k, i0:i1], color=COL_GEN, lw=0.85,
                        label="generated" if (r == 0 and c == 0) else None,
                        zorder=3)
                style(ax, lead if r == 0 else None,
                      rowlab if c == 0 else None)

        if r == nrow - 1:
            for c in range(ncol):
                axes[r, c].set_xlabel("Time (s)" if args.lang == "en"
                                      else "Zaman (s)", fontsize=7)

    # blok basliklari
    if args.layout == "sidebyside":
        fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.115,
                            wspace=0.10, hspace=0.30)
        lblR = "Real" if args.lang == "en" else "Gercek"
        lblG = "Generated" if args.lang == "en" else "Uretilen"
        b0 = axes[0, 0].get_position(); b1 = axes[0, nL - 1].get_position()
        b2 = axes[0, nL].get_position(); b3 = axes[0, ncol - 1].get_position()
        fig.text((b0.x0 + b1.x1) / 2, 0.955, lblR, ha="center",
                 fontsize=8.6, color="#6E6E6E", fontweight="bold")
        fig.text((b2.x0 + b3.x1) / 2, 0.955, lblG, ha="center",
                 fontsize=8.6, color=COL_GEN, fontweight="bold")
        xs = (b1.x1 + b2.x0) / 2
        fig.add_artist(plt.Line2D([xs, xs], [0.09, 0.935], color="#CCCCCC",
                                  lw=0.8, transform=fig.transFigure))

    # olcek cubugu (1 mV) — sag alt panelde
    ax = axes[-1, -1]
    y0 = ax.get_ylim()[0] * 0.82
    ax.plot([t[-1] - 0.45, t[-1] - 0.45], [y0, y0 + 1.0], color="k", lw=1.0)
    ax.text(t[-1] - 0.40, y0 + 0.5, "1 mV", fontsize=6.0, va="center")

    if args.layout == "overlay":
        h, l = axes[0, 0].get_legend_handles_labels()
        if args.lang == "tr" and l and l[0] == "real":
            l = ["gercek", "uretilen"]
        fig.legend(h, l, loc="upper right", ncol=2, frameon=False,
                   fontsize=7.5, bbox_to_anchor=(0.995, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.955))

    OUT.mkdir(parents=True, exist_ok=True)
    d = OUT / args.lang
    d.mkdir(parents=True, exist_ok=True)
    name = "fig10_qualitative"
    if args.layout == "overlay":
        name += "_overlay"
    if args.lowpass:
        name += f"_lp{int(args.lowpass)}"
    for ext in ("pdf", "svg", "png"):
        fig.savefig(d / f"{name}.{ext}", format=ext, dpi=DPI)
    plt.close(fig)
    print(f"  {d / name}.pdf / .svg / .png")

    # secilen kayitlari rapor et (tekrarlanabilirlik)
    print("\n  Secilen kayitlar:")
    for cls, p in zip(classes, picked):
        if p is None:
            print(f"    {cls:<6} — ornek yok"); continue
        gi, m, _, _ = p
        print(f"    {cls:<6} gen #{gi}  <- {m['source_record']}  "
              f"({'M' if int(m['sex'])==0 else 'F'}, {m['age']:.0f})")
    if args.lowpass:
        print(f"\n  [!] Her iki sinyale de {args.lowpass:.0f} Hz alcak geciren"
              " filtre uygulandi.")
        print("      Bu, makale metninde ACIKCA belirtilmelidir.")

    fr.close(); fg.close()


if __name__ == "__main__":
    main()
