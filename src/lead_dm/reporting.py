#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Raporlama
======================
Makaleye dogrudan girecek ciktilar:

  * Excel : her tablo ayri sayfa, basliklar dondurulmus, sutunlar
            otomatik genislikte, sayilar makale formatinda yuvarlanmis.
            En iyi sonuc KALIN, ikinci en iyi ALTI CIZILI (referans
            makalenin tablo konvansiyonu ile ayni).
  * Sekil : 300+ DPI, vektorel PDF + raster PNG, serif tipografi,
            renk korlugune duyarli palet, tutarli stil.

Kullanim
--------
    rep = ExcelReport(RESULTS_DIR / "M1_cpsc2018.xlsx")
    rep.add_table("Tablo3_sinyal_dagilim", df, highlight="min",
                  caption="Sinyal ve dagilim duzeyinde karsilastirma")
    rep.save()

    fig = new_figure(width=7.0, height=4.0)
    ...
    save_figure(fig, RESULTS_DIR / "figures" / "fig3_loss")
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ============================================================================
# GRAFIK STILI
# ============================================================================

DPI = 400                     # >= 300 sarti fazlasiyla saglanir
FIG_FORMATS = ("pdf", "png")  # PDF vektorel (dergi tercihi), PNG onizleme

# Okabe-Ito: renk korlugune duyarli, akademik yayinlarda standart
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#F0E442", "#000000"]

LEAD_COLORS = {
    "I": "#0072B2", "II": "#0072B2", "III": "#0072B2",
    "aVR": "#56B4E9", "aVL": "#56B4E9", "aVF": "#56B4E9",
    "V1": "#D55E00", "V2": "#D55E00", "V3": "#D55E00",
    "V4": "#E69F00", "V5": "#E69F00", "V6": "#E69F00",
}


def setup_style():
    """Matplotlib'i akademik yayin stiline ayarlar. Bir kez cagrilmali."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Nimbus Roman"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.3,
        "lines.linewidth": 1.2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "pdf.fonttype": 42,      # TrueType gomme (dergi sarti)
        "ps.fonttype": 42,
        "axes.prop_cycle": plt.cycler(color=PALETTE),
    })
    return plt


def new_figure(width: float = 7.0, height: float = 4.0, **kw):
    """Genislik inch cinsinden. Tek sutun ~3.5, cift sutun ~7.0."""
    plt = setup_style()
    return plt.figure(figsize=(width, height), **kw)


def save_figure(fig, path_no_ext, formats: Sequence[str] = FIG_FORMATS,
                close: bool = True):
    """
    Ayni sekli birden fazla formatta 300+ DPI kaydeder.

    DIKKAT: Path.with_suffix KULLANILMAZ. Dosya adi nokta iceriyorsa
    (or. 'gen_ptbxl__M2_final__gall1.25_NORM'), with_suffix '.25_NORM'
    kismini uzanti sanip siler ve butun siniflar AYNI dosyaya yazilir.
    Bunun yerine uzanti ada dogrudan eklenir.
    """
    p = Path(path_no_ext)
    p.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        out = p.parent / f"{p.name}.{fmt}"
        fig.savefig(out, format=fmt, dpi=DPI)
        written.append(out)
    if close:
        import matplotlib.pyplot as plt
        plt.close(fig)
    return written


# ============================================================================
# EXCEL RAPORU
# ============================================================================

class ExcelReport:
    """
    Coklu sayfali, bicimlendirilmis Excel raporu.

    highlight secenekleri:
        "min"  -> en dusuk deger en iyi  (RMSE, DTW, MAE)
        "max"  -> en yuksek deger en iyi (Acc, F1, AUC, CS, SC)
        None   -> vurgu yok
        dict   -> sutun basina {"RMSE": "min", "F1(%)": "max", ...}
    """

    def __init__(self, path: Path, meta: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.tables: list[dict] = []
        self.meta = meta or {}

    def add_table(self, sheet: str, df: pd.DataFrame,
                  highlight: str | dict | None = None,
                  caption: str = "",
                  float_fmt: dict | None = None,
                  index: bool = False):
        self.tables.append({
            "sheet": sheet[:31],          # Excel sayfa adi siniri
            "df": df.copy(),
            "highlight": highlight,
            "caption": caption,
            "float_fmt": float_fmt or {},
            "index": index,
        })
        return self

    # ------------------------------------------------------------------
    def save(self):
        from openpyxl import Workbook
        from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
        from openpyxl.utils import get_column_letter

        wb = Workbook()
        wb.remove(wb.active)

        head_font = Font(bold=True, size=10, color="FFFFFF")
        head_fill = PatternFill("solid", fgColor="34495E")
        best_font = Font(bold=True)
        second_font = Font(underline="single")
        cap_font = Font(italic=True, size=9, color="555555")
        thin = Side(style="thin", color="BBBBBB")
        border = Border(bottom=thin)

        # --- Ozet sayfasi ---
        ws = wb.create_sheet("00_Ozet")
        ws["A1"] = "LEAD-DM Deney Raporu"
        ws["A1"].font = Font(bold=True, size=13)
        r = 3
        for k, v in self.meta.items():
            ws.cell(r, 1, str(k)).font = Font(bold=True)
            ws.cell(r, 2, str(v))
            r += 1
        r += 1
        ws.cell(r, 1, "Tablolar").font = Font(bold=True)
        r += 1
        for t in self.tables:
            ws.cell(r, 1, t["sheet"])
            ws.cell(r, 2, t["caption"])
            r += 1
        ws.column_dimensions["A"].width = 28
        ws.column_dimensions["B"].width = 70

        # --- Tablolar ---
        for t in self.tables:
            df, sheet = t["df"], t["sheet"]
            ws = wb.create_sheet(sheet)
            row0 = 1
            if t["caption"]:
                ws.cell(1, 1, t["caption"]).font = cap_font
                row0 = 3

            cols = list(df.columns)
            if t["index"]:
                cols = [df.index.name or ""] + cols

            for j, c in enumerate(cols, start=1):
                cell = ws.cell(row0, j, str(c))
                cell.font = head_font
                cell.fill = head_fill
                cell.alignment = Alignment(horizontal="center",
                                           vertical="center", wrap_text=True)

            # hangi sutunda hangi yon
            hl = t["highlight"]
            hl_map = {}
            if isinstance(hl, dict):
                hl_map = hl
            elif isinstance(hl, str):
                hl_map = {c: hl for c in df.columns
                          if pd.api.types.is_numeric_dtype(df[c])}

            # her sutun icin en iyi / ikinci en iyi satirlari bul
            best_idx, second_idx = {}, {}
            for c, mode in hl_map.items():
                if c not in df.columns:
                    continue
                s = pd.to_numeric(df[c], errors="coerce")
                if s.notna().sum() < 2:
                    continue
                order = s.sort_values(ascending=(mode == "min"))
                order = order.dropna()
                if len(order) >= 1:
                    best_idx[c] = order.index[0]
                if len(order) >= 2:
                    second_idx[c] = order.index[1]

            for i, (ridx, rowdata) in enumerate(df.iterrows()):
                excel_row = row0 + 1 + i
                j = 1
                if t["index"]:
                    ws.cell(excel_row, 1, str(ridx)).font = Font(bold=True)
                    j = 2
                for c in df.columns:
                    v = rowdata[c]
                    if isinstance(v, (np.floating, float)) and not pd.isna(v):
                        v = float(v)
                    if isinstance(v, (np.integer,)):
                        v = int(v)
                    cell = ws.cell(excel_row, j, v)
                    fmt = t["float_fmt"].get(c)
                    if fmt:
                        cell.number_format = fmt
                    elif isinstance(v, float):
                        cell.number_format = "0.0000"
                    cell.alignment = Alignment(horizontal="center")
                    cell.border = border
                    if best_idx.get(c) == ridx:
                        cell.font = best_font
                    elif second_idx.get(c) == ridx:
                        cell.font = second_font
                    j += 1

            # sutun genislikleri
            for j, c in enumerate(cols, start=1):
                vals = [str(c)] + [str(x) for x in
                                   (df.index if (t["index"] and j == 1)
                                    else df[c] if c in df.columns else [])]
                w = max(10, min(34, max(len(v) for v in vals) + 3))
                ws.column_dimensions[get_column_letter(j)].width = w

            ws.freeze_panes = ws.cell(row0 + 1, 2 if t["index"] else 1)

        wb.save(self.path)
        return self.path


# ============================================================================
# HAZIR TABLO ISKELETLERI  (referans makalenin tablolari)
# ============================================================================

TABLE_SPECS = {
    # Tablo 3: sinyal + dagilim duzeyi
    "T3_signal_distribution": {
        "columns": ["Dataset", "Model", "RMSE", "DTW", "SC", "CS", "DR"],
        "highlight": {"RMSE": "min", "DTW": "min", "SC": "max",
                      "CS": "max", "DR": "max"},
        "caption": ("Tablo 3 - Sinyal ve dagilim duzeyinde karsilastirma. "
                    "En iyi kalin, ikinci en iyi alti cizili."),
    },
    # Tablo 4: hastalik teshis duzeyi
    "T4_diagnosis": {
        "columns": ["Dataset", "Classifier", "Model",
                    "Fid_Acc(%)", "Fid_F1(%)", "Fid_AUC(%)",
                    "Div_Acc(%)", "Div_F1(%)", "Div_AUC(%)"],
        "highlight": {c: "max" for c in
                      ["Fid_Acc(%)", "Fid_F1(%)", "Fid_AUC(%)",
                       "Div_Acc(%)", "Div_F1(%)", "Div_AUC(%)"]},
        "caption": ("Tablo 4 - Hastalik teshis duzeyinde sadakat ve "
                    "cesitlilik degerlendirmesi."),
    },
    # Tablo 5: verimlilik
    "T5_efficiency": {
        "columns": ["Model", "Params(M)", "Memory(MB)",
                    "TrainTime(h)", "Inference(s)"],
        "highlight": {"Memory(MB)": "min", "TrainTime(h)": "min",
                      "Inference(s)": "min"},
        "caption": "Tablo 5 - Verimlilik karsilastirmasi.",
    },
    # Tablo 7: ablasyon
    "T7_ablation": {
        "columns": ["Dataset", "Ablation", "RMSE", "DTW", "SC", "CS", "DR",
                    "Fid_Acc(%)", "Fid_F1(%)", "Fid_AUC(%)",
                    "Div_Acc(%)", "Div_F1(%)", "Div_AUC(%)"],
        "highlight": {"RMSE": "min", "DTW": "min", "SC": "max", "CS": "max",
                      **{c: "max" for c in ["Fid_Acc(%)", "Fid_F1(%)",
                                            "Fid_AUC(%)", "Div_Acc(%)",
                                            "Div_F1(%)", "Div_AUC(%)"]}},
        "caption": "Tablo 7 - Bilesen ablasyonu.",
    },
    # Tablo 9: kosul enjeksiyon mekanizmalari
    "T9_conditioning": {
        "columns": ["Dataset", "Mechanism", "RMSE", "DTW", "SC", "CS", "DR",
                    "Fid_Acc(%)", "Fid_F1(%)", "Fid_AUC(%)",
                    "Div_Acc(%)", "Div_F1(%)", "Div_AUC(%)"],
        "highlight": {"RMSE": "min", "DTW": "min", "SC": "max", "CS": "max",
                      **{c: "max" for c in ["Fid_Acc(%)", "Fid_F1(%)",
                                            "Fid_AUC(%)", "Div_Acc(%)",
                                            "Div_F1(%)", "Div_AUC(%)"]}},
        "caption": "Tablo 9 - Kosul enjeksiyon mekanizmalari.",
    },
    # Tablo 11: kosul eksikligi / bozulmasi
    "T11_condition_robustness": {
        "columns": ["Dataset", "Scenario", "RMSE", "DTW", "SC", "CS", "DR",
                    "Fid_Acc(%)", "Fid_F1(%)", "Fid_AUC(%)",
                    "Div_Acc(%)", "Div_F1(%)", "Div_AUC(%)"],
        "highlight": {"RMSE": "min", "DTW": "min", "SC": "max", "CS": "max"},
        "caption": ("Tablo 11 - Kosul bilgisinin eksikligine ve hatasina "
                    "dayaniklilik."),
    },
    # Tablo 13: fizyolojik oznitelik tutarliligi
    "T13_physio_consistency": {
        "columns": ["Dataset", "Task", "Signal", "MAE", "Acc(%)"],
        "highlight": None,
        "caption": ("Tablo 13 - Uretilen sinyallerde yas tahmini ve "
                    "cinsiyet siniflandirma tutarliligi."),
    },
    # Ek: egitim gunlugu
    "TRAIN_log": {
        "columns": ["step", "loss", "eps_mse", "freq", "lr", "time_s"],
        "highlight": None,
        "caption": "Egitim gunlugu.",
    },
}


def empty_table(spec_key: str) -> pd.DataFrame:
    return pd.DataFrame(columns=TABLE_SPECS[spec_key]["columns"])


def add_spec_table(report: ExcelReport, spec_key: str, df: pd.DataFrame,
                   sheet: str | None = None):
    spec = TABLE_SPECS[spec_key]
    return report.add_table(sheet or spec_key, df,
                            highlight=spec["highlight"],
                            caption=spec["caption"])


# ============================================================================
# ORTAK SEKILLER
# ============================================================================

def plot_training_curves(log_df: pd.DataFrame, out_path, title: str = ""):
    """Kayip egrileri: toplam + bilesen terimler."""
    plt = setup_style()
    cols = [c for c in ["loss", "eps_mse", "freq", "interlead", "pde"]
            if c in log_df.columns]
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))

    ax = axes[0]
    ax.plot(log_df["step"], log_df["loss"], color=PALETTE[0], lw=1.0)
    if "val_loss" in log_df.columns:
        v = log_df.dropna(subset=["val_loss"])
        ax.plot(v["step"], v["val_loss"], color=PALETTE[1], lw=1.2,
                marker="o", ms=2.5, label="dogrulama")
        ax.legend()
    ax.set_xlabel("Iterasyon"); ax.set_ylabel("Toplam kayip")
    ax.set_yscale("log"); ax.grid(True)
    ax.set_title("(a) Toplam kayip", loc="left")

    ax = axes[1]
    for i, c in enumerate(cols):
        if c == "loss":
            continue
        ax.plot(log_df["step"], log_df[c], color=PALETTE[i], lw=1.0, label=c)
    ax.set_xlabel("Iterasyon"); ax.set_ylabel("Bilesen kayip")
    ax.set_yscale("log"); ax.grid(True); ax.legend()
    ax.set_title("(b) Kayip bilesenleri", loc="left")

    if title:
        fig.suptitle(title, y=1.03, fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_path)


def plot_ecg_12lead(x12: np.ndarray, out_path, fs: int = 100,
                    title: str = "", second: np.ndarray | None = None,
                    labels=("Uretilen", "Gercek")):
    """
    12 derivasyonu klinik duzende (4 sutun x 3 satir + ritim seridi)
    cizer. second verilirse ust uste bindirir.
    """
    plt = setup_style()
    from .config import CANON_12
    t = np.arange(x12.shape[-1]) / fs

    fig, axes = plt.subplots(3, 4, figsize=(7.2, 4.6), sharex=True)
    order = [["I", "aVR", "V1", "V4"],
             ["II", "aVL", "V2", "V5"],
             ["III", "aVF", "V3", "V6"]]
    for r in range(3):
        for c in range(4):
            name = order[r][c]
            k = CANON_12.index(name)
            ax = axes[r, c]
            if second is not None:
                ax.plot(t, second[k], color="#999999", lw=0.8,
                        label=labels[1] if (r == 0 and c == 0) else None)
            ax.plot(t, x12[k], color=LEAD_COLORS[name], lw=0.9,
                    label=labels[0] if (r == 0 and c == 0) else None)
            ax.text(0.01, 0.86, name, transform=ax.transAxes,
                    fontsize=8, fontweight="bold")
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.grid(True, axis="x", alpha=0.25)
            if r == 2:
                ax.set_xlabel("Zaman (s)")
    if second is not None:
        axes[0, 0].legend(loc="lower left", fontsize=7, ncol=2)
    if title:
        fig.suptitle(title, y=0.99, fontsize=10)
    fig.tight_layout()
    return save_figure(fig, out_path)


def plot_lead_attention(w: np.ndarray, class_names: Sequence[str],
                        lead_names: Sequence[str], out_path,
                        title: str = "Derivasyon ilgi haritasi (LTCM)"):
    """
    LTCM'in ogrendigi derivasyon secicilik haritasi.
    w: (n_classes, n_leads), ortalama 1.0
    Bu, makalenin en yorumlanabilir sekli.
    """
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(0.55 * len(lead_names) + 2.4,
                                    0.42 * len(class_names) + 1.6))
    vmax = float(np.nanmax(np.abs(w - 1.0))) or 1.0
    im = ax.imshow(w, cmap="RdBu_r", vmin=1 - vmax, vmax=1 + vmax,
                   aspect="auto")
    ax.set_xticks(range(len(lead_names)))
    ax.set_xticklabels(lead_names)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    for i in range(w.shape[0]):
        for j in range(w.shape[1]):
            ax.text(j, i, f"{w[i, j]:.2f}", ha="center", va="center",
                    fontsize=6.5,
                    color="white" if abs(w[i, j] - 1) > 0.6 * vmax else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label("Derivasyon agirligi (1.0 = notr)", fontsize=8)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    return save_figure(fig, out_path)


def plot_metric_bars(df: pd.DataFrame, value_col: str, group_col: str,
                     out_path, ylabel: str = "", title: str = "",
                     baseline: float | None = None,
                     baseline_label: str = "Referans"):
    """Model karsilastirmasi icin cubuk grafik."""
    plt = setup_style()
    fig, ax = plt.subplots(figsize=(max(3.5, 0.75 * len(df)), 3.0))
    xs = np.arange(len(df))
    ax.bar(xs, df[value_col].to_numpy(), color=PALETTE[0], width=0.62)
    if baseline is not None:
        ax.axhline(baseline, color=PALETTE[1], ls="--", lw=1.0,
                   label=baseline_label)
        ax.legend()
    ax.set_xticks(xs)
    ax.set_xticklabels(df[group_col].astype(str), rotation=30, ha="right")
    ax.set_ylabel(ylabel or value_col)
    ax.grid(True, axis="y")
    if title:
        ax.set_title(title, loc="left")
    fig.tight_layout()
    return save_figure(fig, out_path)
