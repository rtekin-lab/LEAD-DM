#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Sonuc toplayici
============================
results/ altindaki tum ara Excel dosyalarini tarar ve MAKALE-HAZIR tek bir
calisma kitabi uretir: results/PAPER_RESULTS.xlsx

Uretilen sayfalar (referans makalenin tablo numaralariyla):
    T3   sinyal + dagilim duzeyi            (METRICS_*.xlsx)
    T3b  normalize oranlar (gercek-gercek)  (METRICS_*.xlsx)
    T4   hastalik teshis duzeyi             (EVAL_*.xlsx)
    T4b  kendi baseline'imiza gore oranlar
    T5   verimlilik                         (GEN_* + DM_*.xlsx)
    T11  kosul dayanikliligi (senaryolar)   (METRICS_*senaryo*)
    T13  fizyolojik tutarlilik              (PHYSIO_*.xlsx)
    REF  referans makale degerleri (elle girilmis, karsilastirma icin)
    LOG  hangi dosyadan ne alindi

Kullanim:
    python collect_results.py
    python collect_results.py --pattern "M1_long"      # sadece bir modeli topla
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from lead_dm.config import RESULTS_DIR
from lead_dm.reporting import ExcelReport, setup_style, save_figure, PALETTE


# ============================================================================
# REFERANS MAKALE DEGERLERI  (CDM-DL-PSI, Wang ve ark. 2026)
# ============================================================================

REFERENCE = {
    "T3": [  # Dataset, Model, RMSE, DTW, SC, CS, DR
        ("PTB-XL", "WaveGAN", 0.2810, 98.23, 0.1678, 0.78, 1.00),
        ("PTB-XL", "Pulse2Pulse", 0.2893, 97.91, 0.1674, 0.83, 1.00),
        ("PTB-XL", "SSSD-ECG", 0.2655, 92.76, 0.1710, 0.91, 1.00),
        ("PTB-XL", "DSAT-ECG", 0.2657, 93.42, 0.1708, 0.89, 1.00),
        ("PTB-XL", "FlowECG", 0.2682, 95.30, 0.1709, 0.96, 1.00),
        ("PTB-XL", "CDM-DL-PSI", 0.2531, 92.84, 0.1711, 1.00, 1.00),
        ("CPSC2018", "SSSD-ECG", 0.2770, 96.25, 0.1713, 0.85, 1.00),
        ("CPSC2018", "DSAT-ECG", 0.2694, 89.21, 0.1714, 0.85, 1.00),
        ("CPSC2018", "FlowECG", 0.2690, 91.57, 0.1715, 0.88, 1.00),
        ("CPSC2018", "CDM-DL-PSI", 0.2499, 83.92, 0.1714, 0.96, 1.00),
    ],
    "T4": [  # Dataset, Classifier, Model, Fid Acc/F1/AUC, Div Acc/F1/AUC
        ("PTB-XL", "Xresnet1d50", "Baseline", 63.33, 72.77, 90.64,
         None, None, None),
        ("PTB-XL", "Xresnet1d50", "SSSD-ECG", 62.88, 64.83, 89.73,
         35.44, 52.60, 78.48),
        ("PTB-XL", "Xresnet1d50", "DSAT-ECG", 58.46, 58.36, 87.99,
         44.86, 55.87, 77.99),
        ("PTB-XL", "Xresnet1d50", "FlowECG", 63.51, 65.49, 91.58,
         44.49, 58.07, 79.90),
        ("PTB-XL", "Xresnet1d50", "CDM-DL-PSI", 63.35, 70.20, 92.03,
         54.57, 66.40, 85.59),
        ("CPSC2018", "Xresnet1d50", "Baseline", 75.84, 79.61, 95.82,
         None, None, None),
        ("CPSC2018", "Xresnet1d50", "SSSD-ECG", 59.97, 61.87, 92.66,
         45.41, 51.44, 79.17),
        ("CPSC2018", "Xresnet1d50", "DSAT-ECG", 72.78, 66.45, 94.02,
         45.71, 51.13, 81.90),
        ("CPSC2018", "Xresnet1d50", "FlowECG", 57.50, 64.90, 94.15,
         41.48, 54.61, 85.33),
        ("CPSC2018", "Xresnet1d50", "CDM-DL-PSI", 69.29, 73.66, 94.25,
         57.93, 64.03, 90.10),
    ],
    "T5": [  # Model, Memory(MB), TrainTime(h), Inference(s)
        ("WaveGAN", 3016, 41, 5.0e-5),
        ("Pulse2Pulse", 1898, 41, 5.0e-5),
        ("SSSD-ECG", 18768, 10.6, 26.0),
        ("DSAT-ECG", 67584, 35, 32.0),
        ("FlowECG", 18768, 10.5, 6.0),
        ("CDM-DL-PSI", 4960, 10, 12.0),
    ],
    "T13": [  # Dataset, Task, Signal, MAE, Acc
        ("PTB-XL", "Age prediction", "Real ECG", 7.31, None),
        ("PTB-XL", "Age prediction", "Generated ECG", 7.70, None),
        ("PTB-XL", "Gender classification", "Real ECG", None, 83.73),
        ("PTB-XL", "Gender classification", "Generated ECG", None, 82.72),
        ("CPSC2018", "Age prediction", "Real ECG", 8.40, None),
        ("CPSC2018", "Age prediction", "Generated ECG", 8.80, None),
        ("CPSC2018", "Gender classification", "Real ECG", None, 82.68),
        ("CPSC2018", "Gender classification", "Generated ECG", None, 79.48),
    ],
}

SCENARIOS = ("only_dl", "only_psi", "perturbed")


# ============================================================================
def read_sheet(path: Path, name: str, header=2) -> pd.DataFrame | None:
    try:
        return pd.read_excel(path, sheet_name=name, header=header)
    except Exception:
        return None


def read_meta(path: Path) -> dict:
    """00_Ozet sayfasindaki anahtar/deger ciftlerini okur."""
    try:
        df = pd.read_excel(path, sheet_name="00_Ozet", header=None)
    except Exception:
        return {}
    out = {}
    for _, r in df.iterrows():
        k = r.iloc[0]
        if isinstance(k, str) and len(r) > 1 and pd.notna(r.iloc[1]):
            out[k.strip()] = r.iloc[1]
    return out


def config_id(tag: dict) -> str:
    """
    Satirlari BENZERSIZ tanimlayan kimlik. Subset ve senaryo dahil edilmezse
    tam set ile alt orneklem satirlari ayirt edilemez (ilk surumdeki hata).
    """
    parts = [tag["dataset"], tag["model"]]
    parts.append(f"g={tag['guidance']}" if tag["guidance"] else "g=yok")
    parts.append(tag["scenario"] or "baseline")
    parts.append(f"n={tag['subset']}" if tag["subset"] else "n=tam")
    return " | ".join(parts)


def parse_tag(stem: str) -> dict:
    """
    'METRICS_ptbxl__M1_long__gall1.25' -> dataset/model/guidance/senaryo
    """
    body = re.sub(r"^(METRICS|EVAL|GEN|PHYSIO|DM|AE|DIAG|SIGDIAG)_", "", stem)
    parts = body.split("__")
    d = {"dataset": parts[0] if parts else "",
         "model": parts[1] if len(parts) > 1 else "",
         "guidance": "", "scenario": "", "subset": ""}
    for p in parts[2:]:
        if p.startswith("gall"):
            d["guidance"] = p[4:]
        elif p.startswith("g") and re.match(r"^g[\d.]+$", p):
            d["guidance"] = p[1:] + " (kosul-basina)"
        elif p in SCENARIOS:
            d["scenario"] = p
        elif p.startswith("n"):
            d["subset"] = p[1:]
    return d


# ============================================================================
def collect(args):
    RES = Path(args.results)
    log = []
    T3, T3b, T4, T5, T11, T13 = [], [], [], [], [], []

    files = sorted(RES.glob("*.xlsx"))
    if args.pattern:
        files = [f for f in files if args.pattern in f.stem]
    print(f"Taranan dosya: {len(files)}")

    for f in files:
        stem = f.stem
        tag = parse_tag(stem)
        kind = stem.split("_")[0]

        # ---------------- METRICS_* -> T3 / T3b / T11 ----------------
        if kind == "METRICS":
            df = read_sheet(f, "T3_sinyal_dagilim")
            if df is not None and len(df):
                for _, r in df.iterrows():
                    row = {"Config": config_id(tag),
                           "Dataset": tag["dataset"], "Model": r.get("Model"),
                           "Guidance": tag["guidance"],
                           "Scenario": tag["scenario"] or "baseline",
                           "Subset": tag["subset"],
                           "RMSE": r.get("RMSE"), "DTW": r.get("DTW"),
                           "SC": r.get("SC"), "CS": r.get("CS"),
                           "DR": r.get("DR"), "Source": stem}
                    (T11 if tag["scenario"] else T3).append(row)
                    if not tag["scenario"] and "GERCEK" in str(r.get("Model")):
                        pass
            dfb = read_sheet(f, "T3b_normalize_oranlar")
            if dfb is not None and len(dfb):
                rec = {"Config": config_id(tag),
                       "Dataset": tag["dataset"], "Guidance": tag["guidance"],
                       "Scenario": tag["scenario"] or "baseline",
                       "Source": stem}
                for _, r in dfb.iterrows():
                    rec[f"{r['Metrik']}_orani"] = r.get("Oran")
                T3b.append(rec)
            log.append((stem, "T3/T11"))

        # ---------------- EVAL_* -> T4 ----------------
        elif kind == "EVAL":
            df = read_sheet(f, "tam_sonuclar")
            if df is None:
                df = read_sheet(f, "T4_teshis")
            if df is not None and len(df):
                for _, r in df.iterrows():
                    T4.append({
                        "Config": config_id(tag),
                        "Dataset": tag["dataset"], "Model": tag["model"],
                        "Guidance": tag["guidance"],
                        "Scenario": tag["scenario"] or "baseline",
                        "Subset": tag["subset"],
                        "Classifier": r.get("Classifier"),
                        "Base_Acc(%)": r.get("Base_Acc(%)"),
                        "Base_F1(%)": r.get("Base_F1(%)"),
                        "Base_AUC(%)": r.get("Base_AUC(%)"),
                        "Fid_Acc(%)": r.get("Fid_Acc(%)"),
                        "Fid_F1(%)": r.get("Fid_F1(%)"),
                        "Fid_AUC(%)": r.get("Fid_AUC(%)"),
                        "Div_Acc(%)": r.get("Div_Acc(%)"),
                        "Div_F1(%)": r.get("Div_F1(%)"),
                        "Div_AUC(%)": r.get("Div_AUC(%)"),
                        "Uyari": r.get("Uyari", ""),
                        "Source": stem})
            log.append((stem, "T4"))

        # ---------------- GEN_* -> T5 (cikarim) ----------------
        elif kind == "GEN":
            df = read_sheet(f, "T5_verimlilik")
            m = read_meta(f)
            if df is not None and len(df):
                r = df.iloc[0]
                T5.append({"Config": config_id(tag),
                           "Dataset": tag["dataset"], "Model": tag["model"],
                           "Guidance": tag["guidance"],
                           "Scenario": tag["scenario"] or "baseline",
                           "Params(M)": r.get("Params(M)"),
                           "Memory(MB)": r.get("Memory(MB)"),
                           "Inference(s)": r.get("Inference(s)"),
                           "Batch(s/ECG)": r.get("Batch_throughput(s/ECG)"),
                           "GenTime(min)": m.get("Toplam sure (dk)"),
                           "Source": stem})
            log.append((stem, "T5"))

        # ---------------- DM_* -> T5 (egitim suresi) ----------------
        elif kind == "DM":
            m = read_meta(f)
            if m:
                T5.append({"Dataset": m.get("Veri seti", tag["dataset"]),
                           "Model": m.get("Model", tag["model"]),
                           "Guidance": "", "Scenario": "egitim",
                           "Params(M)": None,
                           "Memory(MB)": None, "Inference(s)": None,
                           "Batch(s/ECG)": None,
                           "TrainTime(h)": m.get("Egitim suresi (h)"),
                           "Source": stem})
            log.append((stem, "T5-egitim"))

        # ---------------- PHYSIO_* -> T13 ----------------
        elif kind == "PHYSIO":
            df = read_sheet(f, "T13_fizyolojik")
            m = read_meta(f)
            if df is not None and len(df):
                for _, r in df.iterrows():
                    T13.append({"Config": config_id(tag),
                                "Dataset": tag["dataset"],
                                "Guidance": tag["guidance"] or "yok",
                                "Scenario": tag["scenario"] or "baseline",
                                "Subset": tag["subset"] or "tam",
                                "Task": r.get("Task"), "Signal": r.get("Signal"),
                                "MAE": r.get("MAE"), "Acc(%)": r.get("Acc(%)"),
                                "Source": stem})
            log.append((stem, "T13"))

    # ---------------- oranlar ----------------
    dfT4 = pd.DataFrame(T4)
    T4b = pd.DataFrame()
    if len(dfT4):
        T4b = dfT4.copy()
        for pre in ("Fid", "Div"):
            for met in ("Acc", "F1", "AUC"):
                a, b = f"{pre}_{met}(%)", f"Base_{met}(%)"
                if a in T4b and b in T4b:
                    T4b[f"{pre}/Base_{met}"] = (
                        pd.to_numeric(T4b[a], errors="coerce")
                        / pd.to_numeric(T4b[b], errors="coerce"))
        keep = (["Config", "Dataset", "Model", "Guidance", "Scenario",
                 "Subset", "Classifier"]
                + [c for c in T4b.columns if "/Base_" in c]
                + ["Base_F1(%)"])
        T4b = T4b[[c for c in keep if c in T4b.columns]]

    # ---------------- yaz ----------------
    RES.mkdir(parents=True, exist_ok=True)
    rep = ExcelReport(RES / "PAPER_RESULTS.xlsx",
                      meta={"Icerik": "Makale-hazir toplu sonuclar",
                            "Taranan dosya": len(files),
                            "Filtre": args.pattern or "(yok)",
                            "Not": ("REF_* sayfalari referans makalenin "
                                    "yayinlanmis degerleridir; baseline'lar "
                                    "farkli oldugu icin ORANLARLA kiyaslayin")})

    def add(name, rows, cap, hl=None):
        df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows
        if len(df):
            rep.add_table(name, df, caption=cap, highlight=hl)
            print(f"  {name:<22}: {len(df)} satir")
        else:
            print(f"  {name:<22}: (bos)")

    add("T3_sinyal_dagilim", T3,
        "Tablo 3 - Sinyal ve dagilim duzeyi. 'GERCEK-GERCEK' satirlari "
        "metriklerin ulasilabilir sinirini gosterir.",
        {"RMSE": "min", "DTW": "min", "SC": "max", "CS": "max"})
    add("T3b_normalize_oranlar", T3b,
        "Tablo 3b - Gercek-gercek taban cizgisine gore normalize oranlar. "
        "RMSE orani: 0.707=ortalamaya cokme, 1.0=tam cesitlilik.")
    add("T4_teshis", dfT4,
        "Tablo 4 - Sadakat ve cesitlilik. Uyari sutunu dolu satirlar "
        "asiri yonlendirme suphesi tasir.")
    add("T4b_oranlar", T4b,
        "Tablo 4b - Kendi baseline'imiza gore oranlar. Referans makaleyle "
        "ADIL karsilastirma bu sutunlar uzerinden yapilir.")
    add("T5_verimlilik", T5, "Tablo 5 - Verimlilik.",
        {"Memory(MB)": "min", "Inference(s)": "min"})
    add("T11_kosul_dayaniklilik", T11,
        "Tablo 11 - Kosul eksikligi/bozulmasi. CS metrigi en ayirt edici.",
        {"CS": "max", "DR": "max"})
    add("T13_fizyolojik", T13,
        "Tablo 13 - Yas/cinsiyet tutarliligi. Scenario=only_dl satirlari "
        "NEGATIF KONTROLDUR (kosul dusuruldugunde oznitelik okunamamali).")

    # ---------------- saglik / eksiklik kontrolu ----------------
    checks = []
    if len(dfT4):
        # a) sadakat baseline'i asan satirlar
        for _, r in dfT4.iterrows():
            try:
                if float(r["Fid_AUC(%)"]) > float(r["Base_AUC(%)"]) + 1.0:
                    checks.append({"Tur": "UYARI",
                                   "Konu": "Sadakat AUC baseline'i asiyor",
                                   "Detay": r["Config"],
                                   "Aksiyon": "yonlendirme agirligini dusurun"})
            except (TypeError, ValueError):
                pass
        # b) farkli baseline degerleri (epoch farki)
        for ds, g in dfT4.groupby("Dataset"):
            u = sorted(set(round(float(x), 2) for x in g["Base_F1(%)"]
                           if pd.notna(x)))
            if len(u) > 1:
                checks.append({"Tur": "DIKKAT",
                               "Konu": f"{ds}: farkli baseline F1 degerleri",
                               "Detay": str(u),
                               "Aksiyon": ("farkli --epochs ile calisan "
                                           "satirlari BIRBIRIYLE kiyaslamayin")})
        # c) EVAL'i olmayan modeller
        gen_models = {c.split(" | ")[1] for c in
                      [x.get("Config", "") for x in T5] if " | " in c}
        eval_models = set(dfT4["Model"].astype(str))
        for m in sorted(gen_models - eval_models):
            checks.append({"Tur": "EKSIK",
                           "Konu": f"{m} icin EVAL sonucu yok",
                           "Detay": "GEN/METRICS var ama EVAL yok",
                           "Aksiyon": "evaluate.py calistirin"})
    add("KONTROL", checks,
        "Otomatik saglik ve eksiklik kontrolu. Makaleyi yazmadan once "
        "bu sayfadaki her satiri ele alin.")

    # referans sayfalari
    rep.add_table("REF_T3", pd.DataFrame(
        REFERENCE["T3"], columns=["Dataset", "Model", "RMSE", "DTW", "SC",
                                  "CS", "DR"]),
        caption="Referans makale Tablo 3 (yayinlanmis degerler).")
    rep.add_table("REF_T4", pd.DataFrame(
        REFERENCE["T4"], columns=["Dataset", "Classifier", "Model",
                                  "Fid_Acc(%)", "Fid_F1(%)", "Fid_AUC(%)",
                                  "Div_Acc(%)", "Div_F1(%)", "Div_AUC(%)"]),
        caption="Referans makale Tablo 4.")
    rep.add_table("REF_T5", pd.DataFrame(
        REFERENCE["T5"], columns=["Model", "Memory(MB)", "TrainTime(h)",
                                  "Inference(s)"]),
        caption="Referans makale Tablo 5.")
    rep.add_table("REF_T13", pd.DataFrame(
        REFERENCE["T13"], columns=["Dataset", "Task", "Signal", "MAE",
                                   "Acc(%)"]),
        caption="Referans makale Tablo 13.")
    rep.add_table("LOG", pd.DataFrame(log, columns=["Dosya", "Hedef sayfa"]),
                  caption="Hangi ara dosyadan hangi tabloya veri alindi.")

    out = rep.save()
    print(f"\n[Excel] {out}")
    return out


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=str(RESULTS_DIR))
    ap.add_argument("--pattern", default="",
                    help="yalnizca adinda bu gecen dosyalari topla")
    args = ap.parse_args()
    print("=" * 78)
    print("LEAD-DM | Sonuc Toplayici")
    print("=" * 78)
    collect(args)


if __name__ == "__main__":
    main()
