#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EKG Veri Seti Denetim Betigi
=============================
PTB-XL, CPSC2018 (PhysioNet Challenge formati) ve Chapman veri setlerini
ayni kanonik forma getirip karsilastirilabilir istatistikler uretir.

Amac: uc seti birlestirmeden ONCE su sorulari cevaplamak
  - Kanal sayilari ve siralari tutuyor mu?
  - Genlikler ayni birimde mi (mV)? Olcek farki var mi?
  - Duz kanal / NaN / doygunluk (clipping) ne kadar?
  - Kanal basina RMS ve yuzdelik dagilimlari ortusuyor mu?

Kullanim:
    python ecg_dataset_audit.py
    python ecg_dataset_audit.py --n 500 --plots
    python ecg_dataset_audit.py --n -1          # tum kayitlar (yavas)

Gereksinimler: numpy, pandas, scipy, wfdb, openpyxl  (matplotlib opsiyonel)
    pip install numpy pandas scipy wfdb openpyxl matplotlib
"""

from __future__ import annotations

import argparse
import sys
import traceback
import os
from pathlib import Path

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# YAPILANDIRMA
# ----------------------------------------------------------------------------

# Ham veri koku. ECG_DATA_ROOT ortam degiskeni veya --raw argumaniyla
# belirlenir; depoda hicbir yerel makine yolu bulunmaz.
BASE = Path(os.environ.get("ECG_DATA_ROOT", "data/raw"))

PATHS = {
    "PTB-XL":   BASE / "ptb_xl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    "CPSC2018": BASE / "cpsc_2018",
    "Chapman":  BASE / "ECGData_Chapman",
}

OUT_DIR = Path("./audit_output")

TARGET_FS = 100          # hedef ornekleme hizi (makale ile ayni)
TARGET_SEC = 10          # hedef sure
TARGET_LEN = TARGET_FS * TARGET_SEC

# Kanonik 12 derivasyon sirasi
CANON_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"]

# Modelin dogrudan uretecegi 8 kanal (kalan 4'u dogrusal donusumle turetilir)
CANON_8 = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "aVF"]

DEFAULT_N = 300          # veri seti basina ornek kayit sayisi


# ----------------------------------------------------------------------------
# YARDIMCI FONKSIYONLAR
# ----------------------------------------------------------------------------

def normalize_lead_name(name: str) -> str:
    """Derivasyon adlarini kanonik forma cevirir (AVR -> aVR, i -> I, vb.)."""
    s = str(name).strip().replace(" ", "")
    low = s.lower()
    mapping = {
        "i": "I", "ii": "II", "iii": "III",
        "avr": "aVR", "avl": "aVL", "avf": "aVF",
        "v1": "V1", "v2": "V2", "v3": "V3",
        "v4": "V4", "v5": "V5", "v6": "V6",
        "lead i": "I", "leadi": "I",
    }
    return mapping.get(low, s)


def reorder_to_canonical(sig: np.ndarray, lead_names: list[str]):
    """
    (n_samples, n_leads) diziyi kanonik 12-derivasyon sirasina getirir.
    Eksik derivasyon varsa o sutun NaN doldurulur.
    Doner: (yeniden_siralanmis_dizi, eksik_derivasyon_listesi)
    """
    norm = [normalize_lead_name(x) for x in lead_names]
    idx = {nm: i for i, nm in enumerate(norm)}
    out = np.full((sig.shape[0], len(CANON_12)), np.nan, dtype=np.float64)
    missing = []
    for j, lead in enumerate(CANON_12):
        if lead in idx:
            out[:, j] = sig[:, idx[lead]]
        else:
            missing.append(lead)
    return out, missing


def resample_to(sig: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    """Anti-alias filtreli yeniden ornekleme. sig: (n_samples, n_leads)."""
    if fs_in == fs_out:
        return sig
    from scipy.signal import decimate, resample_poly
    from math import gcd

    if fs_in % fs_out == 0:
        q = fs_in // fs_out
        # decimate q>13 icin kararsiz olabilir, kademeli uygula
        out = sig
        while q > 1:
            step = min(q, 10)
            while q % step != 0:
                step -= 1
            out = decimate(out, step, axis=0, ftype="fir", zero_phase=True)
            q //= step
        return out
    g = gcd(fs_in, fs_out)
    return resample_poly(sig, fs_out // g, fs_in // g, axis=0)


def fix_length(sig: np.ndarray, target_len: int) -> np.ndarray:
    """Bastan kirp veya sonuna sifir ekle."""
    n = sig.shape[0]
    if n == target_len:
        return sig
    if n > target_len:
        return sig[:target_len, :]
    pad = np.zeros((target_len - n, sig.shape[1]), dtype=sig.dtype)
    return np.vstack([sig, pad])


def signal_stats(sig12: np.ndarray, extra: dict) -> dict:
    """Kanonik (TARGET_LEN, 12) sinyalden istatistik cikarir. Birim: mV."""
    rec = dict(extra)

    with np.errstate(invalid="ignore"):
        std_per_lead = np.nanstd(sig12, axis=0)
        rms_per_lead = np.sqrt(np.nanmean(sig12 ** 2, axis=0))
        max_abs = np.nanmax(np.abs(sig12), axis=0)

    rec["n_nan"] = int(np.isnan(sig12).sum())
    rec["n_inf"] = int(np.isinf(sig12).sum())
    # duz kanal: standart sapma ~0
    rec["n_flat_leads"] = int(np.sum(std_per_lead < 1e-6))
    rec["flat_lead_names"] = "|".join(
        [CANON_12[i] for i in np.where(std_per_lead < 1e-6)[0]]
    )
    # doygunluk supheli: bir kanalda maksimum degere cok sayida tekrar
    clip_count = 0
    for j in range(sig12.shape[1]):
        col = sig12[:, j]
        col = col[~np.isnan(col)]
        if col.size == 0:
            continue
        mx = np.max(np.abs(col))
        if mx > 0 and np.sum(np.abs(col) >= mx * 0.999) > 10:
            clip_count += 1
    rec["n_clipped_leads"] = clip_count

    rec["p01_all"] = float(np.nanpercentile(sig12, 1))
    rec["p99_all"] = float(np.nanpercentile(sig12, 99))
    rec["max_abs_all"] = float(np.nanmax(max_abs)) if max_abs.size else np.nan

    for j, lead in enumerate(CANON_12):
        rec[f"rms_{lead}"] = float(rms_per_lead[j])
    return rec


# ----------------------------------------------------------------------------
# CPSC2018 : Dx_map.csv tespiti
# ----------------------------------------------------------------------------

def inspect_dx_map(root: Path) -> None:
    """Dx_map.csv'nin dx_mapping_scored.csv ile ayni olup olmadigini kontrol eder."""
    print("\n" + "=" * 78)
    print("Dx_map.csv INCELEMESI")
    print("=" * 78)

    candidates = list(root.glob("**/Dx_map.csv")) + \
                 list(root.glob("**/dx_mapping*.csv"))
    if not candidates:
        print("  [!] Dx_map.csv veya dx_mapping*.csv bulunamadi.")
        return

    for f in candidates:
        print(f"\n  Dosya: {f}")
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"  [!] Okunamadi: {e}")
            continue

        print(f"  Boyut : {df.shape[0]} satir x {df.shape[1]} sutun")
        print(f"  Sutunlar: {list(df.columns)}")

        cols_lower = [c.lower().replace(" ", "").replace("_", "") for c in df.columns]

        has_counts = any(k in cols_lower for k in
                         ["cpsc", "cpscextra", "ptbxl", "georgia",
                          "chapmanshaoxing", "ningbo", "total"])
        has_snomed = any("snomed" in c for c in cols_lower)
        has_abbrev = any("abbrev" in c for c in cols_lower)

        print("\n  TESPIT:")
        if has_counts and has_snomed:
            print("  -> Bu dosya dx_mapping_scored.csv / dx_mapping_unscored.csv ile")
            print("     AYNI yapida. Veri seti basina etiket sayilarini iceriyor.")
            print("     Bize gereken sutunlari dogrudan kullanabilirsiniz.")
        elif has_snomed and has_abbrev:
            print("  -> Bu, SNOMED kodu <-> kisaltma <-> tam ad eslemesi iceren bir")
            print("     sozluk. dx_mapping_scored.csv DEGIL (veri seti basina sayim")
            print("     sutunlari yok), ama etiket cozumlemesi icin YETERLI.")
            print("     Sayim matrisi isterseniz betigin sonundaki fonksiyon")
            print("     bunu .hea dosyalarindan kendisi uretiyor.")
        else:
            print("  -> Yapisi taninamadi. Ilk 5 satiri asagida:")

        print("\n  Ilk 5 satir:")
        print(df.head().to_string(max_colwidth=40))


# ----------------------------------------------------------------------------
# VERI SETI YUKLEYICILERI
# ----------------------------------------------------------------------------

def load_ptbxl(root: Path, n: int, rng: np.random.Generator):
    """PTB-XL: 100 Hz WFDB + ptbxl_database.csv"""
    import wfdb

    db_path = root / "ptbxl_database.csv"
    if not db_path.exists():
        raise FileNotFoundError(f"ptbxl_database.csv bulunamadi: {db_path}")

    db = pd.read_csv(db_path, index_col="ecg_id")
    print(f"  Toplam kayit (ham)      : {len(db)}")
    print(f"  Benzersiz hasta         : {db['patient_id'].nunique()}")

    # Makaledeki yas temizligi
    before = len(db)
    db = db[db["age"].notna()]
    db = db[(db["age"] > 0) & (db["age"] < 120)]   # 300 = >89 gizlilik kodu
    print(f"  Yas filtresi sonrasi    : {len(db)}  (elenen: {before - len(db)})")
    print(f"  Cinsiyet eksik          : {int(db['sex'].isna().sum())}")
    if "strat_fold" in db.columns:
        print(f"  strat_fold mevcut       : EVET (1-10) -> hasta duzeyinde bolme kullanin")

    ids = db.index.to_numpy()
    if n > 0 and n < len(ids):
        ids = rng.choice(ids, size=n, replace=False)

    rows = []
    for eid in ids:
        rel = db.loc[eid, "filename_lr"]        # 'records100/00000/00001_lr'
        rec_path = root / rel
        try:
            rec = wfdb.rdrecord(str(rec_path))
            sig = np.asarray(rec.p_signal, dtype=np.float64)   # mV
            fs_in = int(rec.fs)
            leads = list(rec.sig_name)
            gains = tuple(np.round(np.asarray(rec.adc_gain, dtype=float), 3))
            units = tuple(rec.units)
        except Exception as e:
            rows.append({"dataset": "PTB-XL", "record": str(eid),
                         "error": f"{type(e).__name__}: {e}"})
            continue

        sig12, missing = reorder_to_canonical(sig, leads)
        sig12 = resample_to(sig12, fs_in, TARGET_FS)
        sig12 = fix_length(sig12, TARGET_LEN)

        rows.append(signal_stats(sig12, {
            "dataset": "PTB-XL",
            "record": str(eid),
            "fs_orig": fs_in,
            "n_sig_orig": sig.shape[1],
            "len_sec_orig": round(sig.shape[0] / fs_in, 2),
            "lead_order_orig": "|".join(leads),
            "missing_leads": "|".join(missing),
            "adc_gain": str(gains[:3]) + ("..." if len(gains) > 3 else ""),
            "units": units[0] if units else "?",
            "age": float(db.loc[eid, "age"]),
            "sex": int(db.loc[eid, "sex"]) if pd.notna(db.loc[eid, "sex"]) else -1,
            "error": "",
        }))
    return pd.DataFrame(rows)


def load_cpsc(root: Path, n: int, rng: np.random.Generator):
    """CPSC2018: PhysioNet Challenge formati, g1..g7 alt klasorleri, .hea + .mat"""
    import wfdb

    hea_files = sorted(root.glob("**/*.hea"))
    print(f"  Bulunan .hea dosyasi    : {len(hea_files)}")
    if len(hea_files) == 0:
        raise FileNotFoundError(f"{root} altinda .hea bulunamadi")

    # g1..g7 dagilimi
    groups = {}
    for f in hea_files:
        groups[f.parent.name] = groups.get(f.parent.name, 0) + 1
    print(f"  Klasor dagilimi         : {dict(sorted(groups.items()))}")
    if len(hea_files) > 7000:
        print("  [!] UYARI: 6877'den fazla kayit var. CPSC-Extra karismis olabilir.")
        print("      Makale sadece 6877'lik orijinal seti kullaniyor.")

    files = np.array(hea_files, dtype=object)
    if n > 0 and n < len(files):
        files = rng.choice(files, size=n, replace=False)

    rows = []
    for hea in files:
        base = str(Path(hea).with_suffix(""))
        try:
            rec = wfdb.rdrecord(base)
            sig = np.asarray(rec.p_signal, dtype=np.float64)   # mV
            fs_in = int(rec.fs)
            leads = list(rec.sig_name)
            gains = tuple(np.round(np.asarray(rec.adc_gain, dtype=float), 3))
            units = tuple(rec.units)
            comments = rec.comments or []
        except Exception as e:
            rows.append({"dataset": "CPSC2018", "record": Path(hea).stem,
                         "error": f"{type(e).__name__}: {e}"})
            continue

        age, sex, dx = np.nan, "", ""
        for c in comments:
            cl = c.lower()
            if cl.startswith("age:"):
                try:
                    age = float(c.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif cl.startswith("sex:"):
                sex = c.split(":", 1)[1].strip()
            elif cl.startswith("dx:"):
                dx = c.split(":", 1)[1].strip()

        sig12, missing = reorder_to_canonical(sig, leads)
        sig12 = resample_to(sig12, fs_in, TARGET_FS)
        sig12 = fix_length(sig12, TARGET_LEN)

        rows.append(signal_stats(sig12, {
            "dataset": "CPSC2018",
            "record": Path(hea).stem,
            "fs_orig": fs_in,
            "n_sig_orig": sig.shape[1],
            "len_sec_orig": round(sig.shape[0] / fs_in, 2),
            "lead_order_orig": "|".join(leads),
            "missing_leads": "|".join(missing),
            "adc_gain": str(gains[:3]) + ("..." if len(gains) > 3 else ""),
            "units": units[0] if units else "?",
            "age": age,
            "sex": sex,
            "dx": dx,
            "error": "",
        }))
    return pd.DataFrame(rows)


def load_chapman(root: Path, n: int, rng: np.random.Generator):
    """Chapman: ECGData/ altinda CSV, 500 Hz, 5000x12, birim mikrovolt olabilir."""
    data_dir = root / "ECGData"
    if not data_dir.exists():
        cands = [p for p in root.iterdir() if p.is_dir() and "ecgdata" in p.name.lower()]
        if not cands:
            raise FileNotFoundError(f"ECGData klasoru bulunamadi: {root}")
        data_dir = cands[0]
        print(f"  [i] ECGData yerine kullanilan klasor: {data_dir.name}")

    csvs = sorted(data_dir.glob("*.csv"))
    print(f"  Bulunan CSV dosyasi     : {len(csvs)}")
    if not csvs:
        raise FileNotFoundError(f"{data_dir} altinda CSV yok")

    # Diagnostics.xlsx
    diag = None
    for name in ["Diagnostics.xlsx", "Diagnostics.xls", "Diagnostics.csv"]:
        p = root / name
        if p.exists():
            diag = pd.read_excel(p) if p.suffix.startswith(".xls") else pd.read_csv(p)
            print(f"  Diagnostics dosyasi     : {name}  ({diag.shape[0]} satir)")
            print(f"  Sutunlar                : {list(diag.columns)}")
            break
    if diag is None:
        print("  [!] Diagnostics.xlsx bulunamadi -> yas/cinsiyet/etiket okunamayacak")
    else:
        fcol = next((c for c in diag.columns if "file" in c.lower()), diag.columns[0])
        diag = diag.set_index(diag[fcol].astype(str).str.replace(".csv", "", regex=False))

    # Basligi tespit et
    head = pd.read_csv(csvs[0], nrows=1, header=None)
    first_row_numeric = pd.to_numeric(head.iloc[0], errors="coerce").notna().all()
    header_opt = None if first_row_numeric else 0
    if header_opt == 0:
        hdr = pd.read_csv(csvs[0], nrows=0)
        detected_leads = [normalize_lead_name(c) for c in hdr.columns]
        print(f"  CSV basligi             : VAR -> {detected_leads}")
    else:
        detected_leads = list(CANON_12)
        print(f"  CSV basligi             : YOK -> varsayilan sira kullanilacak:")
        print(f"                            {detected_leads}")
        print("  [!] Bu siralamayi Chapman dokumantasyonuyla DOGRULAYIN.")

    files = np.array(csvs, dtype=object)
    if n > 0 and n < len(files):
        files = rng.choice(files, size=n, replace=False)

    # Birim tespiti icin ham genlikleri once topla
    raw_p99 = []
    rows = []
    for f in files:
        stem = Path(f).stem
        try:
            arr = pd.read_csv(f, header=header_opt).to_numpy(dtype=np.float64)
        except Exception as e:
            rows.append({"dataset": "Chapman", "record": stem,
                         "error": f"{type(e).__name__}: {e}"})
            continue

        if arr.shape[1] != len(detected_leads):
            rows.append({"dataset": "Chapman", "record": stem,
                         "error": f"beklenmeyen sutun sayisi: {arr.shape[1]}"})
            continue

        raw_p99.append(np.nanpercentile(np.abs(arr), 99))
        rows.append({"_raw": arr, "_stem": stem})

    # Olcek karari: mikrovolt mu millivolt mu?
    med_p99 = float(np.nanmedian(raw_p99)) if raw_p99 else np.nan
    if np.isnan(med_p99):
        scale, unit_guess = 1.0, "bilinmiyor"
    elif med_p99 > 50:
        scale, unit_guess = 1e-3, "mikrovolt (uV) -> mV'a cevrildi"
    else:
        scale, unit_guess = 1.0, "millivolt (mV) - cevirme yok"
    print(f"  Ham |sinyal| p99 medyani: {med_p99:.2f}")
    print(f"  Birim tespiti           : {unit_guess}")

    out_rows = []
    for r in rows:
        if "_raw" not in r:
            out_rows.append(r)
            continue
        arr = r["_raw"] * scale
        stem = r["_stem"]

        sig12, missing = reorder_to_canonical(arr, detected_leads)
        sig12 = resample_to(sig12, 500, TARGET_FS)
        sig12 = fix_length(sig12, TARGET_LEN)

        age, sex, rhythm = np.nan, "", ""
        if diag is not None and stem in diag.index:
            row = diag.loc[stem]
            for c in diag.columns:
                cl = str(c).lower()
                if "age" in cl:
                    age = pd.to_numeric(row[c], errors="coerce")
                elif "gender" in cl or cl == "sex":
                    sex = str(row[c])
                elif "rhythm" in cl:
                    rhythm = str(row[c])

        out_rows.append(signal_stats(sig12, {
            "dataset": "Chapman",
            "record": stem,
            "fs_orig": 500,
            "n_sig_orig": arr.shape[1],
            "len_sec_orig": round(arr.shape[0] / 500, 2),
            "lead_order_orig": "|".join(detected_leads),
            "missing_leads": "|".join(missing),
            "adc_gain": f"scale={scale}",
            "units": "mV",
            "age": float(age) if pd.notna(age) else np.nan,
            "sex": sex,
            "dx": rhythm,
            "error": "",
        }))
    return pd.DataFrame(out_rows)


# ----------------------------------------------------------------------------
# RAPORLAMA
# ----------------------------------------------------------------------------

def print_report(df: pd.DataFrame) -> None:
    ok = df[df["error"] == ""] if "error" in df.columns else df

    print("\n" + "=" * 78)
    print("OZET RAPOR")
    print("=" * 78)

    if "error" in df.columns:
        bad = df[df["error"] != ""]
        if len(bad):
            print(f"\n[!] Okunamayan kayit: {len(bad)}")
            print(bad.groupby("dataset")["error"].apply(
                lambda s: s.value_counts().head(3).to_dict()).to_string())

    print("\n--- Format tutarliligi ---")
    fmt = ok.groupby("dataset").agg(
        n=("record", "count"),
        fs_orig=("fs_orig", lambda s: sorted(s.unique())),
        n_sig=("n_sig_orig", lambda s: sorted(s.unique())),
        sure_min=("len_sec_orig", "min"),
        sure_max=("len_sec_orig", "max"),
        birim=("units", lambda s: sorted(set(s))),
    )
    print(fmt.to_string())

    print("\n--- Derivasyon sirasi (veri seti basina benzersiz) ---")
    for ds, g in ok.groupby("dataset"):
        for order, cnt in g["lead_order_orig"].value_counts().items():
            print(f"  {ds:<10} ({cnt:>5} kayit): {order}")
        miss = g[g["missing_leads"] != ""]
        if len(miss):
            print(f"  {ds:<10} [!] eksik derivasyonlu kayit: {len(miss)}")

    print("\n--- Sinyal kalitesi ---")
    q = ok.groupby("dataset").agg(
        duz_kanalli_kayit=("n_flat_leads", lambda s: int((s > 0).sum())),
        ort_duz_kanal=("n_flat_leads", "mean"),
        clip_supheli=("n_clipped_leads", lambda s: int((s > 0).sum())),
        nan_iceren=("n_nan", lambda s: int((s > 0).sum())),
        inf_iceren=("n_inf", lambda s: int((s > 0).sum())),
    ).round(3)
    print(q.to_string())

    print("\n--- Genlik dagilimi (mV) ---")
    a = ok.groupby("dataset").agg(
        p01_medyan=("p01_all", "median"),
        p99_medyan=("p99_all", "median"),
        maxabs_medyan=("max_abs_all", "median"),
        maxabs_p99=("max_abs_all", lambda s: float(np.nanpercentile(s, 99))),
    ).round(4)
    print(a.to_string())
    print("\n  Yorum: p99_medyan degerleri veri setleri arasinda 2 kattan fazla")
    print("         ayrisiyorsa ortak bir global olcek KULLANMAYIN.")

    print("\n--- Kanal basina RMS medyani (mV) ---")
    rms_cols = [f"rms_{l}" for l in CANON_12]
    rms = ok.groupby("dataset")[rms_cols].median().round(4)
    rms.columns = CANON_12
    print(rms.to_string())

    print("\n  Kanallar arasi RMS orani (max/min, veri seti icinde):")
    for ds in rms.index:
        v = rms.loc[ds].to_numpy(dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size:
            print(f"    {ds:<10}: {v.max()/v.min():.2f}x "
                  f"(en yuksek: {rms.loc[ds].idxmax()}, en dusuk: {rms.loc[ds].idxmin()})")

    print("\n  Ayni kanalin veri setleri arasi orani (max/min):")
    for lead in CANON_12:
        v = rms[lead].to_numpy(dtype=float)
        v = v[np.isfinite(v) & (v > 0)]
        if v.size > 1:
            ratio = v.max() / v.min()
            flag = "  <-- DIKKAT" if ratio > 2.0 else ""
            print(f"    {lead:<4}: {ratio:.2f}x{flag}")

    print("\n--- Demografi ---")
    if "age" in ok.columns:
        d = ok.groupby("dataset")["age"].agg(
            ["count", "mean", "std", "min", "max"]).round(2)
        d["eksik"] = ok.groupby("dataset")["age"].apply(lambda s: int(s.isna().sum()))
        print(d.to_string())
    if "sex" in ok.columns:
        print("\n  Cinsiyet degerleri (ham):")
        for ds, g in ok.groupby("dataset"):
            print(f"    {ds:<10}: {g['sex'].value_counts().head(5).to_dict()}")


def make_plots(df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[i] matplotlib yok, grafikler atlandi.")
        return

    ok = df[df["error"] == ""]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) p99 dagilimi
    fig, ax = plt.subplots(figsize=(9, 5))
    for ds, g in ok.groupby("dataset"):
        ax.hist(g["p99_all"].dropna(), bins=60, alpha=0.5, label=ds, density=True)
    ax.set_xlabel("Kayit basina 99. yuzdelik genlik (mV)")
    ax.set_ylabel("Yogunluk")
    ax.set_title("Genlik dagilimi karsilastirmasi")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "amplitude_p99.png", dpi=130)
    plt.close(fig)

    # 2) kanal basina RMS
    rms_cols = [f"rms_{l}" for l in CANON_12]
    datasets = sorted(ok["dataset"].unique())
    fig, axes = plt.subplots(len(datasets), 1, figsize=(11, 3.2 * len(datasets)),
                             sharex=True, squeeze=False)
    for ax, ds in zip(axes[:, 0], datasets):
        g = ok[ok["dataset"] == ds]
        data = [g[c].dropna().to_numpy() for c in rms_cols]
        ax.boxplot(data, labels=CANON_12, showfliers=False)
        ax.set_ylabel("RMS (mV)")
        ax.set_title(f"{ds} - kanal basina RMS")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "rms_per_lead.png", dpi=130)
    plt.close(fig)

    print(f"\n[i] Grafikler kaydedildi: {out_dir}")


# ----------------------------------------------------------------------------
# ETIKET SAYIM MATRISI (CPSC icin bonus)
# ----------------------------------------------------------------------------

def label_matrix_from_headers(root: Path, dataset_name: str) -> pd.DataFrame | None:
    """WFDB .hea dosyalarindaki #Dx: satirlarindan SNOMED sayim tablosu uretir."""
    from collections import Counter
    c = Counter()
    total = 0
    for hea in root.glob("**/*.hea"):
        try:
            txt = hea.read_text(errors="ignore")
        except Exception:
            continue
        for line in txt.splitlines():
            if line.startswith("#Dx:"):
                codes = [x.strip() for x in line.split(":", 1)[1].split(",") if x.strip()]
                c.update(codes)
                total += 1
                break
    if not c:
        return None
    df = pd.DataFrame({"snomed": list(c.keys()), dataset_name: list(c.values())})
    df = df.sort_values(dataset_name, ascending=False).reset_index(drop=True)
    print(f"\n  {dataset_name}: {total} kayitta {len(df)} farkli SNOMED kodu")
    print(df.head(15).to_string(index=False))
    return df


# ----------------------------------------------------------------------------
# ANA AKIS
# ----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="EKG veri seti denetimi")
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help="veri seti basina ornek kayit sayisi (-1 = hepsi)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--plots", action="store_true", help="grafik uret")
    ap.add_argument("--out", type=str, default=str(OUT_DIR))
    ap.add_argument("--only", type=str, default="",
                    help="sadece bu veri setini denetle (PTB-XL/CPSC2018/Chapman)")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("EKG VERI SETI DENETIMI")
    print("=" * 78)
    print(f"Hedef form : {TARGET_FS} Hz, {TARGET_SEC} s ({TARGET_LEN} ornek), mV")
    print(f"Ornek sayisi/veri seti: {'hepsi' if args.n < 0 else args.n}")

    loaders = {
        "PTB-XL": load_ptbxl,
        "CPSC2018": load_cpsc,
        "Chapman": load_chapman,
    }

    frames = []
    for name, fn in loaders.items():
        if args.only and args.only.lower() != name.lower():
            continue
        print("\n" + "-" * 78)
        print(f"[{name}]  {PATHS[name]}")
        print("-" * 78)
        if not PATHS[name].exists():
            print(f"  [!] Yol bulunamadi, atlandi.")
            continue
        try:
            frames.append(fn(PATHS[name], args.n, rng))
        except Exception:
            print(f"  [!] HATA:")
            traceback.print_exc(limit=3)

    if not frames:
        print("\nHicbir veri seti okunamadi.")
        return 1

    df = pd.concat(frames, ignore_index=True)
    if "error" not in df.columns:
        df["error"] = ""
    df["error"] = df["error"].fillna("")

    csv_path = out_dir / "audit_records.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[i] Kayit bazli sonuclar: {csv_path}")

    print_report(df)

    if args.plots:
        make_plots(df, out_dir)

    # CPSC icin ek: Dx_map ve etiket sayimi
    if not args.only or args.only.lower() == "cpsc2018":
        if PATHS["CPSC2018"].exists():
            inspect_dx_map(PATHS["CPSC2018"])
            print("\n" + "=" * 78)
            print("CPSC2018 ETIKET SAYIMI (.hea dosyalarindan)")
            print("=" * 78)
            lm = label_matrix_from_headers(PATHS["CPSC2018"], "CPSC2018")
            if lm is not None:
                lm.to_csv(out_dir / "cpsc2018_label_counts.csv", index=False)
                print(f"\n[i] Tam liste: {out_dir / 'cpsc2018_label_counts.csv'}")

    print("\n" + "=" * 78)
    print("BITTI")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
