#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
EKG On Isleme Hatti  (PTB-XL / CPSC2018 / Chapman  ->  HDF5)
=============================================================

Denetim betiginin bulgulari dogrultusunda uc veri setini tek tip forma
getirip diske yazar. Her veri seti icin bir .h5 + bir .csv uretilir.

CIKTI FORMATI
-------------
<out>/ptbxl.h5      : signals (N,12,1000) float32, mV, 100 Hz, kanonik sira
<out>/ptbxl_meta.csv: satir sirasi signals ile AYNI
   sutunlar: idx, record, parent_record, window_idx, age, sex, split,
             labels (| ile ayrilmis), n_labels, max_abs, kept
<out>/ptbxl_labels.npy : (N, C) uint8 cok-sicak etiket matrisi
(ayni sekilde cpsc2018.* ve chapman.*)

UYGULANAN ISLEMLER
------------------
 1. Kanonik derivasyon sirasi          : I,II,III,aVR,aVL,aVF,V1..V6
 2. Yeniden ornekleme (anti-alias)     : -> 100 Hz
 3. Sabit uzunluk                      : 10 s = 1000 ornek
 4. Birim                              : mV  (Chapman uV -> mV)
 5. Yas temizligi                      : age<=0 veya >=120 elenir (PTB-XL 300 kodu)
 6. Cinsiyet harmonizasyonu            : 0=erkek, 1=kadin
 7. Aykiri deger filtresi              : max|x| > MAX_ABS_MV elenir
 8. CPSC uzun kayit stratejisi         : crop | windows
 9. Bolme                              : PTB-XL strat_fold, digerleri kayit duzeyi
10. Etiketler                          : PTB-XL 5 superclass / CPSC 9 sinif /
                                          Chapman 4 ritim grubu

NORMALIZASYON UYGULANMAZ. Denetim, setler arasi RMS farkinin <1.3x oldugunu
gosterdi; olcekleme kararini model tarafina birakmak icin ham mV saklanir.

KULLANIM
--------
    python prepare_data.py                     # hepsi, varsayilan ayarlar
    python prepare_data.py --only chapman
    python prepare_data.py --cpsc-strategy windows
    python prepare_data.py --limit 200         # hizli deneme
    python prepare_data.py --verify            # sadece mevcut cikti dogrulama

Gereksinimler: numpy pandas scipy wfdb h5py openpyxl tqdm
"""

from __future__ import annotations

import argparse
import ast
import sys
import traceback
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:  # tqdm yoksa sessiz gec
    def tqdm(x, **kw):
        return x

# ============================================================================
# YAPILANDIRMA
# ============================================================================

# Ham veri koku. ECG_DATA_ROOT ortam degiskeni veya --raw argumaniyla
# belirlenir; depoda hicbir yerel makine yolu bulunmaz.
BASE = Path(os.environ.get("ECG_DATA_ROOT", "data/raw"))

PATHS = {
    "ptbxl":    BASE / "ptb_xl" / "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    "cpsc2018": BASE / "cpsc_2018",
    "chapman":  BASE / "ECGData_Chapman",
}

OUT_DIR = Path("./prepared")

TARGET_FS = 100
TARGET_SEC = 10
TARGET_LEN = TARGET_FS * TARGET_SEC          # 1000

CANON_12 = ["I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6"]

# Modelin dogrudan uretecegi 8 kanal; kalan 4'u dogrusal donusumle turetilir.
GEN_8 = ["V1", "V2", "V3", "V4", "V5", "V6", "I", "aVF"]
GEN_8_IDX = [CANON_12.index(l) for l in GEN_8]

MAX_ABS_MV = 8.0        # bunun ustu artefakt sayilir (denetim: %1'den az kayip)
AGE_MIN, AGE_MAX = 0, 120
SEED = 42

# --- Etiket semalari -------------------------------------------------------

PTBXL_CLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]

CPSC_SNOMED = {
    "426783006": "NORM",    # sinus rhythm
    "164889003": "AF",      # atrial fibrillation
    "270492004": "I-AVB",   # first degree AV block
    "164909002": "LBBB",
    "59118001":  "RBBB",
    "284470004": "PAC",     # premature atrial contraction (makalede PAV)
    "164884008": "PVC",     # ventricular ectopic beats
    "429622005": "STD",     # ST depression
    "164931005": "STE",     # ST elevation
    # olasi es kodlar
    "63593006":  "PAC",
    "427172004": "PVC",
    "713427006": "RBBB",    # complete RBBB
    "733534002": "LBBB",    # complete LBBB
}
CPSC_CLASSES = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]

CHAPMAN_RHYTHM_GROUP = {
    "AFIB": "AFIB", "AF": "AFIB",
    "SVT": "GSVT", "AT": "GSVT", "SAAWR": "GSVT",
    "ST": "GSVT", "AVNRT": "GSVT", "AVRT": "GSVT",
    "SB": "SB",
    "SR": "SR", "SI": "SR",
}
CHAPMAN_CLASSES = ["AFIB", "GSVT", "SB", "SR"]


# ============================================================================
# ORTAK YARDIMCILAR
# ============================================================================

def normalize_lead_name(name) -> str:
    s = str(name).strip().replace(" ", "").lower()
    m = {"i": "I", "ii": "II", "iii": "III",
         "avr": "aVR", "avl": "aVL", "avf": "aVF",
         "v1": "V1", "v2": "V2", "v3": "V3",
         "v4": "V4", "v5": "V5", "v6": "V6"}
    return m.get(s, str(name).strip())


def reorder_to_canonical(sig: np.ndarray, lead_names):
    """(n_samples, n_leads) -> (n_samples, 12) kanonik sira. Eksik -> NaN."""
    idx = {normalize_lead_name(n): i for i, n in enumerate(lead_names)}
    out = np.full((sig.shape[0], 12), np.nan, dtype=np.float64)
    missing = []
    for j, lead in enumerate(CANON_12):
        if lead in idx:
            out[:, j] = sig[:, idx[lead]]
        else:
            missing.append(lead)
    return out, missing


def resample_to(sig: np.ndarray, fs_in: int, fs_out: int) -> np.ndarray:
    if fs_in == fs_out:
        return sig
    from math import gcd
    from scipy.signal import decimate, resample_poly
    if fs_in % fs_out == 0:
        q = fs_in // fs_out
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


def to_windows(sig: np.ndarray, win: int, strategy: str):
    """
    (n, 12) -> liste[(win, 12)]
      crop    : sadece ilk pencere (kisa ise sifir doldur)
      windows : ortusmeyen tum tam pencereler (kalan artik atilir)
    """
    n = sig.shape[0]
    if n < win:
        pad = np.zeros((win - n, sig.shape[1]), dtype=sig.dtype)
        return [np.vstack([sig, pad])]
    if strategy == "crop":
        return [sig[:win]]
    k = n // win
    return [sig[i * win:(i + 1) * win] for i in range(k)]


def harmonize_sex(v) -> int:
    s = str(v).strip().upper()
    if s in ("0", "M", "MALE"):
        return 0
    if s in ("1", "F", "FEMALE"):
        return 1
    return -1


def valid_age(a) -> bool:
    return pd.notna(a) and AGE_MIN < float(a) < AGE_MAX


def multihot(label_list, classes):
    v = np.zeros(len(classes), dtype=np.uint8)
    for lb in label_list:
        if lb in classes:
            v[classes.index(lb)] = 1
    return v


def record_level_split(keys, rng, ratios=(0.8, 0.1, 0.1)):
    """Kayit duzeyinde rastgele bolme. keys: benzersiz ust kayit kimlikleri."""
    keys = np.asarray(sorted(set(keys)))
    rng.shuffle(keys)
    n = len(keys)
    n_tr = int(round(n * ratios[0]))
    n_va = int(round(n * ratios[1]))
    out = {}
    for k in keys[:n_tr]:
        out[k] = "train"
    for k in keys[n_tr:n_tr + n_va]:
        out[k] = "val"
    for k in keys[n_tr + n_va:]:
        out[k] = "test"
    return out


# ============================================================================
# HDF5 YAZICI
# ============================================================================

class H5Writer:
    """Buyuyebilen, parcali (chunked) HDF5 yazici."""

    def __init__(self, path: Path, n_leads=12, length=TARGET_LEN, batch=256):
        import h5py
        self.h5 = h5py.File(path, "w")
        self.ds = self.h5.create_dataset(
            "signals", shape=(0, n_leads, length),
            maxshape=(None, n_leads, length),
            dtype="float32", chunks=(min(batch, 64), n_leads, length),
            compression="gzip", compression_opts=4,
        )
        self._buf = []
        self._batch = batch
        self.n = 0

    def add(self, sig12: np.ndarray):
        """sig12: (length, 12) -> (12, length) olarak saklanir."""
        self._buf.append(sig12.T.astype(np.float32))
        if len(self._buf) >= self._batch:
            self.flush()

    def flush(self):
        if not self._buf:
            return
        arr = np.stack(self._buf, axis=0)
        self.ds.resize(self.n + arr.shape[0], axis=0)
        self.ds[self.n:self.n + arr.shape[0]] = arr
        self.n += arr.shape[0]
        self._buf = []

    def close(self, attrs: dict):
        self.flush()
        for k, v in attrs.items():
            self.h5.attrs[k] = v
        self.h5.close()


# ============================================================================
# PTB-XL
# ============================================================================

def prepare_ptbxl(root: Path, out: Path, args, rng):
    import wfdb

    db = pd.read_csv(root / "ptbxl_database.csv", index_col="ecg_id")
    scp = pd.read_csv(root / "scp_statements.csv", index_col=0)
    scp = scp[scp.diagnostic == 1]

    n0 = len(db)
    db = db[db["age"].apply(valid_age)]
    print(f"  yas filtresi        : {n0} -> {len(db)}  (elenen {n0 - len(db)})")

    def to_superclass(scp_str):
        try:
            d = ast.literal_eval(scp_str)
        except Exception:
            return []
        return sorted({scp.loc[k, "diagnostic_class"] for k in d
                       if k in scp.index and pd.notna(scp.loc[k, "diagnostic_class"])})

    db["labels"] = db["scp_codes"].apply(to_superclass)
    n1 = len(db)
    db = db[db["labels"].apply(len) > 0]
    print(f"  etiketsiz elendi    : {n1} -> {len(db)}")

    # Resmi katmanli fold: 1-8 train, 9 val, 10 test  (hasta duzeyinde)
    if "strat_fold" in db.columns:
        split_of = db["strat_fold"].apply(
            lambda f: "train" if f <= 8 else ("val" if f == 9 else "test"))
        print("  bolme               : strat_fold (1-8/9/10, hasta duzeyinde)")
    else:
        sp = record_level_split(db.index.tolist(), rng)
        split_of = db.index.to_series().map(sp)
        print("  bolme               : rastgele (strat_fold yok)")

    ids = db.index.to_numpy()
    if args.limit > 0:
        ids = ids[:args.limit]

    w = H5Writer(out / "ptbxl.h5")
    rows, labs = [], []
    dropped = {"read": 0, "outlier": 0, "missing_lead": 0}

    for eid in tqdm(ids, desc="  PTB-XL", ncols=78):
        try:
            rec = wfdb.rdrecord(str(root / db.loc[eid, "filename_lr"]))
            sig = np.asarray(rec.p_signal, dtype=np.float64)
            fs = int(rec.fs)
            leads = list(rec.sig_name)
        except Exception:
            dropped["read"] += 1
            continue

        s12, missing = reorder_to_canonical(sig, leads)
        if missing:
            dropped["missing_lead"] += 1
            continue
        s12 = resample_to(s12, fs, TARGET_FS)

        for wi, win in enumerate(to_windows(s12, TARGET_LEN, "crop")):
            if not np.isfinite(win).all():
                dropped["read"] += 1
                continue
            ma = float(np.max(np.abs(win)))
            if ma > MAX_ABS_MV:
                dropped["outlier"] += 1
                continue
            w.add(win)
            lb = db.loc[eid, "labels"]
            rows.append({
                "idx": len(rows),
                "record": f"{eid}_w{wi}", "parent_record": str(eid),
                "window_idx": wi,
                "age": float(db.loc[eid, "age"]),
                "sex": harmonize_sex(db.loc[eid, "sex"]),
                "split": split_of.loc[eid],
                "labels": "|".join(lb), "n_labels": len(lb),
                "max_abs": round(ma, 4),
            })
            labs.append(multihot(lb, PTBXL_CLASSES))

    _finish("ptbxl", w, rows, labs, PTBXL_CLASSES, out, dropped)


# ============================================================================
# CPSC2018
# ============================================================================

def prepare_cpsc(root: Path, out: Path, args, rng):
    import wfdb

    heas = sorted(root.glob("**/*.hea"))
    print(f"  bulunan .hea        : {len(heas)}")
    if len(heas) > 7000:
        print("  [!] 6877'den fazla kayit -> CPSC-Extra karismis olabilir.")
    if args.limit > 0:
        heas = heas[:args.limit]

    w = H5Writer(out / "cpsc2018.h5")
    rows, labs = [], []
    dropped = {"read": 0, "outlier": 0, "missing_lead": 0,
               "bad_age": 0, "no_label": 0}
    parents = []

    for hea in tqdm(heas, desc="  CPSC2018", ncols=78):
        try:
            rec = wfdb.rdrecord(str(Path(hea).with_suffix("")))
            sig = np.asarray(rec.p_signal, dtype=np.float64)
            fs = int(rec.fs)
            leads = list(rec.sig_name)
            comments = rec.comments or []
        except Exception:
            dropped["read"] += 1
            continue

        age, sex_raw, dx = np.nan, "", ""
        for c in comments:
            cl = c.lower()
            if cl.startswith("age:"):
                try:
                    age = float(c.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif cl.startswith("sex:"):
                sex_raw = c.split(":", 1)[1].strip()
            elif cl.startswith("dx:"):
                dx = c.split(":", 1)[1].strip()

        if not valid_age(age):
            dropped["bad_age"] += 1
            continue

        codes = [x.strip() for x in dx.split(",") if x.strip()]
        lb = sorted({CPSC_SNOMED[c] for c in codes if c in CPSC_SNOMED})
        if not lb:
            dropped["no_label"] += 1
            continue

        s12, missing = reorder_to_canonical(sig, leads)
        if missing:
            dropped["missing_lead"] += 1
            continue
        s12 = resample_to(s12, fs, TARGET_FS)

        stem = Path(hea).stem
        for wi, win in enumerate(to_windows(s12, TARGET_LEN, args.cpsc_strategy)):
            if not np.isfinite(win).all():
                dropped["read"] += 1
                continue
            ma = float(np.max(np.abs(win)))
            if ma > MAX_ABS_MV:
                dropped["outlier"] += 1
                continue
            w.add(win)
            rows.append({
                "idx": len(rows),
                "record": f"{stem}_w{wi}", "parent_record": stem,
                "window_idx": wi,
                "age": age, "sex": harmonize_sex(sex_raw), "split": "",
                "labels": "|".join(lb), "n_labels": len(lb),
                "max_abs": round(ma, 4),
            })
            labs.append(multihot(lb, CPSC_CLASSES))
            parents.append(stem)

    # Bolme: ust kayit duzeyinde -> ayni kaydin pencereleri hep ayni tarafta
    sp = record_level_split(parents, rng)
    for r in rows:
        r["split"] = sp[r["parent_record"]]

    _finish("cpsc2018", w, rows, labs, CPSC_CLASSES, out, dropped)


# ============================================================================
# CHAPMAN
# ============================================================================

def prepare_chapman(root: Path, out: Path, args, rng):
    data_dir = root / "ECGData"
    if not data_dir.exists():
        cands = [p for p in root.iterdir() if p.is_dir() and "ecgdata" in p.name.lower()]
        if not cands:
            raise FileNotFoundError(f"ECGData bulunamadi: {root}")
        data_dir = cands[0]
        print(f"  [i] klasor          : {data_dir.name}")

    diag_path = next((root / n for n in
                      ["Diagnostics.xlsx", "Diagnostics.xls", "Diagnostics.csv"]
                      if (root / n).exists()), None)
    if diag_path is None:
        raise FileNotFoundError("Diagnostics dosyasi bulunamadi")
    diag = (pd.read_excel(diag_path) if diag_path.suffix.startswith(".xls")
            else pd.read_csv(diag_path))

    def find_col(*keys):
        for c in diag.columns:
            cl = str(c).lower()
            if any(k in cl for k in keys):
                return c
        return None

    c_file = find_col("filename", "file")
    c_age = find_col("age")
    c_sex = find_col("gender", "sex")
    c_rhy = find_col("rhythm")
    print(f"  Diagnostics sutunlar: file={c_file} age={c_age} sex={c_sex} rhythm={c_rhy}")
    diag["_key"] = diag[c_file].astype(str).str.replace(".csv", "", regex=False)
    diag = diag.set_index("_key")

    csvs = sorted(data_dir.glob("*.csv"))
    print(f"  bulunan CSV         : {len(csvs)}")
    if args.limit > 0:
        csvs = csvs[:args.limit]

    # Baslik ve olcek tespiti (ilk 50 dosyadan)
    head = pd.read_csv(csvs[0], nrows=1, header=None)
    numeric_first = pd.to_numeric(head.iloc[0], errors="coerce").notna().all()
    header_opt = None if numeric_first else 0
    if header_opt == 0:
        leads = [normalize_lead_name(c) for c in pd.read_csv(csvs[0], nrows=0).columns]
    else:
        leads = list(CANON_12)
        print("  [!] CSV basligi yok, varsayilan sira kullaniliyor")
    print(f"  derivasyon sirasi   : {leads}")

    probe = []
    for f in csvs[:min(50, len(csvs))]:
        try:
            a = pd.read_csv(f, header=header_opt).to_numpy(dtype=np.float64)
            probe.append(np.nanpercentile(np.abs(a), 99))
        except Exception:
            pass
    med = float(np.nanmedian(probe)) if probe else np.nan
    scale = 1e-3 if (np.isfinite(med) and med > 50) else 1.0
    print(f"  ham |x| p99 medyani : {med:.2f}  ->  olcek = {scale}")

    w = H5Writer(out / "chapman.h5")
    rows, labs = [], []
    dropped = {"read": 0, "outlier": 0, "missing_lead": 0,
               "bad_age": 0, "no_label": 0, "no_meta": 0}
    parents = []

    for f in tqdm(csvs, desc="  Chapman", ncols=78):
        stem = Path(f).stem
        if stem not in diag.index:
            dropped["no_meta"] += 1
            continue
        meta = diag.loc[stem]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]

        age = pd.to_numeric(meta[c_age], errors="coerce") if c_age else np.nan
        if not valid_age(age):
            dropped["bad_age"] += 1
            continue

        rhy = str(meta[c_rhy]).strip().upper() if c_rhy else ""
        grp = CHAPMAN_RHYTHM_GROUP.get(rhy)
        if grp is None:
            dropped["no_label"] += 1
            continue

        try:
            arr = pd.read_csv(f, header=header_opt).to_numpy(dtype=np.float64) * scale
        except Exception:
            dropped["read"] += 1
            continue
        if arr.shape[1] != len(leads):
            dropped["read"] += 1
            continue

        s12, missing = reorder_to_canonical(arr, leads)
        if missing:
            dropped["missing_lead"] += 1
            continue
        s12 = resample_to(s12, 500, TARGET_FS)

        for wi, win in enumerate(to_windows(s12, TARGET_LEN, "crop")):
            if not np.isfinite(win).all():
                dropped["read"] += 1
                continue
            ma = float(np.max(np.abs(win)))
            if ma > MAX_ABS_MV:
                dropped["outlier"] += 1
                continue
            w.add(win)
            rows.append({
                "idx": len(rows),
                "record": f"{stem}_w{wi}", "parent_record": stem,
                "window_idx": wi,
                "age": float(age),
                "sex": harmonize_sex(meta[c_sex]) if c_sex else -1,
                "split": "",
                "labels": grp, "n_labels": 1,
                "max_abs": round(ma, 4),
                "rhythm_raw": rhy,
            })
            labs.append(multihot([grp], CHAPMAN_CLASSES))
            parents.append(stem)

    sp = record_level_split(parents, rng)
    for r in rows:
        r["split"] = sp[r["parent_record"]]

    _finish("chapman", w, rows, labs, CHAPMAN_CLASSES, out, dropped)


# ============================================================================
# ORTAK BITIRME
# ============================================================================

def _finish(name, w, rows, labs, classes, out: Path, dropped: dict):
    w.flush()
    meta = pd.DataFrame(rows)
    meta["idx"] = np.arange(len(meta))
    L = np.stack(labs, axis=0) if labs else np.zeros((0, len(classes)), np.uint8)

    assert w.n == len(meta) == len(L), \
        f"hizalama hatasi: signals={w.n} meta={len(meta)} labels={len(L)}"

    w.close({
        "fs": TARGET_FS, "seconds": TARGET_SEC, "length": TARGET_LEN,
        "units": "mV", "lead_order": ",".join(CANON_12),
        "gen8_leads": ",".join(GEN_8), "gen8_index": ",".join(map(str, GEN_8_IDX)),
        "classes": ",".join(classes), "max_abs_mv": MAX_ABS_MV,
        "normalized": "no",
    })
    meta.to_csv(out / f"{name}_meta.csv", index=False)
    np.save(out / f"{name}_labels.npy", L)

    print(f"  YAZILDI             : {w.n} ornek -> {name}.h5")
    print(f"  elenen              : {dropped}")
    if len(meta):
        print(f"  bolme               : {meta['split'].value_counts().to_dict()}")
        print(f"  benzersiz ust kayit : {meta['parent_record'].nunique()}")
        cnt = {c: int(L[:, i].sum()) for i, c in enumerate(classes)}
        print(f"  etiket dagilimi     : {cnt}")
        print(f"  yas ort/std         : {meta.age.mean():.1f} / {meta.age.std():.1f}")
        sx = meta.sex.value_counts().to_dict()
        print(f"  cinsiyet (0=E,1=K)  : {sx}")


# ============================================================================
# DOGRULAMA
# ============================================================================

def verify(out: Path):
    import h5py
    print("\n" + "=" * 78)
    print("DOGRULAMA")
    print("=" * 78)
    for name in ["ptbxl", "cpsc2018", "chapman"]:
        h5p = out / f"{name}.h5"
        if not h5p.exists():
            print(f"\n[{name}] dosya yok, atlandi")
            continue
        meta = pd.read_csv(out / f"{name}_meta.csv")
        L = np.load(out / f"{name}_labels.npy")
        with h5py.File(h5p, "r") as f:
            X = f["signals"]
            print(f"\n[{name}]")
            print(f"  signals shape : {X.shape}  dtype={X.dtype}")
            print(f"  meta / labels : {len(meta)} / {L.shape}")
            print(f"  attrs         : fs={f.attrs['fs']} units={f.attrs['units']} "
                  f"classes={f.attrs['classes']}")
            assert X.shape[0] == len(meta) == L.shape[0], "HIZALAMA HATASI"
            k = min(200, X.shape[0])
            sample = X[:k].astype(np.float64)
            print(f"  ilk {k} ornek : mean={sample.mean():.5f} "
                  f"std={sample.std():.4f} max|x|={np.abs(sample).max():.3f} mV")
            print(f"  NaN/Inf       : {int(np.isnan(sample).sum())}/"
                  f"{int(np.isinf(sample).sum())}")
            # 8-kanaldan 12'ye rekonstruksiyon tutarliligi
            i8 = [int(v) for v in f.attrs["gen8_index"].split(",")]
            s = sample[0]                                    # (12,1000)
            I, aVF = s[CANON_12.index("I")], s[CANON_12.index("aVF")]
            err = {
                "II":  np.abs(0.5 * I + aVF - s[CANON_12.index("II")]).max(),
                "III": np.abs(-0.5 * I + aVF - s[CANON_12.index("III")]).max(),
                "aVR": np.abs(-0.75 * I - 0.5 * aVF - s[CANON_12.index("aVR")]).max(),
                "aVL": np.abs(0.75 * I - 0.5 * aVF - s[CANON_12.index("aVL")]).max(),
            }
            print(f"  lead rekonstruksiyon max hata (mV): "
                  f"{ {k2: round(float(v), 4) for k2, v in err.items()} }")
            print(f"  gen8 index    : {i8}")


# ============================================================================
# ANA AKIS
# ============================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="EKG on isleme hatti")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--only", default="", help="ptbxl | cpsc2018 | chapman")
    ap.add_argument("--limit", type=int, default=0, help="veri seti basina ust sinir (0=hepsi)")
    ap.add_argument("--cpsc-strategy", choices=["crop", "windows"], default="crop",
                    help="crop: ilk 10 s | windows: ortusmeyen tum 10 s pencereler")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--verify", action="store_true", help="sadece dogrulama yap")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.verify:
        verify(out)
        return 0

    rng = np.random.default_rng(args.seed)
    print("=" * 78)
    print("EKG ON ISLEME HATTI")
    print("=" * 78)
    print(f"Hedef  : {TARGET_FS} Hz, {TARGET_SEC} s ({TARGET_LEN} ornek), mV, 12 kanal")
    print(f"Filtre : 0<yas<120, max|x|<={MAX_ABS_MV} mV")
    print(f"CPSC   : {args.cpsc_strategy}")
    print(f"Cikti  : {out.resolve()}")

    jobs = {"ptbxl": prepare_ptbxl, "cpsc2018": prepare_cpsc, "chapman": prepare_chapman}
    for name, fn in jobs.items():
        if args.only and args.only.lower() != name:
            continue
        print("\n" + "-" * 78)
        print(f"[{name}]  {PATHS[name]}")
        print("-" * 78)
        if not PATHS[name].exists():
            print("  [!] yol bulunamadi, atlandi")
            continue
        try:
            fn(PATHS[name], out, args, rng)
        except Exception:
            print("  [!] HATA:")
            traceback.print_exc(limit=4)

    verify(out)
    print("\n" + "=" * 78)
    print("BITTI")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
