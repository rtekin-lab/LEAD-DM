#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Asama 4a : Sinyal ve dagilim duzeyi metrikler  (Tablo 3)
=====================================================================

Metrikler
---------
Sinyal duzeyi (eslesmis gercek/sentetik cift basina):
    RMSE  genlik sapmasi
    DTW   Dinamik Zaman Bukmesi mesafesi (Sakoe-Chiba bantli, GPU)
    SC    Spektral Tutarlilik (magnitude-squared coherence, frekans ortalamasi)

Dagilim duzeyi:
    CS    Coverage Score  - uretilen ornekler gercek kumeleri ne kadar kapsiyor
    DR    Distribution Recall

GERCEK-GERCEK TABAN CIZGISI  (referans makalede YOK)
-----------------------------------------------------
Referans makale SC icin "ayni hastalik/yas/cinsiyete sahip iki GERCEK hasta
arasinda bile SC ~0.20" diyor ama RMSE ve DTW icin bu alt siniri vermiyor.
Bu olmadan "RMSE 0.2531 ne kadar iyi" sorusu cevaplanamaz.

Bu betik ayni kosula sahip GERCEK cift'ler uzerinden tum metriklerin
ulasilabilir alt/ust sinirini olcer. Sonuc makalede ayri bir satir olarak
raporlanmalidir; metodolojik bir katkidir.

Kullanim:
    python metrics.py --dataset ptbxl --gen ptbxl__M1_leadtcn
    python metrics.py --dataset ptbxl --gen ptbxl__M1_leadtcn --n 2000
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lead_dm.config import (RESULTS_DIR, PREPARED_DIR, GENERATED_DIR,
                            DATASET_CLASSES, CANON_12, FS, SIG_LEN,
                            GEN8_LEADS, require_files)
from lead_dm.reporting import (ExcelReport, setup_style, save_figure, PALETTE,
                               add_spec_table)

# Yollar lead_dm/config.py'den gelir (tek kaynak).
GEN_DIR = GENERATED_DIR
PREP_DIR = PREPARED_DIR


# ============================================================================
# VERI
# ============================================================================

def h5_gather(dset, idx: np.ndarray) -> np.ndarray:
    """
    HDF5'ten keyfi sirali/tekrarli indekslerle okuma.

    h5py fancy indexing ARTAN ve TEKRARSIZ indeks ister. Eslesme
    listelerinde ayni gercek kayit birden fazla cift'te partner olabilir
    (tekrar) ve sira karisik olabilir. Bu yardimci:
      1) benzersiz + sirali indekslerle bir kez okur
      2) istenen siraya ve tekrarlara geri esler
    """
    idx = np.asarray(idx)
    uniq, inv = np.unique(idx, return_inverse=True)
    block = np.asarray(dset[uniq], dtype=np.float32)
    return block[inv]


def load_real(dataset: str, split: str | None = None):
    import h5py
    require_files(PREP_DIR, dataset, "Hazirlanmis gercek")
    meta = pd.read_csv(PREP_DIR / f"{dataset}_meta.csv")
    labels = np.load(PREP_DIR / f"{dataset}_labels.npy")
    f = h5py.File(PREP_DIR / f"{dataset}.h5", "r")
    idx = np.arange(len(meta))
    if split:
        idx = idx[(meta["split"].astype(str) == split).to_numpy()]
    return f, meta, labels, idx


def load_gen(tag: str, split: str | None = None):
    import h5py
    require_files(GEN_DIR, tag, "Uretilmis")
    meta = pd.read_csv(GEN_DIR / f"{tag}_meta.csv")
    labels = np.load(GEN_DIR / f"{tag}_labels.npy")
    f = h5py.File(GEN_DIR / f"{tag}.h5", "r")
    idx = np.arange(len(meta))
    if split:
        idx = idx[(meta["split"].astype(str) == split).to_numpy()]
    return f, meta, labels, idx


# ============================================================================
# SINYAL DUZEYI METRIKLER
# ============================================================================

def rmse_pairs(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """(N,12,T) x (N,12,T) -> (N,) kayit basina RMSE"""
    return np.sqrt(((A - B) ** 2).mean(axis=(1, 2)))


@torch.no_grad()
def dtw_banded(A: torch.Tensor, B: torch.Tensor, band: int = 25,
               device=None) -> torch.Tensor:
    """
    Sakoe-Chiba bantli DTW, KOSEGEN (anti-diagonal) paralellestirmesiyle.

    Neden kosegen: (i,j) hucresi (i-1,j-1), (i-1,j) ve (i,j-1)'e baglidir.
    Satir bazli hesapta (i,j-1) ayni satirda oldugu icin ic dongu ARDISIK
    kalir -> T*W adet kucuk GPU cagrisi (T=1000, W=101 icin 101.000).
    (i,o) izgarasinda d = 2i + o tanimlarsak:
        (i-1,j-1) -> d-2 ,  (i-1,j) -> d-1 ,  (i,j-1) -> d-1
    yani bir kosegendeki tum hucreler BIRBIRINDEN BAGIMSIZ. Ardisik adim
    sayisi T*W'den ~2T+2r'ye duser (~180x hizlanma).

    A, B : (N, T)
    Doner: (N,) yol uzunluguna gore normalize DTW mesafesi
    """
    device = device or A.device
    A = A.to(device).float(); B = B.to(device).float()
    N, T = A.shape
    r = min(band, T - 1)
    W = 2 * r + 1
    INF = 1e12

    # D[n, i, k] ; k = o + r , o = j - i
    D = torch.full((N, T, W), INF, device=device)
    ar = torch.arange(T, device=device)

    for d in range(-r, 2 * (T - 1) + r + 1):
        # 2i + o = d , o in [-r, r] , i in [0, T-1]
        i_lo = max(0, (d - r + 1) // 2)
        i_hi = min(T - 1, (d + r) // 2)
        if i_lo > i_hi:
            continue
        i = ar[i_lo:i_hi + 1]
        o = d - 2 * i
        ok = (o >= -r) & (o <= r)
        i, o = i[ok], o[ok]
        if i.numel() == 0:
            continue
        j = i + o
        ok = (j >= 0) & (j < T)
        i, o, j = i[ok], o[ok], j[ok]
        if i.numel() == 0:
            continue
        k = o + r

        cost = (A[:, i] - B[:, j]) ** 2                      # (N, M)

        best = torch.full_like(cost, INF)
        m_i0 = (i == 0)
        m_k0 = (k == 0)
        m_kW = (k == W - 1)

        # kosegen: D[i-1, k]      (j-1 = (i-1)+o)
        idx = torch.where(m_i0, torch.zeros_like(i), i - 1)
        v = D[:, idx, k]
        best = torch.minimum(best, torch.where(m_i0, torch.full_like(v, INF), v))
        # yukari : D[i-1, k+1]    (j = (i-1)+(o+1))
        kk = torch.where(m_kW, k, k + 1)
        v = D[:, idx, kk]
        best = torch.minimum(best, torch.where(m_i0 | m_kW,
                                               torch.full_like(v, INF), v))
        # sol    : D[i, k-1]      (j-1 = i+(o-1))
        kk = torch.where(m_k0, k, k - 1)
        v = D[:, i, kk]
        best = torch.minimum(best, torch.where(m_k0,
                                               torch.full_like(v, INF), v))
        # baslangic hucresi (0,0)
        start = m_i0 & (o == 0)
        best = torch.where(start, torch.zeros_like(best), best)

        D[:, i, k] = cost + best

    return (D[:, T - 1, r] / (2 * T)).cpu()


@torch.no_grad()
def dtw_multilead(A: np.ndarray, B: np.ndarray, band: int = 25,
                  device="cpu", chunk: int = 256,
                  downsample: int = 4, progress: bool = True) -> np.ndarray:
    """
    (N,12,T) cift -> (N,) 12 derivasyon ortalamasi.

    downsample: DTW oncesi zaman ekseninde alt ornekleme carpani.
        100 Hz'de 1000 ornek uzerinde DTW gereksiz ince; 4x alt ornekleme
        (25 Hz, 250 ornek) hizalamayi korur ve maliyeti 16x dusurur.
        Bant da ayni oranda kucultulmelidir (varsayilan band=25 @ 250 ornek
        = 1 saniye kayma toleransi).
    """
    N, L, T = A.shape
    dev = torch.device(device)
    if downsample > 1:
        A = A[:, :, ::downsample]
        B = B[:, :, ::downsample]
    out = np.zeros(N)
    rng_ = range(0, N, chunk)
    if progress:
        try:
            from tqdm import tqdm
            rng_ = tqdm(list(rng_), desc="  DTW", ncols=78, leave=False)
        except ImportError:
            pass
    for i0 in rng_:
        i1 = min(i0 + chunk, N)
        a = torch.from_numpy(np.ascontiguousarray(
            A[i0:i1].reshape(-1, A.shape[-1])))
        b = torch.from_numpy(np.ascontiguousarray(
            B[i0:i1].reshape(-1, B.shape[-1])))
        d = dtw_banded(a, b, band, dev).view(i1 - i0, L).mean(1)
        out[i0:i1] = d.numpy()
    return out


def spectral_coherence(A: np.ndarray, B: np.ndarray, fs: int = FS,
                       nperseg: int = 256) -> np.ndarray:
    """
    Magnitude-squared coherence, frekans ve derivasyon uzerinden ortalama.
    (N,12,T) x (N,12,T) -> (N,)

    NaN korumasi: scipy'nin coherence'i Cxy = |Pxy|^2 / (Pxx*Pyy) hesaplar.
    Bir derivasyon sabitse veya bir frekans biniminde guc sifirsa payda
    sifir olur ve NaN uretir. Burada:
      * varyansi ihmal edilebilir derivasyonlar atlanir
      * kalan NaN binler ortalamadan cikarilir
      * hicbir gecerli deger yoksa kayit NaN olarak isaretlenir ve
        genel ortalamada nanmean ile gormezden gelinir
    """
    from scipy.signal import coherence
    N, L, T = A.shape
    out = np.full(N, np.nan)
    it = range(N)
    try:
        from tqdm import tqdm
        it = tqdm(it, desc="  SC ", ncols=78, leave=False)
    except ImportError:
        pass
    with np.errstate(invalid="ignore", divide="ignore"):
        for i in it:
            vals = []
            for l in range(L):
                a, b = A[i, l], B[i, l]
                if a.std() < 1e-8 or b.std() < 1e-8:
                    continue
                _, Cxy = coherence(a, b, fs=fs, nperseg=min(nperseg, T))
                Cxy = Cxy[np.isfinite(Cxy)]
                if Cxy.size:
                    vals.append(float(Cxy.mean()))
            if vals:
                out[i] = float(np.mean(vals))
    return out


# ============================================================================
# DAGILIM DUZEYI METRIKLER
# ============================================================================

def _features(X: np.ndarray) -> np.ndarray:
    """
    Basit ama anlamli oznitelik vektoru (siniflandirici gerektirmez):
      derivasyon basina  [std, p95-p05, RMS, sifir gecis orani,
                          band enerjileri 0-5 / 5-15 / 15-40 Hz]
    """
    N, L, T = X.shape
    F = []
    Xf = np.fft.rfft(X, axis=-1)
    freqs = np.fft.rfftfreq(T, d=1.0 / FS)
    P = np.abs(Xf) ** 2
    bands = [(0.5, 5), (5, 15), (15, 40)]
    for lo, hi in bands:
        m = (freqs >= lo) & (freqs < hi)
        F.append(P[:, :, m].sum(-1))
    F.append(X.std(-1))
    F.append(np.percentile(X, 95, axis=-1) - np.percentile(X, 5, axis=-1))
    F.append(np.sqrt((X ** 2).mean(-1)))
    F.append((np.diff(np.signbit(X), axis=-1) != 0).mean(-1))
    Z = np.concatenate([f.reshape(N, -1) for f in F], axis=1)
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-8)
    return Z


def coverage_recall(real: np.ndarray, gen: np.ndarray, k: int = 5):
    """
    Naeem ve ark. tarzi coverage + recall.

    coverage : gercek orneklerin kacinin k-NN yaricapi icinde en az bir
               uretilen ornek var
    recall   : uretilen orneklerin kacinin gercek manifolda dustugu
    """
    from scipy.spatial import cKDTree
    tr = cKDTree(real)
    tg = cKDTree(gen)
    # gercek orneklerin k-NN yaricapi
    dr, _ = tr.query(real, k=k + 1)
    radius_r = dr[:, -1]
    # her gercek ornege en yakin uretilen
    dg, _ = tg.query(real, k=1)
    coverage = float((dg <= radius_r).mean())
    # recall: uretilen ornek gercek manifoldun icinde mi
    dgg, _ = tg.query(gen, k=k + 1)
    radius_g = dgg[:, -1]
    drg, _ = tr.query(gen, k=1)
    recall = float((drg <= radius_g).mean())
    return coverage, recall


# ============================================================================
# ESLESTIRME
# ============================================================================

def match_real_pairs(meta: pd.DataFrame, labels: np.ndarray, idx: np.ndarray,
                     rng: np.random.Generator, n: int, age_tol: int = 3):
    """
    Ayni hastalik + cinsiyet + (yas +- tol) olan GERCEK cift'ler bulur.
    Bu, tum metriklerin ulasilabilir alt sinirini verir.
    """
    sub = meta.iloc[idx].reset_index(drop=True)
    lab = labels[idx]
    key = [tuple(lab[i]) + (int(sub.loc[i, "sex"]),) for i in range(len(sub))]
    ages = sub["age"].to_numpy()
    groups = {}
    for i, k in enumerate(key):
        groups.setdefault(k, []).append(i)

    pairs = []
    keys = [k for k, v in groups.items() if len(v) >= 2]
    rng.shuffle(keys)
    for k in keys:
        members = np.array(groups[k])
        rng.shuffle(members)
        for a in members:
            cand = members[(np.abs(ages[members] - ages[a]) <= age_tol)
                           & (members != a)]
            if len(cand):
                pairs.append((idx[a], idx[rng.choice(cand)]))
            if len(pairs) >= n:
                break
        if len(pairs) >= n:
            break
    return pairs


# ============================================================================
# ANA AKIS
# ============================================================================

def evaluate_pairs(A, B, name, device, band, do_dtw=True, do_sc=True,
                   downsample=4):
    import time as _t
    out = {"Pairs": len(A)}
    t0 = _t.time()
    out["RMSE"] = float(rmse_pairs(A, B).mean())
    if do_dtw:
        out["DTW"] = float(dtw_multilead(A, B, band=band, device=device,
                                         downsample=downsample).mean())
    if do_sc:
        sc = spectral_coherence(A, B)
        out["SC"] = float(np.nanmean(sc))
        out["SC_valid"] = int(np.isfinite(sc).sum())
    print(f"    ({name}: {_t.time()-t0:.1f} s)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl", choices=list(DATASET_CLASSES))
    ap.add_argument("--gen", required=True,
                    help="generated/ altindaki etiket, or. ptbxl__M1_leadtcn")
    ap.add_argument("--n", type=int, default=1500,
                    help="metrik icin ornek cift sayisi")
    ap.add_argument("--band", type=int, default=25,
                    help="DTW bant yaricapi (alt ornekleme SONRASI adim)")
    ap.add_argument("--dtw-downsample", type=int, default=4,
                    help="DTW oncesi zaman ekseninde alt ornekleme carpani")
    ap.add_argument("--k", type=int, default=5, help="CS/DR icin k-NN")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-dtw", action="store_true")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("LEAD-DM | Asama 4a: Sinyal ve Dagilim Metrikleri")
    print("=" * 78)

    fr, mr, lr, ir = load_real(args.dataset)
    fg, mg, lg, ig = load_gen(args.gen)
    print(f"Gercek   : {len(ir)} kayit")
    print(f"Uretilen : {len(ig)} kayit")

    # --- eslesmis gercek/sentetik ---
    src = mg["source_record"].tolist()
    rec2row = {r: i for i, r in enumerate(mr["record"].tolist())}
    pair_idx = [(rec2row[s], j) for j, s in enumerate(src) if s in rec2row]
    rng.shuffle(pair_idx)
    pair_idx = pair_idx[:args.n]
    print(f"Eslesmis cift: {len(pair_idx)}")

    ri = np.array([p[0] for p in pair_idx])
    gi = np.array([p[1] for p in pair_idx])
    R = h5_gather(fr["signals"], ri)
    G = h5_gather(fg["signals"], gi)

    print("\n[1] Sentetik vs Gercek (eslesmis)")
    res_gen = evaluate_pairs(R, G, "sentetik", args.device, args.band,
                             do_dtw=not args.no_dtw,
                             downsample=args.dtw_downsample)
    for k, v in res_gen.items():
        print(f"  {k:<6}: {v}")

    # --- GERCEK-GERCEK taban cizgisi ---
    print("\n[2] Gercek-Gercek taban cizgisi (ayni kosul, farkli hasta)")
    pairs = match_real_pairs(mr, lr, ir, rng, args.n)
    print(f"  bulunan cift: {len(pairs)}")
    if len(pairs) >= 50:
        a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
        RA = h5_gather(fr["signals"], a)
        RB = h5_gather(fr["signals"], b)
        res_real = evaluate_pairs(RA, RB, "gercek-gercek", args.device,
                                  args.band, do_dtw=not args.no_dtw,
                                  downsample=args.dtw_downsample)
        for k, v in res_real.items():
            print(f"  {k:<6}: {v}")
    else:
        res_real = {"Pairs": len(pairs)}
        print("  [!] yeterli cift bulunamadi")

    # --- dagilim duzeyi ---
    print("\n[3] Dagilim duzeyi (CS / DR)")
    nD = min(3000, len(ir), len(ig))
    sr = np.sort(rng.choice(len(ir), nD, replace=False))
    sg = np.sort(rng.choice(len(ig), nD, replace=False))
    FR = _features(h5_gather(fr["signals"], sr))
    FG = _features(h5_gather(fg["signals"], sg))
    cs, dr = coverage_recall(FR, FG, k=args.k)
    print(f"  CS = {cs:.4f}   DR = {dr:.4f}   (n={nD}, k={args.k})")

    # --- normalize oranlar (metodolojik katki) ---
    #
    # Gercek ornekler x = mu_c + e (mu_c kosullu ortalama, Var(e)=s^2) ise:
    #     uretici tam ORTALAMAYI verirse   RMSE = s
    #     iki GERCEK hasta arasinda        RMSE = s*sqrt(2)
    # Dolayisiyla
    #     oran = RMSE_uretilen / RMSE_gercek-gercek
    #     0.707 -> cesitlilik YOK (ortalamaya cokme)
    #     1.000 -> dogru kosullu ornekleme
    # Bu cerceve, referans makalenin vermedigi bir yorumlama olcegi sunar:
    # ham RMSE tek basina "iyi mi" sorusunu cevaplayamaz.
    ratios = {}
    for k in ("RMSE", "DTW", "SC"):
        rv, gv = res_real.get(k), res_gen.get(k)
        if rv and gv and rv > 0:
            ratios[k] = gv / rv
    print("\n[4] Normalize oranlar (uretilen / gercek-gercek)")
    if "RMSE" in ratios:
        print(f"  RMSE orani : {ratios['RMSE']:.3f}   "
              f"(0.707 = ortalamaya cokme, 1.000 = tam cesitlilik)")
    if "DTW" in ratios:
        print(f"  DTW  orani : {ratios['DTW']:.3f}")
    if "SC" in ratios:
        print(f"  SC   orani : {ratios['SC']:.3f}   (1.000 = ulasilabilir tavan)")
    if "RMSE" in ratios:
        r = ratios["RMSE"]
        pos = (r - 0.7071) / (1.0 - 0.7071)
        print(f"  -> cesitlilik konumu: %{100*max(0,min(1,pos)):.0f} "
              f"(0 = ortalama, 100 = tam cesitlilik)")

    # --- rapor ---
    rows = [{"Dataset": args.dataset, "Model": args.gen,
             "RMSE": res_gen.get("RMSE"), "DTW": res_gen.get("DTW"),
             "SC": res_gen.get("SC"), "CS": cs, "DR": dr,
             "Pairs": res_gen["Pairs"]},
            {"Dataset": args.dataset, "Model": "GERCEK-GERCEK (taban)",
             "RMSE": res_real.get("RMSE"), "DTW": res_real.get("DTW"),
             "SC": res_real.get("SC"), "CS": 1.0, "DR": 1.0,
             "Pairs": res_real.get("Pairs")}]
    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = ExcelReport(RESULTS_DIR / f"METRICS_{args.gen}.xlsx",
                      meta={"Asama": "4a - sinyal/dagilim metrikleri",
                            "Veri seti": args.dataset,
                            "Uretilen": args.gen,
                            "DTW bant": args.band, "k-NN": args.k,
                            "Not": ("GERCEK-GERCEK satiri metriklerin "
                                    "ulasilabilir alt/ust sinirini gosterir")})
    add_spec_table(rep, "T3_signal_distribution", df, sheet="T3_sinyal_dagilim")
    dfR = pd.DataFrame([{
        "Metrik": k, "Uretilen": res_gen.get(k),
        "Gercek-Gercek": res_real.get(k), "Oran": v,
        "Yorum": {"RMSE": "0.707=ortalamaya cokme, 1.0=tam cesitlilik",
                  "DTW": "0.707=ortalamaya cokme, 1.0=tam cesitlilik",
                  "SC": "1.0 = ulasilabilir tavan"}.get(k, "")}
        for k, v in ratios.items()])
    rep.add_table("T3b_normalize_oranlar", dfR,
                  caption=("Normalize oranlar. Gercek-gercek taban cizgisi "
                           "metriklerin ulasilabilir sinirini verir; ham "
                           "degerler tek basina yorumlanamaz."))
    print(f"\n[Excel] {rep.save()}")

    # sekil
    plt = setup_style()
    fig, ax = plt.subplots(1, 3, figsize=(8.4, 2.6))
    names = ["RMSE", "DTW", "SC"]
    for i, n in enumerate(names):
        vals = [df.loc[0, n], df.loc[1, n]]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in vals):
            continue
        ax[i].bar([0, 1], vals, color=[PALETTE[0], PALETTE[2]], width=0.55)
        ax[i].set_xticks([0, 1])
        ax[i].set_xticklabels(["Sentetik\nvs Gercek", "Gercek\nvs Gercek"])
        ax[i].set_ylabel(n); ax[i].grid(True, axis="y")
        ax[i].set_title(f"({chr(97+i)}) {n}", loc="left")
    fig.tight_layout()
    save_figure(fig, RESULTS_DIR / "figures" / f"metrics_{args.gen}")
    print(f"[Sekil] {RESULTS_DIR/'figures'}")

    fr.close(); fg.close()


if __name__ == "__main__":
    main()
