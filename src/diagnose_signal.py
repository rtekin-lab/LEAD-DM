#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Sinyal morfolojisi ve spektrum teshisi
===================================================
diagnose_latent.py gizil uzayin DAGILIMSAL ozelliklerini olcer ve "yesil"
verebilir; ama uretilen sinyalin EKG'ye BENZEYIP benzemedigini olcmez.
Bu betik o boslugu kapatir.

Gozlenen belirti (PTB-XL, M1_leadtcn):
    ritim dogru, polarite dogru, genlik dogru
    ANCAK QRS darbe gibi (1-3 ornek), vurular arasi genis bantli gurultu
    -> asiri yuksek frekans enerjisi

Testler
-------
S1  Guc spektral yogunlugu : gercek / AE-rekonstruksiyon / uretilen
S2  Yuksek frekans enerji orani (>25 Hz)
S3  R-tepe morfolojisi     : genislik, prominans, vuru sayisi
S4  Kismi denoising cikti sekilleri (t=200/400/600) - EKG olarak cizilir
S5  Otokorelasyon          : ritim yapisi

Yorum
-----
Uretilenin PSD'si gercekten YUKSEK frekansta belirgin fazla ise, sucla
frekans kaybi terimi (magnitude-only + HF agirlikli) buyuk olasilikla
iliskilidir; bu terim FAZDAN BAGIMSIZ oldugu icin dogru spektrumlu ama
gurultulu cozumleri odullendirir.

Kullanim:
    python diagnose_signal.py --dataset ptbxl --dm-name M1_leadtcn
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lead_dm.config import (RUNS_DIR, RESULTS_DIR, DATASET_CLASSES, M1Config,
                            DiffusionConfig, CANON_12, FS, SIG_LEN,
                            SIGNAL_SCALE)
from lead_dm.data import make_loader, leads8_to_12, denormalize
from lead_dm.autoencoder import load_ae_checkpoint
from lead_dm.backbone_m1 import build_m1
from lead_dm.diffusion import GaussianDiffusion, EMA
from lead_dm.reporting import (ExcelReport, setup_style, save_figure,
                               plot_ecg_12lead, PALETTE)


# ============================================================================
def load_models(dataset, dm_name, device):
    p = RUNS_DIR / f"{dataset}__{dm_name}" / "best.pt"
    ck = torch.load(p, map_location=device, weights_only=False)
    ae_dir = ck.get("ae_dir", f"{dataset}__ae")
    ae, ae_cfg, _ = load_ae_checkpoint(RUNS_DIR / ae_dir / "best.pt", device)
    n_classes = ck.get("n_classes", len(DATASET_CLASSES[dataset]))

    m1cfg, dcfg = M1Config(), DiffusionConfig()
    cfgj = p.parent / "config.json"
    if cfgj.exists():
        import json
        j = json.loads(cfgj.read_text(encoding="utf-8"))
        m1cfg = M1Config(**{k: v for k, v in j.get("model", {}).items()
                            if k in M1Config.__dataclass_fields__})
        dcfg = DiffusionConfig(**{k: v for k, v in j.get("diffusion", {}).items()
                                  if k in DiffusionConfig.__dataclass_fields__})
    model = build_m1(n_classes, m1cfg).to(device)
    model.load_state_dict(ck["model"])
    if "ema" in ck:
        e = EMA(model); e.load_state_dict(ck["ema"]); e.apply_to(model)
    model.eval()
    diff = GaussianDiffusion(dcfg, autoencoder=ae).to(device)
    print(f"  DM: step={ck.get('step')}  val={ck.get('val'):.5f}")
    print(f"  lambda_freq={dcfg.lambda_freq}  freq_hf_gain={dcfg.freq_hf_gain}")
    return ae, model, diff, dcfg


# ============================================================================
def psd(x: np.ndarray, fs=FS, nperseg=256):
    """(N,12,T) -> (freqs, mean PSD over records & leads)"""
    from scipy.signal import welch
    f, P = welch(x, fs=fs, nperseg=min(nperseg, x.shape[-1]), axis=-1)
    return f, P.mean(axis=(0, 1))


def hf_ratio(x: np.ndarray, fs=FS, cut=25.0):
    f, P = psd(x, fs)
    m = f >= cut
    return float(P[m].sum() / max(P.sum(), 1e-12))


def qrs_stats(x12: np.ndarray, fs=FS, lead="II"):
    """
    R tepelerini bulup genislik ve sayilarini olcer.
    x12: (N,12,T) mV
    """
    from scipy.signal import find_peaks, peak_widths
    li = CANON_12.index(lead)
    n_beats, widths, prom = [], [], []
    for i in range(x12.shape[0]):
        s = x12[i, li]
        s = s - np.median(s)
        h = max(0.25 * np.percentile(np.abs(s), 99), 1e-3)
        pk, props = find_peaks(s, height=h, distance=int(0.25 * fs))
        n_beats.append(len(pk))
        if len(pk):
            w = peak_widths(s, pk, rel_height=0.5)[0]
            widths.append(np.median(w) / fs * 1000.0)     # ms
            prom.append(float(np.median(props["peak_heights"])))
    return {
        "beats_per_10s": float(np.mean(n_beats)),
        "hr_bpm": float(np.mean(n_beats)) * 6.0,
        # DIKKAT: bu klinik QRS SURESI (Q basi - S sonu, 80-100 ms) DEGIL.
        # peak_widths(rel_height=0.5) R tepesinin yari-prominans genisligini
        # olcer; sivri bir R dalgasi icin ~25-35 ms cikar. Karsilastirma
        # GERCEK veriyle yapilmali, klinik normlarla degil.
        "r_halfwidth_ms": float(np.median(widths)) if widths else float("nan"),
        "r_amp_mV": float(np.median(prom)) if prom else float("nan"),
    }


def autocorr_peak(x12: np.ndarray, fs=FS, lead="II"):
    li = CANON_12.index(lead)
    out = []
    for i in range(min(64, x12.shape[0])):
        s = x12[i, li] - x12[i, li].mean()
        a = np.correlate(s, s, "full")[len(s) - 1:]
        a = a / (a[0] + 1e-12)
        lo, hi = int(0.3 * fs), int(2.0 * fs)
        seg = a[lo:hi]
        out.append(float(seg.max()))
    return float(np.mean(out))


# ============================================================================
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl", choices=list(DATASET_CLASSES))
    ap.add_argument("--dm-name", default="M1_leadtcn")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--guidance", type=float, default=0.0,
                    help="butunsel CFG agirligi (tum kosullar birlikte). "
                         "1.0 = kosullu model, >1 keskinlestirir.")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(0)

    print("=" * 78)
    print("LEAD-DM | Sinyal Morfolojisi ve Spektrum Teshisi")
    print("=" * 78)
    ae, model, diff, dcfg = load_models(args.dataset, args.dm_name, device)

    te = make_loader(args.dataset, "test", args.n, shuffle=False, num_workers=2)
    b = next(iter(te))
    x = b["x"][:args.n].to(device)
    cond = {"disease": b["disease"][:args.n].to(device),
            "age_bits": b["age_bits"][:args.n].to(device),
            "sex": b["sex"][:args.n].to(device)}

    REAL = leads8_to_12(denormalize(x)).cpu().numpy()
    z0 = ae.encode_scaled(x)
    RECON = leads8_to_12(denormalize(ae.decode_scaled(z0))).cpu().numpy()

    C, T = model.channels, ae.cfg.latent_len
    guid = ({"all": args.guidance} if args.guidance else None)
    if guid:
        print(f"  yonlendirme: {guid}")
    GEN = {}
    for name, kw in [("DDIM-50", dict(ddim_steps=50)),
                     ("DDIM-250", dict(ddim_steps=250)),
                     ("DDPM-1000", dict(ddim_steps=None))]:
        z = diff.sample(model, cond, (args.n, C, T), device,
                        guidance=guid, progress=False, **kw)
        GEN[name] = leads8_to_12(denormalize(ae.decode_scaled(z))).cpu().numpy()
        print(f"  {name} uretildi")

    # ---------------- S1/S2 spektrum ----------------
    print("\n[S1/S2] Guc spektrumu ve yuksek frekans orani")
    sets = {"GERCEK": REAL, "AE-rekon": RECON, **GEN}
    rows = []
    for k, v in sets.items():
        r = hf_ratio(v)
        rows.append({"Kaynak": k, "HF_orani(>25Hz)": r,
                     "std_mV": float(v.std()),
                     "max_mV": float(np.abs(v).max())})
        print(f"  {k:<10}: HF(>25Hz) = {r*100:6.2f}%   std={v.std():.4f} mV")
    ref = rows[0]["HF_orani(>25Hz)"]
    for r in rows:
        r["HF_kat(gercege_gore)"] = r["HF_orani(>25Hz)"] / max(ref, 1e-12)
    dfS = pd.DataFrame(rows)
    worst = max(rows[2:], key=lambda r: r["HF_kat(gercege_gore)"])
    print(f"\n  -> Uretilen sinyalde HF enerjisi gercegin "
          f"{worst['HF_kat(gercege_gore)']:.1f} KATI")

    # ---------------- S3 morfoloji ----------------
    print("\n[S3] QRS morfolojisi (lead II)")
    mrows = []
    for k, v in sets.items():
        m = qrs_stats(v)
        m["Kaynak"] = k
        mrows.append(m)
        print(f"  {k:<10}: {m['hr_bpm']:5.1f} bpm   "
              f"R yari-gen. {m['r_halfwidth_ms']:6.1f} ms   "
              f"R genligi {m['r_amp_mV']:.3f} mV")
    dfM = pd.DataFrame(mrows)[["Kaynak", "hr_bpm", "r_halfwidth_ms",
                               "r_amp_mV", "beats_per_10s"]]
    print("  (GERCEK satiriyla karsilastirin; klinik QRS suresi degildir)")

    # ---------------- S5 otokorelasyon ----------------
    print("\n[S5] Ritim yapisi (otokorelasyon tepe degeri)")
    arows = []
    for k, v in sets.items():
        a = autocorr_peak(v)
        arows.append({"Kaynak": k, "autocorr_peak": a})
        print(f"  {k:<10}: {a:.4f}   (yuksek = duzenli ritim)")
    dfA = pd.DataFrame(arows)

    # ---------------- S4 kismi denoising ----------------
    print("\n[S4] Kismi denoising ciktilari")
    figdir = RESULTS_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    for t_start in (200, 400, 600):
        tt = torch.full((4,), t_start, device=device, dtype=torch.long)
        zt, _ = diff.q_sample(z0[:4], tt)
        z = zt.clone()
        c4 = {k: v[:4] for k, v in cond.items()}
        for i in range(t_start, -1, -1):
            ti = torch.full((4,), i, device=device, dtype=torch.long)
            out = model(z, ti, c4["disease"], c4["age_bits"], c4["sex"])
            zz0 = diff.decode_output(z, ti, out)[0].clamp(-4, 4)
            mean = (diff._extract(diff.posterior_mean_c0, ti, z.shape) * zz0
                    + diff._extract(diff.posterior_mean_ct, ti, z.shape) * z)
            if i > 0:
                lv = diff._extract(diff.posterior_logvar, ti, z.shape)
                z = mean + (0.5 * lv).exp() * torch.randn_like(z)
            else:
                z = mean
        xr = leads8_to_12(denormalize(ae.decode_scaled(z))).cpu().numpy()
        plot_ecg_12lead(xr[0], figdir / f"sigdiag_{args.dataset}_partial_t{t_start}",
                        fs=FS, second=REAL[0],
                        labels=(f"t={t_start}'den geri kazanim", "Gercek"),
                        title=f"Kismi denoising (t_start={t_start})")
        print(f"  t={t_start} sekli kaydedildi")

    # ---------------- Sekiller ----------------
    plt = setup_style()
    f_ax, P_ax = psd(REAL)
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
    for i, (k, v) in enumerate(sets.items()):
        f_, P_ = psd(v)
        ax[0].semilogy(f_, P_, color=PALETTE[i], lw=1.1, label=k)
    ax[0].axvline(25, color="k", ls=":", lw=.8)
    ax[0].set_xlabel("Frekans (Hz)"); ax[0].set_ylabel("PSD")
    ax[0].legend(fontsize=7); ax[0].grid(True, which="both")
    ax[0].set_title("(a) Guc spektral yogunlugu", loc="left")

    ks = list(sets); xs = np.arange(len(ks))
    ax[1].bar(xs, dfS["HF_orani(>25Hz)"] * 100,
              color=[PALETTE[i] for i in range(len(ks))], width=0.6)
    ax[1].set_xticks(xs); ax[1].set_xticklabels(ks, rotation=25, ha="right")
    ax[1].set_ylabel("HF enerji orani >25 Hz (%)")
    ax[1].grid(True, axis="y")
    ax[1].set_title("(b) Yuksek frekans fazlaligi", loc="left")
    fig.tight_layout()
    save_figure(fig, figdir / f"sigdiag_{args.dataset}_spectrum")

    plot_ecg_12lead(GEN["DDPM-1000"][0],
                    figdir / f"sigdiag_{args.dataset}_gen_vs_real",
                    fs=FS, second=REAL[0],
                    labels=("Uretilen (DDPM-1000)", "Gercek"),
                    title="Uretilen ve gercek EKG karsilastirmasi")

    # ---------------- Excel ----------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = ExcelReport(RESULTS_DIR / f"SIGDIAG_{args.dataset}_{args.dm_name}.xlsx",
                      meta={"Icerik": "Sinyal morfolojisi / spektrum teshisi",
                            "Veri seti": args.dataset, "Model": args.dm_name,
                            "lambda_freq": dcfg.lambda_freq,
                            "freq_hf_gain": dcfg.freq_hf_gain})
    rep.add_table("S1_spektrum", dfS,
                  caption="Guc spektrumu ve >25 Hz enerji orani.")
    rep.add_table("S3_morfoloji", dfM,
                  caption="QRS morfolojisi. Gercek QRS genisligi 80-100 ms.")
    rep.add_table("S5_ritim", dfA, caption="Otokorelasyon tepe degeri.")
    print(f"\n[Excel] {rep.save()}")
    print(f"[Sekil] {figdir}")

    # ---------------- Karar ----------------
    print("\n" + "=" * 78)
    print("TESHIS")
    print("=" * 78)
    hf_k = worst["HF_kat(gercege_gore)"]
    g = lambda col, k: float(dfM.loc[dfM.Kaynak == k, col].iloc[0])
    a = lambda k: float(dfA.loc[dfA.Kaynak == k, "autocorr_peak"].iloc[0])
    sd = lambda k: float(dfS.loc[dfS.Kaynak == k, "std_mV"].iloc[0])

    checks = [
        ("HF enerji fazlaligi", hf_k, 2.0, "less"),
        ("R yari-genislik orani",
         g("r_halfwidth_ms", "DDPM-1000") / g("r_halfwidth_ms", "GERCEK"),
         (0.7, 1.4), "range"),
        ("R genlik orani",
         g("r_amp_mV", "DDPM-1000") / g("r_amp_mV", "GERCEK"),
         (0.75, 1.25), "range"),
        ("Sinyal std orani", sd("DDPM-1000") / sd("GERCEK"),
         (0.8, 1.25), "range"),
        ("Ritim duzenliligi orani", a("DDPM-1000") / a("GERCEK"),
         (0.8, 1.25), "range"),
    ]
    fails = []
    for name, val, thr, kind in checks:
        if kind == "less":
            ok = val < thr
            txt = f"< {thr}"
        else:
            ok = thr[0] <= val <= thr[1]
            txt = f"{thr[0]}-{thr[1]}"
        print(f"  {name:<26}: {val:.2f}x   (hedef {txt})")
        print("     " + ("[OK]" if ok else "[!!] eksik"))
        if not ok:
            fails.append(name)

    print("\n  ORNEKLEME ADIMI ETKISI:")
    for k in ("DDIM-50", "DDIM-250", "DDPM-1000"):
        print(f"    {k:<11}: otokorr={a(k):.4f}  std={sd(k):.4f} mV  "
              f"R={g('r_amp_mV', k):.3f} mV")
    print("    (adim sayisiyla artiyorsa model saglam, ornekleme sinirli)")

    if fails:
        print("\n  ONERILER (ucuzdan pahaliya):")
        print("    1) Ornekleme: DDPM-1000 veya DDIM-500 kullanin")
        print("       python diagnose_signal.py --dataset "
              f"{args.dataset} --dm-name {args.dm_name} --guidance 1.5")
        print("    2) Kosullu yonlendirme (yeniden egitim YOK):")
        print("       generate.py --guidance-disease 1.5")
        print("    3) Daha uzun egitim:")
        print("       train_diffusion.py --dataset "
              f"{args.dataset} --iters 300000 --name M1_long")
        print("    4) Kayip ablasyonu (planli deney):")
        print("       train_diffusion.py --lambda-freq 0 --name M1_nofreq")


if __name__ == "__main__":
    main()
