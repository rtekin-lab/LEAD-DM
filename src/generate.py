#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Asama 3 : Kosullu uretim
=====================================
Referans makalenin kurgusu birebir uygulanir (Bolum 4.3):

  * Rastgele uretim YOK. Her GERCEK kaydin etiket bilgisi (hastalik, yas,
    cinsiyet) alinir ve o kosullarla bir sentetik kayit uretilir.
  * Bolme aynen tasinir: gercek kayit A egitim bolumundeyse, A'nin
    etiketinden uretilen A' de uretilen veri setinin egitim bolumune gider.
  * Boylece R-Train / R-Test ve G-Train / G-Test dortlusu olusur.

Cikti (prepare_data.py ile AYNI format, dogrudan degerlendirmede kullanilir):
    generated/<dataset>__<name>.h5            signals (N,12,1000) mV
    generated/<dataset>__<name>_meta.csv
    generated/<dataset>__<name>_labels.npy

Tablo 11 senaryolari (--scenario):
    baseline     tam ve dogru kosul
    only_dl      yalnizca hastalik etiketi (yas/cinsiyet maskeli)
    only_psi     yalnizca yas + cinsiyet (hastalik maskeli)
    perturbed    %30 olasilikla hatali hastalik/cinsiyet, yas +-10 yil

Kullanim:
    python generate.py --dataset ptbxl
    python generate.py --dataset ptbxl --ddim 250
    python generate.py --dataset ptbxl --scenario only_dl
    python generate.py --dataset ptbxl --guidance-disease 2.0
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lead_dm.config import (RUNS_DIR, RESULTS_DIR, GENERATED_DIR,
                            DATASET_CLASSES, M1Config, DiffusionConfig,
                            GEN8_LEADS, CANON_12, FS, SIG_LEN, SIGNAL_SCALE,
                            AGE_MAX)
from lead_dm.data import (ECGDataset, leads8_to_12, denormalize, gray_encode,
                          SEX_MALE, SEX_FEMALE)
from lead_dm.autoencoder import load_ae_checkpoint
from lead_dm.backbone_m1 import build_m1
from lead_dm.diffusion import GaussianDiffusion, EMA
from lead_dm.reporting import (ExcelReport, plot_ecg_12lead, setup_style,
                               save_figure, PALETTE)

# Yollar lead_dm/config.py'den gelir (tek kaynak).
GEN_DIR = GENERATED_DIR


# ============================================================================
def load_models(dataset: str, dm_name: str, device):
    dm_path = RUNS_DIR / f"{dataset}__{dm_name}" / "best.pt"
    if not dm_path.exists():
        raise FileNotFoundError(f"Difuzyon modeli bulunamadi: {dm_path}")
    ck = torch.load(dm_path, map_location=device, weights_only=False)

    # Difuzyon HANGI AE ile egitildiyse uretim de ONU kullanmali.
    # (A2_flat_latent ablasyonunda bu eslesme yoktu: model duz gizil
    #  uzayda egitilip lead-yapili dekoderle cozulunce sonuc cop cikti.)
    ae_dir = ck.get("ae_dir")
    if ae_dir is None:
        ae_dir = f"{dataset}__ae"
        if "flat" in dm_name.lower() or "A2" in dm_name:
            ae_dir = f"{dataset}__ae_flat"
            print(f"  [!] checkpoint'te ae_dir yok; model adindan "
                  f"'{ae_dir}' tahmin edildi")
        else:
            print(f"  [!] checkpoint'te ae_dir yok; varsayilan '{ae_dir}'")
    ae, ae_cfg, ae_ck = load_ae_checkpoint(
        RUNS_DIR / ae_dir / "best.pt", device)
    print(f"  AE kaynagi    : {ae_dir}")
    n_classes = ck.get("n_classes", len(DATASET_CLASSES[dataset]))

    cfg_path = dm_path.parent / "config.json"
    m1cfg = M1Config()
    dcfg = DiffusionConfig()
    if cfg_path.exists():
        import json
        j = json.loads(cfg_path.read_text(encoding="utf-8"))
        m1cfg = M1Config(**{k: v for k, v in j.get("model", {}).items()
                            if k in M1Config.__dataclass_fields__})
        dcfg = DiffusionConfig(**{k: v for k, v in j.get("diffusion", {}).items()
                                  if k in DiffusionConfig.__dataclass_fields__})

    model = build_m1(n_classes, m1cfg).to(device)
    model.load_state_dict(ck["model"])
    if "ema" in ck:
        ema = EMA(model)
        ema.load_state_dict(ck["ema"])
        ema.apply_to(model)          # EMA agirliklari ornekleme icin sart
    model.eval()

    diff = GaussianDiffusion(dcfg, autoencoder=ae).to(device)

    # Gizil uzay boyutu difuzyon omurgasiyla uyusmali
    if model.channels != ae_cfg.latent_ch:
        raise RuntimeError(
            f"GIZIL UZAY UYUSMAZLIGI: difuzyon {model.channels} kanal "
            f"bekliyor, AE {ae_cfg.latent_ch} kanal uretiyor.\n"
            f"  AE: {ae_dir}   DM: {dm_name}\n"
            f"  Yanlis AE ile uretim yapiliyor olabilir.")
    if ae_cfg.lead_independent != (m1cfg.modulation == "ltcm"
                                   and "flat" not in ae_dir):
        pass  # bilgi amacli; zorunlu degil
    print(f"  DM yuklendi   : step={ck.get('step')}  val={ck.get('val'):.5f}")
    print(f"  Parametrizasyon: {diff.param}  zamanlama={dcfg.schedule}")
    return ae, model, diff, n_classes, m1cfg, dcfg


# ============================================================================
def apply_scenario(cond: dict, scenario: str, n_classes: int,
                   rng: np.random.Generator, device):
    """
    Tablo 11 senaryolarini uygular.
    Doner: (cond, d_null, a_null, g_null)
    """
    B = cond["disease"].shape[0]
    z = lambda: torch.zeros(B, device=device)
    d_null = a_null = g_null = z()

    if scenario == "baseline":
        pass
    elif scenario == "only_dl":
        a_null = torch.ones(B, device=device)
        g_null = torch.ones(B, device=device)
    elif scenario == "only_psi":
        d_null = torch.ones(B, device=device)
    elif scenario == "perturbed":
        dis = cond["disease"].clone()
        flip = torch.from_numpy(rng.random(B) < 0.30).to(device)
        if flip.any():
            idx = torch.where(flip)[0]
            wrong = torch.zeros(len(idx), n_classes, device=device)
            j = torch.from_numpy(rng.integers(0, n_classes, len(idx))).to(device)
            wrong[torch.arange(len(idx), device=device), j] = 1.0
            dis[idx] = wrong
        cond["disease"] = dis

        sex = cond["sex"].clone()
        fs_ = torch.from_numpy(rng.random(B) < 0.30).to(device)
        sex[fs_] = 1 - sex[fs_].clamp(0, 1)
        cond["sex"] = sex

        # yas +-10 yil, %30 olasilikla
        ab = cond["age_bits"].cpu().numpy()
        shift = rng.random(B) < 0.30
        delta = rng.integers(-10, 11, B)
        new_bits = []
        for i in range(B):
            a = decode_age(ab[i])
            if shift[i]:
                a = int(np.clip(a + delta[i], 1, AGE_MAX))
            new_bits.append(gray_encode(a))
        cond["age_bits"] = torch.from_numpy(
            np.stack(new_bits)).float().to(device)
    else:
        raise ValueError(scenario)
    return cond, d_null, a_null, g_null


def decode_age(bits) -> int:
    b = [int(round(float(x))) for x in bits]
    n = b[0]; out = n
    for x in b[1:]:
        n ^= x
        out = (out << 1) | n
    return out


# ============================================================================
@torch.no_grad()
def generate(args):
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    print("=" * 78)
    print("LEAD-DM | Asama 3: Kosullu Uretim")
    print("=" * 78)
    print(f"Veri seti  : {args.dataset}")
    print(f"Model      : {args.dm_name}")
    print(f"Senaryo    : {args.scenario}")
    print(f"Ornekleme  : {'DDPM-1000' if args.ddim is None else f'DDIM-{args.ddim}'}")

    ae, model, diff, n_classes, m1cfg, dcfg = load_models(
        args.dataset, args.dm_name, device)

    guidance = None
    if args.guidance_all:
        # Standart CFG: tum kosullar birlikte keskinlestirilir.
        # w = 1.0 kosullu modelin ta kendisi; w > 1 abartir.
        guidance = {"all": args.guidance_all}
        print(f"Yonlendirme (butunsel CFG): w = {args.guidance_all}")
    elif any([args.guidance_disease, args.guidance_age, args.guidance_sex]):
        guidance = {"disease": args.guidance_disease,
                    "age": args.guidance_age,
                    "sex": args.guidance_sex}
        print(f"Yonlendirme (kosul basina): {guidance}")
        print("  [i] Sifir agirlikli kosullar tamamen DUSURULUR.")

    ds = ECGDataset(args.dataset, "all")
    cond_all = ds.all_conditions()
    N_all = len(ds)

    # --limit: bastan kesmek yerine BOLME ORANLARINI KORUYAN rastgele
    # alt orneklem al. Boylece yonlendirme taramasi gibi hizli denemelerde
    # train/val/test dengesi bozulmaz.
    if args.limit > 0 and args.limit < N_all:
        sel = []
        for sp, grp in ds.meta.groupby("split"):
            k = max(1, int(round(args.limit * len(grp) / N_all)))
            idx = grp.index.to_numpy()
            sel.append(rng.choice(idx, size=min(k, len(idx)), replace=False))
        sel = np.sort(np.concatenate(sel))
        print(f"Alt orneklem: {len(sel)}/{N_all} "
              f"(bolme oranlari korunarak)")
    else:
        sel = np.arange(N_all)
    N = len(sel)
    print(f"Uretilecek : {N} kayit (gercek etiketlerle eslesmis)")

    C = model.channels
    T = ae.cfg.latent_len
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}__{args.dm_name}"
    if args.scenario != "baseline":
        tag += f"__{args.scenario}"
    if guidance:
        tag += (f"__gall{args.guidance_all}" if args.guidance_all
                else f"__g{args.guidance_disease}")
    if args.limit > 0:
        tag += f"__n{args.limit}"

    import h5py
    h5 = h5py.File(GEN_DIR / f"{tag}.h5", "w")
    dset = h5.create_dataset("signals", shape=(0, 12, SIG_LEN),
                             maxshape=(None, 12, SIG_LEN), dtype="float32",
                             chunks=(min(64, args.batch), 12, SIG_LEN),
                             compression="gzip", compression_opts=4)

    rows, labs, written = [], [], 0
    t_start = time.time()
    _mask_checked = False

    for i0 in range(0, N, args.batch):
        i1 = min(i0 + args.batch, N)
        idx = sel[i0:i1]
        cond = {
            "disease": cond_all["disease"][idx].to(device),
            "age_bits": cond_all["age_bits"][idx].to(device),
            "sex": cond_all["sex"][idx].to(device),
        }
        cond, dn, an, gn = apply_scenario(dict(cond), args.scenario,
                                          n_classes, rng, device)

        # Senaryo maskeleri difuzyona DOGRUDAN verilir.
        # (Onceki sarmalayici cozumu calismiyordu: _guided_out maskeleri
        #  None degil SIFIR TENSORU olarak gecirdigi icin ikame hic
        #  devreye girmiyor ve only_dl / only_psi tam kosullu uretime
        #  donusuyordu.)
        null_mask = {"d": dn, "a": an, "g": gn}

        # Ilk parti icin maskelerin GERCEKTEN uygulandigini dogrula:
        # maskeli kosulun gommesi tum orneklerde AYNI olmali.
        if not _mask_checked:
            _mask_checked = True
            with torch.no_grad():
                t0_ = torch.zeros(len(idx), dtype=torch.long, device=device)
                ctx = model.cond(cond["disease"], cond["age_bits"],
                                 cond["sex"], t0_, dn, an, gn)
            for nm_, key, lbl in (("d", "e_d", "hastalik"),
                                  ("a", "e_a", "yas"),
                                  ("g", "e_g", "cinsiyet")):
                masked = float(null_mask[nm_].mean()) > 0.5
                e = ctx[key]
                uniform = bool(torch.allclose(e, e[:1].expand_as(e),
                                              atol=1e-5))
                if masked and not uniform:
                    raise RuntimeError(
                        f"Senaryo maskesi UYGULANMADI: {lbl} dusurulmus "
                        f"olmali ama gommeler ornekler arasi farkli.")
                if masked:
                    print(f"  [OK] {lbl} kosulu dusuruldu (dogrulandi)")

        z = diff.sample(model, cond, (len(idx), C, T), device,
                        guidance=guidance, ddim_steps=args.ddim,
                        progress=False, null_mask=null_mask)
        x8 = ae.decode_scaled(z)
        x12 = leads8_to_12(denormalize(x8)).float().cpu().numpy()

        dset.resize(written + len(idx), axis=0)
        dset[written:written + len(idx)] = x12
        written += len(idx)

        for k, gi in enumerate(idx):
            rows.append({
                "idx": int(gi),
                "record": f"GEN_{ds.meta.iloc[gi]['record']}",
                "parent_record": ds.meta.iloc[gi]["parent_record"],
                "source_record": ds.meta.iloc[gi]["record"],
                "age": float(ds.meta.iloc[gi]["age"]),
                "sex": int(ds.sex[gi]),
                "split": ds.meta.iloc[gi]["split"],
                "labels": ds.meta.iloc[gi]["labels"],
                "max_abs": float(np.abs(x12[k]).max()),
            })
            labs.append(ds.labels[gi])

        if (i0 // args.batch) % 10 == 0:
            el = time.time() - t_start
            eta = el / max(written, 1) * (N - written)
            print(f"  {written:>6}/{N}  ({el/60:.1f} dk gecti, "
                  f"~{eta/60:.1f} dk kaldi)")

    total_time = time.time() - t_start
    h5.attrs.update({"fs": FS, "seconds": 10, "length": SIG_LEN, "units": "mV",
                     "lead_order": ",".join(CANON_12),
                     "classes": ",".join(DATASET_CLASSES[args.dataset]),
                     "source": args.dataset, "model": args.dm_name,
                     "scenario": args.scenario,
                     "sampler": "ddpm" if args.ddim is None else f"ddim{args.ddim}"})
    h5.close()

    meta = pd.DataFrame(rows)
    meta.to_csv(GEN_DIR / f"{tag}_meta.csv", index=False)
    np.save(GEN_DIR / f"{tag}_labels.npy", np.stack(labs).astype(np.uint8))

    print(f"\nUretim tamamlandi: {written} kayit, {total_time/60:.1f} dk")
    print(f"  bolme: {meta['split'].value_counts().to_dict()}")

    # ---------------- Cikarim suresi (Tablo 5) ----------------
    print("\nCikarim suresi olcumu (tek 12-derivasyon EKG)...")
    single = {k: v[:1] for k, v in cond.items()}
    for _ in range(2):                      # isinma
        diff.sample(model, single, (1, C, T), device,
                    ddim_steps=args.ddim, progress=False)
    if device.type == "cuda":
        torch.cuda.synchronize()
    reps = args.time_reps
    t0 = time.time()
    for _ in range(reps):
        z1 = diff.sample(model, single, (1, C, T), device,
                         ddim_steps=args.ddim, progress=False)
        _ = ae.decode_scaled(z1)
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_sample = (time.time() - t0) / reps
    mem = (torch.cuda.max_memory_allocated() / 1e6
           if device.type == "cuda" else float("nan"))
    print(f"  {per_sample:.3f} s/EKG   (batch=1, {reps} tekrar)")
    print(f"  batch={args.batch} ile: {total_time/max(written,1):.4f} s/EKG")

    write_report(args, tag, meta, per_sample, total_time, written, mem,
                 model, ae, diff, cond, C, T, device, ds)
    return tag


# ============================================================================
def write_report(args, tag, meta, per_sample, total_time, written, mem,
                 model, ae, diff, cond, C, T, device, ds):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    figdir = RESULTS_DIR / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    classes = DATASET_CLASSES[args.dataset]

    n_par = sum(p.numel() for p in model.parameters())
    rep = ExcelReport(RESULTS_DIR / f"GEN_{tag}.xlsx",
                      meta={"Asama": "3 - Kosullu uretim",
                            "Veri seti": args.dataset,
                            "Model": args.dm_name,
                            "Senaryo": args.scenario,
                            "Ornekleyici": ("DDPM-1000" if args.ddim is None
                                            else f"DDIM-{args.ddim}"),
                            "Uretilen kayit": written,
                            "Toplam sure (dk)": round(total_time / 60, 2)})

    # Tablo 5 formatinda verimlilik
    dfE = pd.DataFrame([{
        "Model": f"LEAD-DM/T ({args.dm_name})",
        "Params(M)": n_par / 1e6,
        "Memory(MB)": mem,
        "Inference(s)": per_sample,
        "Batch_throughput(s/ECG)": total_time / max(written, 1),
    }])
    rep.add_table("T5_verimlilik", dfE,
                  caption=("Tablo 5 formati - cikarim suresi tek EKG icin "
                           "(batch=1). Referans: CDM-DL-PSI 12.0 s, "
                           "SSSD-ECG 26 s, DSAT-ECG 32 s, FlowECG 6 s."))

    # Uretilen verinin ozeti
    L = np.load(GEN_DIR / f"{tag}_labels.npy")
    dfS = pd.DataFrame([{"Split": k, "N": int(v)}
                        for k, v in meta["split"].value_counts().items()])
    rep.add_table("bolme", dfS, caption="Uretilen veri setinin bolunmesi.")
    dfL = pd.DataFrame([{"Class": c, "N": int(L[:, j].sum())}
                        for j, c in enumerate(classes)])
    rep.add_table("etiket_dagilimi", dfL,
                  caption="Uretilen veride etiket dagilimi (gercekle ayni).")
    dfA = pd.DataFrame([{"Metric": "max|x| medyan (mV)",
                         "Value": float(meta["max_abs"].median())},
                        {"Metric": "max|x| p99 (mV)",
                         "Value": float(meta["max_abs"].quantile(0.99))},
                        {"Metric": "max|x| > 8 mV sayisi",
                         "Value": int((meta["max_abs"] > 8).sum())}])
    rep.add_table("genlik", dfA,
                  caption="Uretilen sinyallerin genlik saglik kontrolu.")
    print(f"[Excel] {rep.save()}")

    # ---- Sekiller: her sinif icin bir ornek ----
    import h5py
    with h5py.File(GEN_DIR / f"{tag}.h5", "r") as f:
        for j, c in enumerate(classes):
            hit = np.where(L[:, j] > 0)[0]
            if len(hit) == 0:
                continue
            k = int(hit[0])
            x12 = np.asarray(f["signals"][k])
            age = meta.iloc[k]["age"]
            sex = "E" if meta.iloc[k]["sex"] == 0 else "K"
            plot_ecg_12lead(
                x12, figdir / f"gen_{tag}_{c}", fs=FS,
                title=f"Uretilen EKG | kosul: {c}, {sex}, {age:.0f} yas")
    print(f"[Sekil] {figdir}")


# ============================================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ptbxl", choices=list(DATASET_CLASSES))
    p.add_argument("--dm-name", default="M1_leadtcn")
    p.add_argument("--scenario", default="baseline",
                   choices=["baseline", "only_dl", "only_psi", "perturbed"])
    p.add_argument("--ddim", type=int, default=None,
                   help="None -> tam DDPM (1000 adim). 250 iyi bir denge.")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--guidance-all", type=float, default=0.0,
                   help="Standart CFG agirligi (tum kosullar birlikte). "
                        "1.0 = kosullu model, >1 keskinlestirir. ONERILEN.")
    p.add_argument("--guidance-disease", type=float, default=0.0,
                   help="Bilesimsel mod: YALNIZCA hastalik acik. Yas/cinsiyet "
                        "agirligi verilmezse DUSURULUR.")
    p.add_argument("--guidance-age", type=float, default=0.0)
    p.add_argument("--guidance-sex", type=float, default=0.0)
    p.add_argument("--time-reps", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    generate(p.parse_args())


if __name__ == "__main__":
    main()
