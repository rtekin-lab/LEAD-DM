#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Asama 4c : Fizyolojik oznitelik tutarliligi  (Tablo 13)
====================================================================
Bu, calismanin ANA IDDIASINI dogrudan sinayan tek deney:
"verilen yas ve cinsiyet kosulu uretilen sinyale gercekten yansiyor mu?"

Protokol (referans makale Bolum 5.6 ile ayni):
  * GERCEK egitim verisiyle iki model egitilir
        - yas regresyonu      -> MAE
        - cinsiyet siniflandirmasi -> Acc
  * Ayni modeller GERCEK test ve URETILEN test uzerinde olculur
  * Fark kucukse, uretilen sinyaller verilen kosulu tasiyor demektir

Referans degerler (CDM-DL-PSI):
    PTB-XL   yas MAE  7.31 -> 7.70   cinsiyet Acc 83.73 -> 82.72
    CPSC2018 yas MAE  8.40 -> 8.80   cinsiyet Acc 82.68 -> 79.48

Kullanim:
    python physio_check.py --dataset ptbxl --gen ptbxl__M1_long__gall1.25
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from lead_dm.config import (RESULTS_DIR, PREPARED_DIR, GENERATED_DIR,
                            DATASET_CLASSES, SIGNAL_SCALE, AGE_MAX,
                            require_files)
from lead_dm.reporting import (ExcelReport, setup_style, save_figure, PALETTE,
                               add_spec_table)
from evaluate import build_classifier    # TEK KAYNAK: evaluate.py

GEN_DIR = GENERATED_DIR
PREP_DIR = PREPARED_DIR


# ============================================================================
class PhysioECG(Dataset):
    """Hedef: yas (yil) ve cinsiyet (0/1). Cinsiyeti bilinmeyen kayitlar atilir."""

    def __init__(self, h5_path, meta: pd.DataFrame, rows, augment=False):
        self.h5_path = str(h5_path)
        keep = [i for i in rows if int(meta.iloc[i]["sex"]) in (0, 1)]
        self.rows = np.asarray(keep)
        self.age = meta.iloc[self.rows]["age"].to_numpy(dtype=np.float32)
        self.sex = meta.iloc[self.rows]["sex"].to_numpy(dtype=np.float32)
        self.augment = augment
        self._f = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._f is None:
            import h5py
            self._f = h5py.File(self.h5_path, "r")
        x = np.asarray(self._f["signals"][int(self.rows[i])],
                       dtype=np.float32) * SIGNAL_SCALE
        if self.augment:
            sh = np.random.randint(-50, 51)
            if sh:
                x = np.roll(x, sh, axis=-1)
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.tensor(self.age[i] / AGE_MAX),      # [0,1]'e olcekle
                torch.tensor(self.sex[i]))


def get_rows(kind, dataset, gen_tag, split):
    d = PREP_DIR if kind == "real" else GEN_DIR
    stem = dataset if kind == "real" else gen_tag
    require_files(d, stem, "Gercek" if kind == "real" else "Uretilmis")
    meta = pd.read_csv(d / f"{stem}_meta.csv")
    rows = np.where((meta["split"].astype(str) == split).to_numpy())[0]
    return d / f"{stem}.h5", meta, rows


class PhysioNet(nn.Module):
    """
    Tek omurga, iki cikis: yas (regresyon) + cinsiyet (ikili).

    Omurga evaluate.py'nin build_classifier'indan alinir. Mimari
    parametreleri (stem genisligi, expansion vb.) BURADA TEKRARLANMAZ;
    aksi halde evaluate.py guncellendiginde sessizce uyumsuz kalir.
    """

    def __init__(self, n_in=12, backbone: str = "xresnet1d50"):
        super().__init__()
        self.backbone = build_classifier(backbone, n_out=2, n_in=n_in)

    def forward(self, x):
        o = self.backbone(x)
        return o[:, 0], o[:, 1]          # (yas_ciktisi, cinsiyet_logiti)


@torch.no_grad()
def evaluate_model(model, loader, device):
    model.eval()
    A, Ah, S, Sh = [], [], [], []
    for x, a, s in loader:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            pa, ps = model(x.to(device))
        A.append(a.numpy()); Ah.append(pa.float().cpu().numpy())
        S.append(s.numpy()); Sh.append(torch.sigmoid(ps).float().cpu().numpy())
    A = np.concatenate(A) * AGE_MAX
    Ah = np.clip(np.concatenate(Ah), 0, 1) * AGE_MAX
    S = np.concatenate(S); Sh = np.concatenate(Sh)
    from sklearn.metrics import roc_auc_score
    auc = (roc_auc_score(S, Sh) if 0 < S.sum() < len(S) else float("nan"))
    return {"age_MAE": float(np.abs(A - Ah).mean()),
            "age_RMSE": float(np.sqrt(((A - Ah) ** 2).mean())),
            "sex_Acc": float(((Sh >= 0.5).astype(np.float32) == S).mean()),
            "sex_AUC": float(auc), "n": int(len(A))}


def train_physio(h5, meta, tr_rows, va_rows, args, device):
    """Gercek veriyle yas+cinsiyet modeli egitir."""
    model = PhysioNet(backbone=args.backbone).to(device)
    tr = DataLoader(PhysioECG(h5, meta, tr_rows, augment=True),
                    batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    pin_memory=device.type == "cuda")
    va = DataLoader(PhysioECG(h5, meta, va_rows), batch_size=args.batch,
                    shuffle=False, num_workers=max(1, args.workers // 2))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(tr), 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best, best_state = 1e9, None
    for ep in range(args.epochs):
        model.train()
        for x, a, s in tr:
            x, a, s = x.to(device), a.to(device), s.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                pa, ps = model(x)
                loss = (F.l1_loss(pa, a)
                        + 0.5 * F.binary_cross_entropy_with_logits(ps, s))
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sch.step()
        m = evaluate_model(model, va, device)
        score = m["age_MAE"] - 20.0 * m["sex_Acc"]      # ikisini birlikte izle
        if score < best:
            best = score
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"    ep {ep+1:>2}/{args.epochs}  "
                  f"val yas MAE={m['age_MAE']:.2f}  "
                  f"cinsiyet Acc={m['sex_Acc']*100:.2f}")
    if best_state:
        model.load_state_dict(best_state)
    return model


# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl", choices=list(DATASET_CLASSES))
    ap.add_argument("--gen", required=True)
    ap.add_argument("--backbone", default="xresnet1d50",
                    choices=["xresnet1d50", "inceptiontime", "resnet1d"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    print("=" * 78)
    print("LEAD-DM | Asama 4c: Fizyolojik Oznitelik Tutarliligi (Tablo 13)")
    print("=" * 78)
    print(f"Veri seti : {args.dataset}")
    print(f"Uretilen  : {args.gen}")

    rh5, rmeta, r_tr = get_rows("real", args.dataset, args.gen, "train")
    _, _, r_va = get_rows("real", args.dataset, args.gen, "val")
    _, _, r_te = get_rows("real", args.dataset, args.gen, "test")
    gh5, gmeta, g_te = get_rows("gen", args.dataset, args.gen, "test")
    print(f"  gercek train/val/test : {len(r_tr)}/{len(r_va)}/{len(r_te)}")
    print(f"  uretilen test         : {len(g_te)}")

    t0 = time.time()
    print("\n  GERCEK veriyle egitiliyor (yas + cinsiyet)...")
    model = train_physio(rh5, rmeta, r_tr, r_va, args, device)

    real_m = evaluate_model(
        model, DataLoader(PhysioECG(rh5, rmeta, r_te), batch_size=args.batch,
                          num_workers=2), device)
    gen_m = evaluate_model(
        model, DataLoader(PhysioECG(gh5, gmeta, g_te), batch_size=args.batch,
                          num_workers=2), device)

    print("\n" + "=" * 78)
    print(f"{'':<12}{'yas MAE':>10}{'yas RMSE':>11}{'cinsiyet Acc':>14}"
          f"{'cinsiyet AUC':>14}")
    for n, m in (("GERCEK", real_m), ("URETILEN", gen_m)):
        print(f"{n:<12}{m['age_MAE']:>10.2f}{m['age_RMSE']:>11.2f}"
              f"{m['sex_Acc']*100:>13.2f}%{m['sex_AUC']*100:>13.2f}%")
    # Isaret kurallari (karisikligi onlemek icin acikca):
    #   d_age = uretilen - gercek   (pozitif = uretilende yas TAHMINI DAHA ZOR)
    #   d_sex = uretilen - gercek   (pozitif = uretilende cinsiyet DAHA KOLAY)
    d_age = gen_m["age_MAE"] - real_m["age_MAE"]
    d_sex = (gen_m["sex_Acc"] - real_m["sex_Acc"]) * 100
    print(f"{'FARK (ur-ger)':<12}{d_age:>+10.2f}{'':>11}{d_sex:>+13.2f}%")
    if d_age < -0.3 or d_sex > 3.0:
        print("\n  [i] Uretilen sinyallerde yas/cinsiyet GERCEKTEN DAHA KOLAY"
              " okunuyor.")
        print("      Kosullama guclu calisiyor demektir; ancak ayni zamanda")
        print("      kosul-ici degiskenligin gercek veriden bir miktar DUSUK")
        print("      oldugunu gosterir. Makalede bu sekilde yorumlanmalidir.")

    print("\nREFERANS (CDM-DL-PSI):")
    ref = {"ptbxl": (7.31, 7.70, 83.73, 82.72),
           "cpsc2018": (8.40, 8.80, 82.68, 79.48)}.get(args.dataset)
    if ref:
        print(f"  yas MAE {ref[0]:.2f} -> {ref[1]:.2f}  (fark +{ref[1]-ref[0]:.2f})")
        print(f"  cinsiyet Acc {ref[2]:.2f}% -> {ref[3]:.2f}%  "
              f"(fark -{ref[2]-ref[3]:.2f}%)")

    print("\nDEGERLENDIRME:")
    ok_age = abs(d_age) < 1.0
    ok_sex = abs(d_sex) < 5.0
    print(f"  yas MAE farki  < 1.0 yil : {'[OK]' if ok_age else '[!!]'} "
          f"({d_age:+.2f})")
    print(f"  cinsiyet farki < 5.0 pt  : {'[OK]' if ok_sex else '[!!]'} "
          f"({d_sex:+.2f})")
    if ok_age and ok_sex:
        print("  -> Uretilen sinyaller verilen YAS ve CINSIYET kosulunu tasiyor.")
        print("     Hasta ozgu bilgi kosullamasi CALISIYOR.")

    # ---------------- Rapor ----------------
    rows = []
    for task, key, unit in (("Age prediction", "age_MAE", "MAE"),
                            ("Gender classification", "sex_Acc", "Acc(%)")):
        for sig, m in (("Real ECG", real_m), ("Generated ECG", gen_m)):
            v = m[key] * (100 if key.endswith("Acc") else 1)
            rows.append({"Dataset": args.dataset, "Task": task, "Signal": sig,
                         "MAE": v if unit == "MAE" else None,
                         "Acc(%)": v if unit != "MAE" else None})
    df = pd.DataFrame(rows)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = ExcelReport(RESULTS_DIR / f"PHYSIO_{args.gen}.xlsx",
                      meta={"Asama": "4c - fizyolojik tutarlilik (Tablo 13)",
                            "Veri seti": args.dataset, "Uretilen": args.gen,
                            "Yas MAE farki (uretilen-gercek)": round(d_age, 3),
                            "Cinsiyet Acc farki (uretilen-gercek, pt)":
                                round(d_sex, 3),
                            "Sure (dk)": round((time.time() - t0) / 60, 1)})
    add_spec_table(rep, "T13_physio_consistency", df, sheet="T13_fizyolojik")
    rep.add_table("detay", pd.DataFrame([
        {"Signal": "Real", **real_m}, {"Signal": "Generated", **gen_m}]),
        caption="Tum metrikler (RMSE ve AUC dahil).")
    print(f"\n[Excel] {rep.save()}")

    plt = setup_style()
    fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.7))
    ax[0].bar([0, 1], [real_m["age_MAE"], gen_m["age_MAE"]],
              color=[PALETTE[2], PALETTE[0]], width=0.55)
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["Gercek", "Uretilen"])
    ax[0].set_ylabel("Yas MAE (yil)"); ax[0].grid(True, axis="y")
    ax[0].set_title("(a) Yas tahmini", loc="left")
    ax[1].bar([0, 1], [real_m["sex_Acc"] * 100, gen_m["sex_Acc"] * 100],
              color=[PALETTE[2], PALETTE[0]], width=0.55)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["Gercek", "Uretilen"])
    ax[1].set_ylabel("Cinsiyet dogrulugu (%)"); ax[1].set_ylim(50, 100)
    ax[1].grid(True, axis="y")
    ax[1].set_title("(b) Cinsiyet siniflandirma", loc="left")
    fig.tight_layout()
    save_figure(fig, RESULTS_DIR / "figures" / f"physio_{args.gen}")
    print(f"[Sekil] {RESULTS_DIR/'figures'}")


if __name__ == "__main__":
    main()
