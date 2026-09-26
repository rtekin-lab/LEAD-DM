#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Asama 4b : Hastalik teshis duzeyi degerlendirme  (Tablo 4 / 7)
===========================================================================

Referans makalenin protokolu (Bolum 4.5) birebir uygulanir:

    Baseline   : R-Train ile egit -> R-Test'te test et       (referans nokta)
    Sadakat    : R-Train ile egit -> G-Test'te test et       (fidelity)
    Cesitlilik : G-Train ile egit -> R-Test'te test et       (diversity)

Mantik:
  * Uretilen sinyaller gercege benziyorsa, gercek veriyle egitilmis
    siniflandirici sentetik testte de baseline'a yakin sonuc vermeli.
  * Sentetik veriyle egitilmis model gercek testte iyi calisiyorsa,
    uretilen sinyaller gercek dagilimin genis bir kismini kapsiyor demektir.

Siniflandiricilar
-----------------
  xresnet1d50    (~0.75 M)   ana siniflandirici, ablasyonlarda sabitlenir
  inceptiontime  (~0.47 M)
  resnet1d       (kucuk kontrol)

METRIK NOTU
-----------
Referans makale "Acc" tanimini vermiyor. Cok etiketli gorevde birden fazla
makul tanim var; ucunu birden raporluyoruz ve hangisini kullandigimizi
makalede acikca yaziyoruz:
    subset_acc   tum etiketlerin ayni anda dogru olmasi (en kati)
    label_acc    etiket bazinda dogruluk (esik 0.5)
    F1 (makro)   sinif basina F1'in ortalamasi
    AUC (makro)  sinif basina ROC-AUC'un ortalamasi

Kullanim:
    python evaluate.py --dataset ptbxl --gen ptbxl__M1_leadtcn
    python evaluate.py --dataset ptbxl --gen ptbxl__M1_leadtcn \\
        --classifiers xresnet1d50 inceptiontime
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
                            DATASET_CLASSES, SIG_LEN, SIGNAL_SCALE,
                            require_files)
from lead_dm.reporting import (ExcelReport, setup_style, save_figure, PALETTE,
                               add_spec_table)

# Yollar lead_dm/config.py'den gelir (tek kaynak).
GEN_DIR = GENERATED_DIR
PREP_DIR = PREPARED_DIR


# ============================================================================
# SINIFLANDIRICILAR
# ============================================================================

def conv_layer(ni, nf, ks=3, stride=1, act=True, zero_bn=False):
    layers = [nn.Conv1d(ni, nf, ks, stride=stride, padding=ks // 2, bias=False),
              nn.BatchNorm1d(nf)]
    nn.init.constant_(layers[1].weight, 0.0 if zero_bn else 1.0)
    if act:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class XResBlock1d(nn.Module):
    """xresnet bottleneck blogu (expansion=4)."""

    def __init__(self, expansion, ni, nh, stride=1, ks=5):
        super().__init__()
        nf, ni_ = nh * expansion, ni * expansion
        if expansion == 1:
            layers = [conv_layer(ni_, nh, ks, stride=stride),
                      conv_layer(nh, nf, ks, zero_bn=True, act=False)]
        else:
            layers = [conv_layer(ni_, nh, 1),
                      conv_layer(nh, nh, ks, stride=stride),
                      conv_layer(nh, nf, 1, zero_bn=True, act=False)]
        self.convs = nn.Sequential(*layers)
        self.idconv = (nn.Identity() if ni_ == nf
                       else conv_layer(ni_, nf, 1, act=False))
        self.pool = (nn.Identity() if stride == 1
                     else nn.AvgPool1d(2, ceil_mode=True))

    def forward(self, x):
        return F.relu(self.convs(x) + self.idconv(self.pool(x)))


class XResNet1d(nn.Module):
    """
    Strodthoff ve ark.'nin PTB-XL kiyaslama deposundaki xresnet1d
    mimarisinin sadelestirilmis ama yapisal olarak esdeger uygulamasi.
    xresnet1d50 -> expansion=4, layers=[3,4,6,3]
    """

    def __init__(self, expansion, layers, n_in=12, n_out=5, ks=5,
                 stem_szs=(6, 6, 16), width=13):
        """
        width : taban kanal sayisi. Referans makale Xresnet1d50 icin 0.75 M
                parametre bildiriyor; fastai'nin tam genisligi (64 taban)
                1B'de 18.5 M verir. stem=(6,6,16), width=13 -> 0.775 M.
                Bu, Strodthoff ve ark.'nin PTB-XL kiyaslamasindaki
                olceklendirmeyle uyumludur.
        """
        super().__init__()
        # KRITIK KISIT (fastai xresnet):  stem_out = block_szs[0] * expansion
        # Ilk XResBlock girisi ni*expansion kanal bekler; govde ciktisi bunu
        # tam karsilamalidir. Genisligi olceklerken bu bagi koparirsak
        # "expected 12 channels, but got 13" hatasi alinir.
        assert stem_szs[-1] % expansion == 0, (
            f"stem cikisi ({stem_szs[-1]}) expansion'a ({expansion}) "
            f"bolunebilmeli")
        szs = [n_in, *stem_szs]
        stem = [conv_layer(szs[i], szs[i + 1], ks, stride=2 if i == 0 else 1)
                for i in range(3)]
        block_szs = [stem_szs[-1] // expansion, width, width * 2,
                     width * 4, width * 8][:len(layers) + 1]
        blocks = [self._make_layer(expansion, block_szs[i], block_szs[i + 1],
                                   n_blocks=l, stride=1 if i == 0 else 2, ks=ks)
                  for i, l in enumerate(layers)]
        self.backbone = nn.Sequential(
            *stem, nn.MaxPool1d(3, stride=2, padding=1), *blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.BatchNorm1d(block_szs[-1] * expansion), nn.Dropout(0.5),
            nn.Linear(block_szs[-1] * expansion, n_out))

    @staticmethod
    def _make_layer(expansion, ni, nf, n_blocks, stride, ks):
        return nn.Sequential(*[
            XResBlock1d(expansion, ni if i == 0 else nf, nf,
                        stride if i == 0 else 1, ks)
            for i in range(n_blocks)])

    def forward(self, x):
        return self.head(self.backbone(x))


class InceptionModule1d(nn.Module):
    def __init__(self, ni, nf=32, ks=(39, 19, 9), bottleneck=32):
        super().__init__()
        self.bottleneck = (nn.Conv1d(ni, bottleneck, 1, bias=False)
                           if ni > 1 else nn.Identity())
        nb = bottleneck if ni > 1 else ni
        self.convs = nn.ModuleList(
            [nn.Conv1d(nb, nf, k, padding=k // 2, bias=False) for k in ks])
        self.mp_conv = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(ni, nf, 1, bias=False))
        self.bn = nn.BatchNorm1d(nf * (len(ks) + 1))

    def forward(self, x):
        b = self.bottleneck(x)
        out = torch.cat([c(b) for c in self.convs] + [self.mp_conv(x)], dim=1)
        return F.relu(self.bn(out))


class InceptionTime1d(nn.Module):
    def __init__(self, n_in=12, n_out=5, nf=32, depth=6):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.shortcuts = nn.ModuleList()
        ni = n_in
        for d in range(depth):
            self.blocks.append(InceptionModule1d(ni, nf))
            ni = nf * 4
            if d % 3 == 2:
                self.shortcuts.append(nn.Sequential(
                    nn.Conv1d(n_in if d == 2 else nf * 4, nf * 4, 1, bias=False),
                    nn.BatchNorm1d(nf * 4)))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(nf * 4, n_out))

    def forward(self, x):
        res = x
        si = 0
        for d, blk in enumerate(self.blocks):
            x = blk(x)
            if d % 3 == 2:
                x = F.relu(x + self.shortcuts[si](res))
                res = x
                si += 1
        return self.head(x)


def build_classifier(name: str, n_out: int, n_in: int = 12) -> nn.Module:
    name = name.lower()
    # Genislikler referans makalenin bildirdigi parametre sayilarina
    # gore ayarlanmistir: Xresnet1d50 0.75 M, Inceptiontime 0.47 M
    if name == "xresnet1d50":
        return XResNet1d(4, [3, 4, 6, 3], n_in, n_out,
                         stem_szs=(6, 6, 16), width=13)     # 0.775 M
    if name == "xresnet1d101":
        return XResNet1d(4, [3, 4, 23, 3], n_in, n_out,
                         stem_szs=(6, 6, 16), width=13)
    if name == "inceptiontime":
        return InceptionTime1d(n_in, n_out)                 # ~0.50 M
    if name == "resnet1d":
        return XResNet1d(1, [2, 2, 2, 2], n_in, n_out,
                         stem_szs=(8, 8, 16), width=16)
    raise ValueError(name)


# ============================================================================
# VERI
# ============================================================================

class ArrayECG(Dataset):
    """HDF5'ten dogrudan okur. Sinyal olcegi egitim veri hattiyla ayni."""

    def __init__(self, h5_path, rows, labels, augment=False):
        self.h5_path = str(h5_path)
        self.rows = np.asarray(rows)
        self.labels = labels.astype(np.float32)
        self.augment = augment
        self._f = None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._f is None:
            import h5py
            self._f = h5py.File(self.h5_path, "r")
        gi = int(self.rows[i])
        x = np.asarray(self._f["signals"][gi], dtype=np.float32) * SIGNAL_SCALE
        if self.augment:
            sh = np.random.randint(-50, 51)
            if sh:
                x = np.roll(x, sh, axis=-1)
            x = x * np.float32(np.random.uniform(0.9, 1.1))
        return torch.from_numpy(np.ascontiguousarray(x)), \
            torch.from_numpy(self.labels[i])


def get_split(kind: str, dataset: str, gen_tag: str, split: str):
    """kind: 'real' | 'gen'"""
    d = PREP_DIR if kind == "real" else GEN_DIR
    stem = dataset if kind == "real" else gen_tag
    require_files(d, stem, "Gercek" if kind == "real" else "Uretilmis")
    meta = pd.read_csv(d / f"{stem}_meta.csv")
    labels = np.load(d / f"{stem}_labels.npy")
    rows = np.where((meta["split"].astype(str) == split).to_numpy())[0]
    return d / f"{stem}.h5", rows, labels[rows]


# ============================================================================
# METRIKLER
# ============================================================================

def bootstrap_ci(y_true: np.ndarray, y_prob: np.ndarray, n_boot=1000,
                 alpha=0.05, seed=0) -> dict:
    """
    Test seti uzerinde bootstrap ile %95 guven araligi.

    NEDEN GEREKLI: ablasyonlar tek kosu ve test seti kucuk (~600 ornek).
    Makro F1'de 2-3 puanlik farklar gurultu olabilir. Guven araligi
    olmadan "A daha iyi" demek savunulamaz. Bootstrap, MODELI YENIDEN
    EGITMEDEN test-seti belirsizligini verir (kosu-arasi degiskenligi
    vermez; onun icin farkli tohumlarla tekrar gerekir).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    acc, f1s, aucs = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, yp = y_true[idx], y_prob[idx]
        if yt.sum() == 0:
            continue
        m = compute_metrics(yt, yp)
        acc.append(m["subset_acc"]); f1s.append(m["F1"])
        if not np.isnan(m["AUC"]):
            aucs.append(m["AUC"])
    q = lambda v: (float(np.percentile(v, 100 * alpha / 2)),
                   float(np.percentile(v, 100 * (1 - alpha / 2)))) if v else (
                       float("nan"), float("nan"))
    lo_a, hi_a = q(acc); lo_f, hi_f = q(f1s); lo_u, hi_u = q(aucs)
    return {"Acc_lo": lo_a * 100, "Acc_hi": hi_a * 100,
            "F1_lo": lo_f * 100, "F1_hi": hi_f * 100,
            "AUC_lo": lo_u * 100, "AUC_hi": hi_u * 100,
            "F1_ci_width": (hi_f - lo_f) * 100}


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, thr=0.5) -> dict:
    from sklearn.metrics import f1_score, roc_auc_score
    y_pred = (y_prob >= thr).astype(int)
    out = {}
    out["subset_acc"] = float((y_pred == y_true).all(axis=1).mean())
    out["label_acc"] = float((y_pred == y_true).mean())
    out["F1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    aucs = []
    for j in range(y_true.shape[1]):
        if 0 < y_true[:, j].sum() < len(y_true):
            aucs.append(roc_auc_score(y_true[:, j], y_prob[:, j]))
    out["AUC"] = float(np.mean(aucs)) if aucs else float("nan")
    return out


# ============================================================================
# EGITIM / TEST
# ============================================================================

def train_classifier(name, train_h5, train_rows, train_lab,
                     val_h5, val_rows, val_lab, n_out, args, device, tag=""):
    """Doner: (model, parametre_sayisi, en_iyi_val_metrikleri)"""
    model = build_classifier(name, n_out).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    tr = DataLoader(ArrayECG(train_h5, train_rows, train_lab, augment=True),
                    batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True,
                    pin_memory=device.type == "cuda")
    va = DataLoader(ArrayECG(val_h5, val_rows, val_lab),
                    batch_size=args.batch, shuffle=False,
                    num_workers=max(1, args.workers // 2))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * max(len(tr), 1))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    best_auc, best_state, best_m = -1.0, None, {}
    for ep in range(args.epochs):
        model.train()
        for x, y in tr:
            x, y = x.to(device, non_blocking=True), y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                loss = F.binary_cross_entropy_with_logits(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
        yp, yt = predict(model, va, device)
        m = compute_metrics(yt, yp)
        if m["AUC"] > best_auc:
            best_auc = m["AUC"]
            best_m = dict(m)
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
        if (ep + 1) % max(1, args.epochs // 5) == 0:
            print(f"      ep {ep+1:>2}/{args.epochs}  "
                  f"val AUC={m['AUC']*100:.2f}  F1={m['F1']*100:.2f}")
    if best_state:
        model.load_state_dict(best_state)
    return model, n_par, best_m


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    P, Y = [], []
    for x, y in loader:
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device))).float()
        P.append(p.cpu().numpy()); Y.append(y.numpy())
    return np.concatenate(P), np.concatenate(Y)


def test_on(model, h5, rows, lab, args, device, save_as: Path | None = None,
            with_ci: bool = True):
    dl = DataLoader(ArrayECG(h5, rows, lab), batch_size=args.batch,
                    shuffle=False, num_workers=max(1, args.workers // 2))
    yp, yt = predict(model, dl, device)
    m = compute_metrics(yt, yp)
    if with_ci:
        m.update(bootstrap_ci(yt, yp, n_boot=args.n_boot))
    if save_as is not None:
        save_as.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(save_as, y_true=yt, y_prob=yp)
    return m


# ============================================================================
# ANA AKIS
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ptbxl", choices=list(DATASET_CLASSES))
    ap.add_argument("--gen", required=True)
    ap.add_argument("--classifiers", nargs="+", default=["xresnet1d50"],
                    help="xresnet1d50 inceptiontime resnet1d xresnet1d18")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap tekrar sayisi (guven araligi icin)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    classes = DATASET_CLASSES[args.dataset]
    n_out = len(classes)

    print("=" * 78)
    print("LEAD-DM | Asama 4b: Hastalik Teshis Duzeyi Degerlendirme")
    print("=" * 78)
    print(f"Veri seti : {args.dataset}  ({n_out} sinif)")
    print(f"Uretilen  : {args.gen}")

    # veri bolumleri
    R = {s: get_split("real", args.dataset, args.gen, s)
         for s in ("train", "val", "test")}
    G = {s: get_split("gen", args.dataset, args.gen, s)
         for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        print(f"  {s:<6}: gercek {len(R[s][1]):>6}   uretilen {len(G[s][1]):>6}")

    rows, t0 = [], time.time()
    for cname in args.classifiers:
        print(f"\n{'='*78}\n[{cname}]\n{'='*78}")

        # --- Baseline + Sadakat : R-Train ile egit ---
        print("  R-Train ile egitiliyor...")
        mR, npar, mR_val = train_classifier(
            cname, R["train"][0], R["train"][1], R["train"][2],
            R["val"][0], R["val"][1], R["val"][2], n_out, args, device)
        pd_dir = RESULTS_DIR / "predictions"
        base = test_on(mR, *R["test"], args, device,
                       save_as=pd_dir / f"{args.gen}__{cname}__base.npz")
        fid = test_on(mR, *G["test"], args, device,
                      save_as=pd_dir / f"{args.gen}__{cname}__fid.npz")
        ci = lambda m: (f"F1={m['F1']*100:.2f} "
                        f"[{m['F1_lo']:.2f}-{m['F1_hi']:.2f}]")
        print(f"    Baseline (R->R): Acc={base['subset_acc']*100:.2f} "
              f"{ci(base)} AUC={base['AUC']*100:.2f}")
        print(f"    Sadakat  (R->G): Acc={fid['subset_acc']*100:.2f} "
              f"{ci(fid)} AUC={fid['AUC']*100:.2f}")

        # --- Cesitlilik : G-Train ile egit ---
        print("  G-Train ile egitiliyor...")
        mG, _, mG_val = train_classifier(
            cname, G["train"][0], G["train"][1], G["train"][2],
            G["val"][0], G["val"][1], G["val"][2], n_out, args, device)
        div = test_on(mG, *R["test"], args, device,
                      save_as=pd_dir / f"{args.gen}__{cname}__div.npz")
        print(f"    Cesitlilik (G->R): Acc={div['subset_acc']*100:.2f} "
              f"{ci(div)} AUC={div['AUC']*100:.2f}")
        print(f"    (%95 GA genisligi: sadakat F1 +-{fid['F1_ci_width']/2:.2f} pt, "
              f"cesitlilik F1 +-{div['F1_ci_width']/2:.2f} pt)")

        # --- SAGLIK KONTROLLERI ---
        # Sadakat baseline'i GECEMEZ. Geciyorsa uretilen sinyaller etiket
        # kisayolu tasiyor demektir: her sinif icin abartilmis prototip.
        # Ayni sekilde G-egitimli siniflandiricinin G-val'de gercek veriye
        # gore cok daha yuksek AUC almasi, sentetik verinin "fazla kolay"
        # oldugunu gosterir (asiri kosullandirma).
        warn = []
        if fid["AUC"] > base["AUC"] + 0.01:
            warn.append(f"Sadakat AUC baseline'i asiyor "
                        f"({fid['AUC']*100:.2f} > {base['AUC']*100:.2f}) "
                        "-> ETIKET KISAYOLU / asiri yonlendirme supheli")
        if fid["F1"] > base["F1"] + 0.02:
            warn.append("Sadakat F1 baseline'i asiyor -> ayni suphe")
        sep = mG_val.get("AUC", float("nan")) - mR_val.get("AUC", float("nan"))
        if sep > 0.04:
            warn.append(f"Uretilen veri gercekten COK daha ayirt edilebilir "
                        f"(G-val AUC {mG_val['AUC']*100:.2f} vs "
                        f"R-val {mR_val['AUC']*100:.2f}) -> prototip cokmesi")
        for w in warn:
            print(f"    [!!] {w}")
        if not warn:
            print("    [OK] saglik kontrolleri gecti")

        rows.append({
            "Dataset": args.dataset, "Classifier": cname,
            "Rval_AUC(%)": mR_val.get("AUC", float("nan")) * 100,
            "Gval_AUC(%)": mG_val.get("AUC", float("nan")) * 100,
            "Gval_minus_Rval": sep * 100,
            "Uyari": " | ".join(warn) if warn else "",
            "Params(M)": npar / 1e6, "Model": args.gen,
            "Base_Acc(%)": base["subset_acc"] * 100,
            "Base_F1(%)": base["F1"] * 100, "Base_AUC(%)": base["AUC"] * 100,
            "Fid_Acc(%)": fid["subset_acc"] * 100,
            "Fid_F1(%)": fid["F1"] * 100, "Fid_AUC(%)": fid["AUC"] * 100,
            "Div_Acc(%)": div["subset_acc"] * 100,
            "Div_F1(%)": div["F1"] * 100, "Div_AUC(%)": div["AUC"] * 100,
            "Fid_F1_lo": fid["F1_lo"], "Fid_F1_hi": fid["F1_hi"],
            "Div_F1_lo": div["F1_lo"], "Div_F1_hi": div["F1_hi"],
            "Fid_AUC_lo": fid["AUC_lo"], "Fid_AUC_hi": fid["AUC_hi"],
            "Div_AUC_lo": div["AUC_lo"], "Div_AUC_hi": div["AUC_hi"],
            "Base_labelAcc(%)": base["label_acc"] * 100,
            "Fid_labelAcc(%)": fid["label_acc"] * 100,
            "Div_labelAcc(%)": div["label_acc"] * 100,
        })

    df = pd.DataFrame(rows)
    # Kendi baseline'imiza gore normalize oranlar (referansla kiyas icin)
    for pre in ("Fid", "Div"):
        for met in ("Acc(%)", "F1(%)", "AUC(%)"):
            df[f"{pre}/Base_{met.split('(')[0]}"] = (
                df[f"{pre}_{met}"] / df[f"Base_{met}"])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rep = ExcelReport(RESULTS_DIR / f"EVAL_{args.gen}.xlsx",
                      meta={"Asama": "4b - hastalik teshis duzeyi",
                            "Veri seti": args.dataset, "Uretilen": args.gen,
                            "Siniflandirici epoch": args.epochs,
                            "Acc tanimi": "subset accuracy (tum etiketler dogru)",
                            "F1/AUC": "makro ortalama",
                            "Sure (dk)": round((time.time() - t0) / 60, 1)})
    add_spec_table(rep, "T4_diagnosis",
                   df[["Dataset", "Classifier", "Model",
                       "Fid_Acc(%)", "Fid_F1(%)", "Fid_AUC(%)",
                       "Div_Acc(%)", "Div_F1(%)", "Div_AUC(%)"]],
                   sheet="T4_teshis")
    rep.add_table("tam_sonuclar", df,
                  caption="Baseline dahil tum metrikler (subset ve label acc).")
    print(f"\n[Excel] {rep.save()}")

    # sekil
    plt = setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    metrics = [("Acc(%)", "subset acc"), ("F1(%)", "F1 (makro)"),
               ("AUC(%)", "AUC (makro)")]
    x = np.arange(len(df)); w = 0.26
    for i, (suf, lbl) in enumerate(metrics):
        ax = axes[i]
        ax.bar(x - w, df[f"Base_{suf}"], w, label="Baseline", color=PALETTE[7])
        ax.bar(x, df[f"Fid_{suf}"], w, label="Sadakat", color=PALETTE[0])
        ax.bar(x + w, df[f"Div_{suf}"], w, label="Cesitlilik", color=PALETTE[1])
        ax.set_xticks(x); ax.set_xticklabels(df["Classifier"], rotation=20,
                                             ha="right")
        ax.set_ylabel(lbl); ax.grid(True, axis="y")
        ax.set_title(f"({chr(97+i)}) {lbl}", loc="left")
        if i == 0:
            ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, RESULTS_DIR / "figures" / f"eval_{args.gen}")
    print(f"[Sekil] {RESULTS_DIR/'figures'}")


if __name__ == "__main__":
    main()
