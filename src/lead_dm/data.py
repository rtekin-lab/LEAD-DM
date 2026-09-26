#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Veri katmani
=========================
prepare_data.py'nin urettigi HDF5 + meta.csv + labels.npy uclusunu okur.

Cikti sozlugu:
    x       (8, 1000)  float32   olceklenmis sinyal (V1..V6, I, aVF)
    x12     (12,1000)  float32   tam 12 derivasyon (degerlendirme icin)
    disease (C,)       float32   cok-sicak hastalik vektoru
    age_bits(7,)       float32   Gray kodlu yas
    sex     ()         int64     0=erkek 1=kadin 2=bilinmiyor
    d_null  ()         float32   1 -> hastalik kosulu maskeli
    a_null  ()         float32   1 -> yas kosulu maskeli
    g_null  ()         float32   1 -> cinsiyet kosulu maskeli
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from .config import (PREPARED_DIR, DATASET_CLASSES, GEN8_IDX, CANON_12,
                     SIGNAL_SCALE, SIGNAL_CLIP, AGE_BITS, AGE_MAX,
                     SEX_MALE, SEX_FEMALE, SEX_NULL, SIG_LEN)


# ============================================================================
# KOSUL KODLAMA
# ============================================================================

def gray_encode(n: int, bits: int = AGE_BITS) -> np.ndarray:
    """
    Tamsayiyi Gray koduna cevirir.

    Gray kodunun ozelligi: ardisik sayilar TEK bit farkla temsil edilir.
    Bu, yas surekliligini korur -> 54 ve 55 yas birbirine yakin kodlanir.
    (Referans makale Bolum 4.3.1 ile ayni strateji.)
    """
    n = int(np.clip(n, 0, (1 << bits) - 1))
    g = n ^ (n >> 1)
    return np.array([(g >> i) & 1 for i in range(bits - 1, -1, -1)],
                    dtype=np.float32)


def gray_decode(bits_vec: np.ndarray) -> int:
    """Gray kodundan tamsayiya (dogrulama icin)."""
    bits = [int(b) for b in bits_vec]
    n = bits[0]
    out = n
    for b in bits[1:]:
        n = n ^ b
        out = (out << 1) | n
    return out


def encode_sex(v) -> int:
    s = str(v).strip().upper()
    if s in ("0", "M", "MALE"):
        return SEX_MALE
    if s in ("1", "F", "FEMALE"):
        return SEX_FEMALE
    return SEX_NULL


# ============================================================================
# DERIVASYON REKONSTRUKSIYONU
# ============================================================================

_RECON_FROM_8 = None   # tembel olusturulur


def build_recon_matrix(device=None, dtype=torch.float32) -> torch.Tensor:
    """
    (12, 8) matris: 8 uretilen kanaldan tam 12 derivasyonu turetir.

    GEN8 sirasi: V1,V2,V3,V4,V5,V6,I,aVF   (indeks 0..7)

    Einthoven + Goldberger:
        II  =  0.50*I + 1.0*aVF
        III = -0.50*I + 1.0*aVF
        aVR = -0.75*I - 0.5*aVF
        aVL =  0.75*I - 0.5*aVF
    """
    M = torch.zeros(12, 8, dtype=dtype)
    i_pos, avf_pos = 6, 7
    for k in range(6):                              # V1..V6 dogrudan
        M[CANON_12.index(f"V{k+1}"), k] = 1.0
    M[CANON_12.index("I"),   i_pos]   = 1.0
    M[CANON_12.index("aVF"), avf_pos] = 1.0
    M[CANON_12.index("II"),  i_pos]   =  0.50
    M[CANON_12.index("II"),  avf_pos] =  1.00
    M[CANON_12.index("III"), i_pos]   = -0.50
    M[CANON_12.index("III"), avf_pos] =  1.00
    M[CANON_12.index("aVR"), i_pos]   = -0.75
    M[CANON_12.index("aVR"), avf_pos] = -0.50
    M[CANON_12.index("aVL"), i_pos]   =  0.75
    M[CANON_12.index("aVL"), avf_pos] = -0.50
    if device is not None:
        M = M.to(device)
    return M


def leads8_to_12(x8: torch.Tensor) -> torch.Tensor:
    """(B, 8, T) -> (B, 12, T). Turev islemi, gradyan akisi korunur."""
    global _RECON_FROM_8
    if _RECON_FROM_8 is None or _RECON_FROM_8.device != x8.device \
            or _RECON_FROM_8.dtype != x8.dtype:
        _RECON_FROM_8 = build_recon_matrix(x8.device, x8.dtype)
    return torch.einsum("lk,bkt->blt", _RECON_FROM_8, x8)


# ============================================================================
# DATASET
# ============================================================================

class ECGDataset(Dataset):
    """
    HDF5 dosyasini tembel acar (DataLoader worker'lari icin guvenli).
    """

    def __init__(self,
                 dataset: str,
                 split: str,
                 prepared_dir: Path = PREPARED_DIR,
                 return_full12: bool = False,
                 cond_dropout: float = 0.0,
                 joint_dropout: float = 0.0,
                 seed: int = 0):
        assert dataset in DATASET_CLASSES, f"bilinmeyen veri seti: {dataset}"
        self.dataset = dataset
        self.split = split
        self.dir = Path(prepared_dir)
        self.h5_path = self.dir / f"{dataset}.h5"
        self.classes = DATASET_CLASSES[dataset]
        self.n_classes = len(self.classes)
        self.return_full12 = return_full12
        self.cond_dropout = float(cond_dropout)
        self.joint_dropout = float(joint_dropout)

        meta = pd.read_csv(self.dir / f"{dataset}_meta.csv")
        labels = np.load(self.dir / f"{dataset}_labels.npy")
        assert len(meta) == len(labels), "meta / labels hizasiz"

        if split != "all":
            keep = (meta["split"].astype(str) == split).to_numpy()
        else:
            keep = np.ones(len(meta), dtype=bool)

        self.rows = np.where(keep)[0]
        self.meta = meta.iloc[self.rows].reset_index(drop=True)
        self.labels = labels[self.rows].astype(np.float32)

        # Kosullari onceden kodla (tekrar hesaplamamak icin)
        self.age_bits = np.stack(
            [gray_encode(int(round(min(a, AGE_MAX)))) for a in self.meta["age"]]
        ).astype(np.float32)
        self.sex = np.array([encode_sex(s) for s in self.meta["sex"]],
                            dtype=np.int64)

        self._h5 = None
        self._rng = np.random.default_rng(seed)
        self.gen8_idx = np.asarray(GEN8_IDX, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def _open(self):
        if self._h5 is None:
            import h5py
            self._h5 = h5py.File(self.h5_path, "r")
        return self._h5

    def __getitem__(self, i: int):
        f = self._open()
        gi = int(self.rows[i])
        x12 = np.asarray(f["signals"][gi], dtype=np.float32)      # (12,1000) mV

        # --- olcekleme (normalizasyon YOK, sabit global olcek) ---
        x12 = np.clip(x12 * SIGNAL_SCALE, -SIGNAL_CLIP, SIGNAL_CLIP)
        x8 = x12[self.gen8_idx]                                   # (8,1000)

        # --- kosul maskeleme (classifier-free guidance) ---
        d_null = a_null = g_null = 0.0
        if self.cond_dropout > 0 or self.joint_dropout > 0:
            if self._rng.random() < self.joint_dropout:
                d_null = a_null = g_null = 1.0
            else:
                if self._rng.random() < self.cond_dropout: d_null = 1.0
                if self._rng.random() < self.cond_dropout: a_null = 1.0
                if self._rng.random() < self.cond_dropout: g_null = 1.0

        out = {
            "x":        torch.from_numpy(np.ascontiguousarray(x8)),
            "disease":  torch.from_numpy(self.labels[i]),
            "age_bits": torch.from_numpy(self.age_bits[i]),
            "sex":      torch.tensor(self.sex[i], dtype=torch.long),
            "d_null":   torch.tensor(d_null, dtype=torch.float32),
            "a_null":   torch.tensor(a_null, dtype=torch.float32),
            "g_null":   torch.tensor(g_null, dtype=torch.float32),
            "index":    torch.tensor(gi, dtype=torch.long),
        }
        if self.return_full12:
            out["x12"] = torch.from_numpy(np.ascontiguousarray(x12))
        return out

    # --- degerlendirme icin: tum kosullari toplu al ---
    def all_conditions(self) -> dict:
        return {
            "disease":  torch.from_numpy(self.labels),
            "age_bits": torch.from_numpy(self.age_bits),
            "sex":      torch.from_numpy(self.sex),
            "record":   self.meta["record"].tolist(),
            "parent":   self.meta["parent_record"].tolist(),
            "age":      self.meta["age"].to_numpy(),
        }

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None


# ============================================================================
# DATALOADER YARDIMCILARI
# ============================================================================

def make_loader(dataset: str,
                split: str,
                batch_size: int,
                shuffle: bool = True,
                num_workers: int = 4,
                cond_dropout: float = 0.0,
                joint_dropout: float = 0.0,
                return_full12: bool = False,
                seed: int = 0,
                drop_last: bool | None = None) -> DataLoader:
    ds = ECGDataset(dataset, split,
                    cond_dropout=cond_dropout,
                    joint_dropout=joint_dropout,
                    return_full12=return_full12,
                    seed=seed)
    if drop_last is None:
        drop_last = shuffle
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        drop_last=drop_last, persistent_workers=num_workers > 0,
    )


def denormalize(x_scaled: torch.Tensor) -> torch.Tensor:
    """Olceklenmis sinyali mV'a geri cevirir."""
    return x_scaled / SIGNAL_SCALE


# ============================================================================
# HIZLI DOGRULAMA
# ============================================================================

def sanity_check(dataset: str = "cpsc2018", n: int = 64):
    """Veri katmaninin dogru calistigini kontrol eder."""
    print(f"\n[{dataset}] veri katmani kontrolu")
    ds = ECGDataset(dataset, "train", return_full12=True)
    print(f"  ornek sayisi   : {len(ds)}")
    print(f"  sinif sayisi   : {ds.n_classes}  {ds.classes}")

    xs, x12s = [], []
    for i in range(min(n, len(ds))):
        b = ds[i]
        xs.append(b["x"].numpy())
        x12s.append(b["x12"].numpy())
    X = np.stack(xs)
    X12 = np.stack(x12s)

    print(f"  x sekli        : {X.shape}  (beklenen (n,8,{SIG_LEN}))")
    print(f"  olcekli std    : {X.std():.4f}   (hedef ~1.0)")
    print(f"  olcekli max|x| : {np.abs(X).max():.3f}")
    print(f"  NaN/Inf        : {np.isnan(X).sum()}/{np.isinf(X).sum()}")

    # 8 -> 12 rekonstruksiyon dogrulugu
    rec12 = leads8_to_12(torch.from_numpy(X)).numpy()
    err = np.abs(rec12 - X12).max(axis=(0, 2))
    print("  8->12 rekonstruksiyon max hata (olcekli birim):")
    for lead, e in zip(CANON_12, err):
        flag = "  <-- DIKKAT" if e > 0.05 else ""
        print(f"      {lead:<4}: {e:.5f}{flag}")

    # Gray kodu gidis-donus
    bad = [a for a in range(0, 120)
           if gray_decode(gray_encode(a)) != a]
    print(f"  Gray kodu hatasi: {len(bad)} (0 olmali)")

    # Kosul dagilimi
    lab = ds.labels
    print("  etiket dagilimi:",
          {c: int(lab[:, j].sum()) for j, c in enumerate(ds.classes)})
    print("  cinsiyet       :", dict(zip(*np.unique(ds.sex, return_counts=True))))
    ds.close()


if __name__ == "__main__":
    import sys
    sanity_check(sys.argv[1] if len(sys.argv) > 1 else "cpsc2018")
