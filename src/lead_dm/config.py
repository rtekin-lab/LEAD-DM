#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LEAD-DM  |  Merkezi yapilandirma
=================================
Tum sabitler burada. Deneyler arasinda degisen tek sey bu dosyadaki
(veya CLI ile ezilen) alanlar olmali; kod govdesi sabit kalmali.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Sequence
import json

# ============================================================================
# YOLLAR
# ============================================================================

# TEK KAYNAK. Yol degisikligi YALNIZCA burada yapilir; diger dosyalar
# bu sabitleri import eder. (Daha once metrics.py/evaluate.py kendi
# kopyalarini tutuyordu ve senkron kalmiyordu.)
PREPARED_DIR  = Path("./prepared")       # prepare_data.py ciktisi
GENERATED_DIR = Path("./generated")      # generate.py ciktisi
RUNS_DIR      = Path("./runs")           # egitim ciktilari (checkpoint, log)
RESULTS_DIR   = Path("./results")        # Excel tablolari + sekiller


def require_files(directory: Path, stem: str, kind: str = "veri") -> None:
    """
    Beklenen ucluyu (h5 + _meta.csv + _labels.npy) kontrol eder ve
    eksikse ne bulundugunu listeleyerek anlasilir hata verir.
    """
    directory = Path(directory)
    need = [directory / f"{stem}.h5",
            directory / f"{stem}_meta.csv",
            directory / f"{stem}_labels.npy"]
    missing = [p for p in need if not p.exists()]
    if not missing:
        return
    lines = [f"{kind} dosyalari bulunamadi.",
             f"  Aranan klasor : {directory.resolve()}",
             f"  Aranan ad     : {stem}",
             "  Eksik:"]
    lines += [f"    - {p.name}" for p in missing]
    if directory.exists():
        found = sorted({f.stem.replace("_meta", "").replace("_labels", "")
                        for f in directory.iterdir()
                        if f.suffix in (".h5", ".csv", ".npy")})
        lines.append(f"  Bu klasorde bulunanlar: {found if found else '(bos)'}")
    else:
        lines.append("  [!] Klasor hic yok.")
    lines.append("  Yollar lead_dm/config.py icinde tanimlidir "
                 "(PREPARED_DIR / GENERATED_DIR).")
    raise FileNotFoundError("\n".join(lines))

DATASETS = ("ptbxl", "cpsc2018", "chapman")

DATASET_CLASSES = {
    "ptbxl":    ["NORM", "MI", "STTC", "CD", "HYP"],
    "cpsc2018": ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"],
    "chapman":  ["AFIB", "GSVT", "SB", "SR"],
}

# ============================================================================
# SINYAL SABITLERI
# ============================================================================

FS         = 100          # Hz
SECONDS    = 10
SIG_LEN    = FS * SECONDS # 1000

CANON_12 = ("I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6")

# Modelin dogrudan urettigi 8 kanal. Kalan 4 uzuv derivasyonu
# Einthoven/Goldberger denklemleriyle turetilir (denetimde hata ~0.002 mV).
GEN8_LEADS = ("V1", "V2", "V3", "V4", "V5", "V6", "I", "aVF")
GEN8_IDX   = tuple(CANON_12.index(l) for l in GEN8_LEADS)   # (6,7,8,9,10,11,0,5)
N_GEN_LEADS = len(GEN8_LEADS)                                # 8

# GEN8 icindeki konumlar (LTCM ve asimetrik basliklar icin)
CHEST_POS = (0, 1, 2, 3, 4, 5)   # V1..V6  -> derin baslik
LIMB_POS  = (6, 7)               # I, aVF  -> sig baslik

# Sinyal olcegi: denetimde std ~0.21-0.26 mV cikti.
# x_scaled = x_mV * SIGNAL_SCALE  -> std ~1.0
SIGNAL_SCALE = 4.0
SIGNAL_CLIP  = 20.0   # +-20 (yani +-5 mV); aykiri deger zaten 8 mV'da elenmisti

# ============================================================================
# KOSUL KODLAMA
# ============================================================================

AGE_BITS = 7          # Gray kodu, 0..127 araligi
AGE_MAX  = 119
SEX_MALE, SEX_FEMALE, SEX_NULL = 0, 1, 2

COND_DIM = 128        # her kosul gommesinin boyutu


# ============================================================================
# MODEL YAPILANDIRMASI
# ============================================================================

@dataclass
class AEConfig:
    """Derivasyon-yapili enkoder-dekoder."""
    n_leads: int = N_GEN_LEADS         # 8
    ch_per_lead: int = 32              # gizil uzayda derivasyon basina kanal
    stem_ch_per_lead: int = 16
    mid_ch_per_lead: int = 24
    n_down: int = 3                    # 1000 -> 500 -> 250 -> 125
    kernel: int = 7
    gn_groups: int = 8                 # GroupNorm grup sayisi (lead basina)
    lead_independent: bool = True      # True -> tum conv'lar groups=n_leads

    # --- Gizil uzay duzenlileştirmesi (KRITIK) ---
    # Duzenlileştirilmemis bir otokodlayici enerjiyi az sayida boyuta
    # sikistirir. Difuzyon N(0,I)'dan basladigi icin dekoderin hic gormedigi
    # yonlere enerji koyar -> manifold disi -> gurultu. Stable Diffusion'in
    # VAE'si tam olarak bu yuzden KL duzenlileştirmesi kullanir.
    variational: bool = True           # True -> VAE (mu, logvar)
    kl_weight: float = 1e-6            # KL agirligi (SD ~1e-6)
    latent_norm: str = "per_channel"   # scalar | per_channel

    @property
    def latent_ch(self) -> int:        # 8*32 = 256
        return self.n_leads * self.ch_per_lead

    @property
    def latent_len(self) -> int:       # 1000 / 2^3 = 125
        return SIG_LEN // (2 ** self.n_down)


@dataclass
class M1Config:
    """M1 = LEAD-DM/T  (derivasyon-yerel TCN gurultu tahmincisi)."""
    n_leads: int = N_GEN_LEADS
    ch_per_lead: int = 32
    n_blocks: int = 12
    kernel: int = 7
    dilations: Sequence[int] = (1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8)
    dropout: float = 0.0
    use_mix_path: bool = True          # ablasyon: derivasyonlar arasi karisim
    # Kosullama
    modulation: str = "ltcm"           # ltcm | mcfarn | adaln | crossattn
    # 1.0 olmali. 0.0 verilirse OLU BOLGE olusur: lam=0 iken 'dis' MLP'sine
    # gradyan gitmez, 'dis' sifir iken lam'a gradyan gitmez -> LTCM hic
    # ogrenmez (gozlenen belirti: lam=0.0000 sabit kalir).
    # 'dis' zaten sifir-ilklendirildigi icin lam=1.0 ile de baslangicta
    # gamma=beta=0, yani LTCM hala adaLN-Zero ile OZDES.
    ltcm_lambda_init: float = 1.0
    # Baslik
    # Asama 1 bulgusu: goreli hata (RMSE/lead RMS) uzuv derivasyonlarinda
    # %34-73 daha yuksek. Ayrica I ve aVF hatalari 8->12 turetmede 2.63x ve
    # 3.50x buyuyerek II/III/aVR/aVL'ye yayiliyor. Kapasite uzva kaydirildi.
    asymmetric_heads: bool = True      # uzuv derin, gogus sig
    chest_head_depth: int = 2
    limb_head_depth: int = 4
    uniform_head_depth: int = 2        # asymmetric_heads=False iken

    @property
    def channels(self) -> int:
        return self.n_leads * self.ch_per_lead


@dataclass
class DiffusionConfig:
    n_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    schedule: str = "cosine"           # linear | cosine

    # --- Parametrizasyon (KRITIK) ---
    # "eps": z0_hat = (1/sqrt(ab)) z_t - sqrt(1/ab-1) eps_hat
    #        t=999'da katsayi 157 -> kucuk hata devasa z0_hat -> varyans sismesi
    #        (olculen belirti: ornekleme sonunda z_std 2.9, gercek 1.0)
    # "v"  : v = sqrt(ab) eps - sqrt(1-ab) z0
    #        z0_hat = sqrt(ab) z_t - sqrt(1-ab) v_hat, katsayilar <= 1
    #        Amplifikasyon yapisal olarak yok.
    parameterization: str = "v"        # v | eps

    # Lineer zamanlamada alphabar_T = 4e-5, yani z_T'de sinyal sizintisi var.
    # Ornekleme saf gurultuden basladigi icin egitim/ornekleme uyumsuzlugu
    # olusur. True -> alphabar_T = 0. v-parametrizasyonu ile kullanilmali.
    zero_terminal_snr: bool = True
    # Kayip
    lambda_freq: float = 0.1           # frekans kaybi agirligi
    freq_snr_clip: float = 5.0         # min-SNR tarzi ust kirpma (patlamayi onler)
    freq_hf_gain: float = 4.0          # yuksek frekans vurgusu: w(f) = 1 + gain*f
    lambda_pde: float = 0.0            # M8'de acilir
    lambda_interlead: float = 0.0      # M8'de acilir
    # Maskeli kosul egitimi (classifier-free guidance)
    cond_dropout: float = 0.10         # her kosul bagimsiz dusurulur
    joint_dropout: float = 0.05        # ucu birden dusurulur


@dataclass
class TrainConfig:
    dataset: str = "cpsc2018"
    # Autoencoder
    ae_epochs: int = 120
    ae_batch: int = 128
    ae_lr: float = 1e-3
    # Diffusion
    dm_iters: int = 100_000
    dm_batch: int = 32
    dm_lr: float = 2e-4
    dm_warmup: int = 1_000
    ema_decay: float = 0.9999
    grad_clip: float = 1.0
    amp: bool = True
    num_workers: int = 4
    seed: int = 42
    log_every: int = 100
    ckpt_every: int = 5_000
    val_every: int = 2_000


@dataclass
class ExperimentConfig:
    name: str = "M1_leadtcn"
    ae: AEConfig = field(default_factory=AEConfig)
    model: M1Config = field(default_factory=M1Config)
    diffusion: DiffusionConfig = field(default_factory=DiffusionConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, default=str),
                        encoding="utf-8")

    def run_dir(self) -> Path:
        return RUNS_DIR / f"{self.train.dataset}__{self.name}"


# ============================================================================
# ABLASYON KISAYOLLARI  (Bolum 5, Asama 2)
# ============================================================================

def ablation(name: str, base: ExperimentConfig | None = None) -> ExperimentConfig:
    """Rapordaki A1..A8 ablasyon konfigurasyonlarini uretir."""
    import copy
    cfg = copy.deepcopy(base) if base is not None else ExperimentConfig()
    cfg.name = f"M1_{name}"

    if name == "A1_full":
        pass
    elif name == "A2_flat_latent":
        cfg.ae.lead_independent = False        # duz gizil uzay
        cfg.model.modulation = "mcfarn"        # LTCM lead yapisi ister
    elif name == "A3_single_head":
        cfg.model.asymmetric_heads = False
        cfg.model.uniform_head_depth = 0       # 0 -> tek conv cikis
    elif name == "A4a_symmetric_heads":
        cfg.model.asymmetric_heads = False
        cfg.model.uniform_head_depth = 2
    elif name == "A4b_chest_deep":
        cfg.model.asymmetric_heads = True      # ilk (sezgisel) tasarim
        cfg.model.chest_head_depth = 3
        cfg.model.limb_head_depth = 1
    elif name == "A4c_limb_deep":
        cfg.model.asymmetric_heads = True      # Asama 1 verisine dayali
        cfg.model.chest_head_depth = 2
        cfg.model.limb_head_depth = 4
    elif name == "A5_mcfarn":
        cfg.model.modulation = "mcfarn"
    elif name == "A6_adaln":
        cfg.model.modulation = "adaln"
    elif name == "A7_crossattn":
        cfg.model.modulation = "crossattn"
    elif name == "A8_ltcm_global_only":
        cfg.model.modulation = "ltcm"
        cfg.model.ltcm_lambda_init = -1.0      # ozel deger: dal B/C kapali
    else:
        raise ValueError(f"bilinmeyen ablasyon: {name}")
    return cfg
