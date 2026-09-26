"""
LEAD-DM : Lead-structured Conditional Diffusion Model for 12-Lead ECG Generation
================================================================================
M1 = LEAD-DM/T  (derivasyon-yerel TCN omurga + LTCM + asimetrik basliklar)
"""

__version__ = "0.1.0"

from .config import (ExperimentConfig, AEConfig, M1Config, DiffusionConfig,
                     TrainConfig, ablation,
                     PREPARED_DIR, RUNS_DIR, RESULTS_DIR,
                     DATASET_CLASSES, CANON_12, GEN8_LEADS, GEN8_IDX,
                     FS, SIG_LEN, SIGNAL_SCALE)
from .data import ECGDataset, make_loader, leads8_to_12, gray_encode, denormalize
from .autoencoder import LeadStructuredAE, AELoss
from .backbone_m1 import LeadDM_T, build_m1
from .diffusion import GaussianDiffusion, EMA
from .reporting import (ExcelReport, add_spec_table, empty_table,
                        setup_style, new_figure, save_figure,
                        plot_training_curves, plot_ecg_12lead,
                        plot_lead_attention, plot_metric_bars)

__all__ = [
    "ExperimentConfig", "AEConfig", "M1Config", "DiffusionConfig",
    "TrainConfig", "ablation", "ECGDataset", "make_loader", "leads8_to_12",
    "gray_encode", "denormalize", "LeadStructuredAE", "AELoss",
    "LeadDM_T", "build_m1", "GaussianDiffusion", "EMA",
    "ExcelReport", "add_spec_table", "empty_table",
    "setup_style", "new_figure", "save_figure",
    "plot_training_curves", "plot_ecg_12lead", "plot_lead_attention",
    "plot_metric_bars",
]
