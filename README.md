# LEAD-DM

Lead-structured latent diffusion for conditional 12-lead ECG synthesis.

This repository contains the code that produces every table and figure in the
paper. It does **not** contain the ECG databases, which are distributed by their
original providers under their own terms, nor any generated recordings. Both are
reproduced from the scripts below.

> **Paper** — R. Tekin, *Lead-Structured Latent Diffusion for Conditional 12-Lead
> ECG Synthesis: Prioritizing Diversity and a Condition-Matched Evaluation
> Protocol*. [journal, year, DOI to be added on acceptance]

---

## What the model does

LEAD-DM generates a 12-lead ECG from three conditions alone — a diagnostic
label, patient age and patient sex — plus a random seed. No real signal is
supplied at inference time.

Two design choices distinguish it from other latent diffusion models for ECG.
The autoencoder is grouped by lead, so each of the eight independent leads
occupies its own block of latent channels and no weight connects two leads. And
the three conditions are masked independently during training, so one model
represents every conditional in the family, which is what makes the negative
controls in the paper possible.

---

## Repository layout

```
src/
  lead_dm/            model, data, diffusion and reporting modules
  prepare_data.py     three raw databases  ->  uniform HDF5
  train_autoencoder.py
  train_diffusion.py
  generate.py         conditional sampling
  evaluate.py         fidelity / diversity protocol
  metrics.py          signal-, distribution- and task-level metrics
  physio_check.py     age and sex read-out
  run_ablations.py    ablation sweep
  compare_ablations.py  paired bootstrap
  collect_results.py  assembles the result tables
  diagnose_*.py       signal and latent diagnostics
  check_*.py          pre-flight shape and latent checks
figures/              scripts for the result figures, which read from
                      results/ and generated/
scripts/              dataset audit used before preprocessing
```

---

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/<user>/lead-dm.git
cd lead-dm
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA GPU is recommended but not required. Every result in the paper was
produced on a single 4 GB laptop GPU (NVIDIA RTX 3050 Laptop), with a peak
training memory of 730 MB.

---

## Data

The three databases are public and must be downloaded from their providers. We
do not redistribute them.

| Database | Source | Licence |
|---|---|---|
| PTB-XL | https://physionet.org/content/ptb-xl/ | ODC-BY 1.0 |
| Chapman-Shaoxing | https://physionet.org/content/ecg-arrhythmia/ | CC BY 4.0 |
| CPSC2018 | http://2018.icbeb.org/Challenge.html | see challenge terms |

Place the downloaded archives under `data/raw/` and run:

```bash
python src/prepare_data.py --raw data/raw --out data/prepared
```

This writes `{ptbxl,cpsc2018,chapman}.h5`, `*_meta.csv` and `*_labels.npy` under
`data/prepared/`. The preprocessing applies the canonical lead order,
resampling to 100 Hz, a fixed 10 s window, conversion to millivolts, age and sex
harmonisation, and the artefact filter described in the paper.

Run the audit first if you want to check the raw files before committing to the
preprocessing:

```bash
python scripts/ecg_dataset_audit.py --raw data/raw
```

---

## Reproducing the paper

The pipeline runs in four stages. Stage 1 must finish before stage 2, because
the diffusion model trains inside the frozen latent space.

### 1. Autoencoder

```bash
python src/train_autoencoder.py --dataset ptbxl --name AE_final
```

Roughly 2 hours on the reference hardware. Repeat for `cpsc2018` and `chapman`.

### 2. Diffusion model

```bash
python src/train_diffusion.py --dataset ptbxl --ae-name AE_final \
       --name M2_final --iters 300000
```

Roughly 14 hours on the reference hardware.

### 3. Generation and evaluation

```bash
python src/generate.py --dataset ptbxl --dm-name M2_final \
       --ddim 250 --guidance-all 1.25
python src/evaluate.py --dataset ptbxl --gen ptbxl__M2_final__gall1.25 \
       --classifiers xresnet1d50 inceptiontime
python src/physio_check.py --dataset ptbxl --gen ptbxl__M2_final__gall1.25
```

The guidance weight is 1.25 for PTB-XL and 1.0 for CPSC2018 and Chapman. The
selection rule is described in the paper: the largest weight for which macro F1
under the fidelity protocol stays below the real-data baseline.

### 4. Ablations and tables

```bash
python src/run_ablations.py --dataset ptbxl --iters 100000
python src/compare_ablations.py --dataset ptbxl
python src/collect_results.py --out results/
```

`compare_ablations.py` runs the paired bootstrap over a common test set and
reports the difference, its 95 % interval and a two-sided p-value.

### Figures

```bash
python figures/make_paper_figures.py                    # figures 4, 6-10
python figures/make_figure10.py --dataset ptbxl --gen <generation-name>  # figure 5
```

These read from `results/` and `generated/` and redraw every data-driven figure
in the paper. Figures 1 to 3 are schematic diagrams of the method rather than
plots of data, so their drawing code is not part of this repository. The
vector PDFs are in the published article.

---

## Trained weights

Model checkpoints are archived on Zenodo rather than in this repository, because
they are too large for comfortable version control:
[DOI to be added].

Download them into `runs/` to skip stages 1 and 2 and go straight to generation.

---

## Expected results

With the configuration above on PTB-XL, evaluated with xresnet1d50:

| Experiment | Accuracy | Macro F1 | AUC |
|---|---|---|---|
| Baseline (real, real) | 60.18 | 70.42 | 91.80 |
| Fidelity (real, synthetic) | 63.20 | 67.81 | 92.31 |
| Diversity (synthetic, real) | 48.92 | 63.77 | 87.37 |

Small deviations are expected. Every configuration in the paper was trained
once, so the reported differences are conditional on the seed. The paired
bootstrap quantifies test-set uncertainty but not run-to-run variability.

---

## Citation

```bibtex
@article{tekin2026leaddm,
  title   = {Lead-Structured Latent Diffusion for Conditional 12-Lead ECG
             Synthesis: Prioritizing Diversity and a Condition-Matched
             Evaluation Protocol},
  author  = {Tekin, Ramazan},
  journal = {TBD},
  year    = {2026},
  doi     = {TBD}
}
```

If you use the preprocessing or the evaluation protocol, please also cite the
database papers and the reference model whose protocol we adopt, listed in the
paper's reference list.

---

## Licence

Code is released under the MIT licence (see `LICENSE`). The ECG databases keep
their own licences and are not redistributed here.

---

## Notes and caveats

The generated recordings carry about 70 % of the amplitude variability and 64 %
of the rhythm regularity of real ones. Single-beat morphology is reproduced
accurately. This gap narrows with the number of sampling steps and is discussed
in the paper's limitations.

We did not evaluate whether generated recordings can be linked to individual
training patients. Membership-inference and nearest-neighbour analyses would be
needed before synthetic recordings from this model were shared as a substitute
for real data.

## Contact

Ramazan Tekin — Department of Computer Engineering, Faculty of Engineering and
Architecture, Batman University, 72070 Batman, Türkiye —
ramazan.tekin@batman.edu.tr
