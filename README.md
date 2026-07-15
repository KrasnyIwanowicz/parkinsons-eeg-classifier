# Parkinson's Disease Detection from Resting-State EEG

[![CI](https://github.com/YOUR_USERNAME/parkinsons-eeg-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_USERNAME/parkinsons-eeg-classifier/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

> Replace `YOUR_USERNAME` above with your actual GitHub username once you push this.

## Overview

Parkinson's disease (PD) is currently diagnosed through observation of
motor symptoms — by the time those appear, substantial dopaminergic
neuron loss has already occurred. EEG is a cheap, non-invasive signal
that shows measurable differences between PD patients and healthy
controls (notably slowed cortical oscillations and increased
low-frequency power), which makes it an attractive candidate for
earlier, more accessible screening.

This project builds and rigorously evaluates a pipeline for classifying
PD vs. healthy control from resting-state EEG, comparing a classical
feature-based approach against sequence deep-learning models, with an
explicit focus on two things most student EEG projects skip:

1. **Subject-independent validation** (leave-one-subject-out CV) —
   without it, reported accuracy is largely measuring how well the
   model memorized each subject's individual noise signature, not
   whether it generalizes to a new person.
2. **Explainability** — a classifier that can't say *why* is a hard
   sell for anything medical-adjacent.

## Dataset

**UC San Diego Resting-State EEG Data from Patients with Parkinson's
Disease** — [OpenNeuro ds002778](https://openneuro.org/datasets/ds002778)

- 31 participants: 15 with PD, 16 healthy controls
- 32-channel BioSemi ActiveTwo system, 512 Hz sampling rate
- ~3 minutes of eyes-open resting-state recording per participant
- Standard 10-20 electrode placement, BIDS-formatted

**Citation:**
> Swann, N.C. (2021). UC San Diego Resting State EEG Data from Patients
> with Parkinson's Disease. OpenNeuro. Dataset.
> https://doi.org/10.18112/openneuro.ds002778.v1.0.5

The data is **not** included in this repository (clinical EEG data,
and large enough that it doesn't belong in git). To get it:

```bash
pip install openneuro-py
openneuro-py download --dataset ds002778 --target data/raw
```

## Methodology

**Preprocessing** (`src/preprocessing.py`): 1–45 Hz band-pass filter,
50 Hz notch filter, average reference, ICA-based artifact removal,
2-second fixed-length epoching.

**Features** (`src/features.py`): spectral band power (delta, theta,
alpha, beta, gamma via Welch's method) and Hjorth parameters
(activity, mobility, complexity), per channel.

**Models compared** (`src/models.py`):
| Model | Type | Purpose |
|---|---|---|
| StandardScaler → PCA → SVM | Classical ML | Baseline / sanity floor |
| LSTM | Deep sequence model | Learns temporal structure directly |
| Attention-LSTM | Deep sequence model | Same, plus built-in explainability via attention weights |

**Validation** (`src/evaluation.py`): leave-one-subject-out
cross-validation via `sklearn.model_selection.LeaveOneGroupOut`,
reporting accuracy, ROC-AUC, and confusion matrix.

**Explainability** (`src/explainability.py`): SHAP values for the
baseline model; attention-weight extraction for the attention-LSTM —
two independent ways of answering "what drove this prediction?"

## Repository structure

```
├── configs/default.yaml       # all hyperparameters in one place
├── data/                      # raw/processed data (gitignored, download separately)
├── docs/roadmap.md            # phased project roadmap
├── scripts/run_experiment.py  # end-to-end: load → preprocess → features → LOSO-CV
├── src/
│   ├── data_loading.py        # BIDS/OpenNeuro loading
│   ├── preprocessing.py       # filtering, ICA, epoching
│   ├── features.py            # band power, Hjorth parameters
│   ├── models.py               # baseline SVM pipeline + LSTM/Attention-LSTM
│   ├── evaluation.py           # leave-one-subject-out CV
│   └── explainability.py       # SHAP + attention-weight extraction
├── tests/                      # run on synthetic EEG data — no dataset download needed for CI
└── .github/workflows/ci.yml    # tests run automatically on every push
```

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/parkinsons-eeg-classifier.git
cd parkinsons-eeg-classifier
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# then download the dataset (see Dataset section above) into data/raw
```

## Usage

```bash
# run the full pipeline and get LOSO-validated baseline results
python scripts/run_experiment.py --bids-root data/raw

# run the test suite (uses synthetic data, no download needed)
pytest tests/ -v
```

## Results

*To be filled in as experiments are run — see [docs/roadmap.md](docs/roadmap.md)
for what's done and what's next. Deliberately left empty rather than
filled with placeholder numbers.*

| Model | Accuracy (LOSO) | ROC-AUC (LOSO) |
|---|---|---|
| SVM baseline | — | — |
| LSTM | — | — |
| Attention-LSTM | — | — |

## Limitations

- n = 31 is a small sample even by EEG standards — treat any result
  here as a proof of concept, not a clinical claim.
- Single-site, single-recording-protocol data; nothing here has been
  validated on an independent cohort or a different EEG system.
- Not intended for, and not validated for, actual clinical diagnostic
  use.

## References

- Swann, N.C. (2021). UC San Diego Resting State EEG Data from Patients
  with Parkinson's Disease. OpenNeuro. https://doi.org/10.18112/openneuro.ds002778.v1.0.5
- Delorme, A. & Makeig, S. (2004). EEGLAB: an open-source toolbox for
  analysis of single-trial EEG dynamics. *Journal of Neuroscience
  Methods*, 134, 9–21.

## License

MIT — see [LICENSE](LICENSE).
