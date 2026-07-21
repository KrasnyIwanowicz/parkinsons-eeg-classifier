# Parkinson's Disease Detection from Resting-State EEG

[![CI](https://github.com/KrasnyIwanowicz/parkinsons-eeg-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/KrasnyIwanowicz/parkinsons-eeg-classifier/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)


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

**Structure, as actually shipped** (verified against the dataset repo,
not assumed): there is no "Group" column in `participants.tsv` — the
label lives in the subject ID prefix itself (`sub-hc*` = control,
`sub-pd*` = PD). Healthy controls have a single session (`ses-hc`); PD
patients have **two** sessions, off-medication (`ses-off`) and
on-medication (`ses-on`). This repo's loader defaults to the
off-medication session for the PD-vs-control comparison, so medication
state doesn't confound the group difference. Raw EEG ships as `.bdf`.

**⚠️ A note from the dataset's own maintainers, verbatim from their
README, that matters a lot for how you frame any results here:**
> "An example of an analysis that we could consider problematic ... would
> be using machine learning to classify Parkinson's patients from
> healthy controls using this dataset. This is because there are far
> too few patients for proper statistics... We strongly advise against
> using any such approach because it would mislead patients and people
> who are interested in knowing if they have Parkinson's disease."
>
> — Alex Rockhill (University of Oregon), dataset curator

This project still builds the classifier — that's the point of the
exercise — but treat it explicitly as a **methods/engineering
demonstration** (feature pipeline, validation rigor, explainability),
never as a diagnostic claim, and say so plainly wherever results are
reported. If this project is ever written up for anything more formal
than a portfolio (a paper, a competition submission), the dataset's
README asks that you email the curator first: arockhil@uoregon.edu.

The data is **not** included in this repository (clinical EEG data,
and large enough that it doesn't belong in git). To get it:

```bash
pip install openneuro-py
openneuro-py download --dataset ds002778 --target-dir data/raw
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
git clone https://github.com/KrasnyIwanowicz/parkinsons-eeg-classifier.git
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

**Baseline (StandardScaler → PCA → SVM), leave-one-subject-out CV, n = 31,
32 scalp EEG channels:**

| Level | Accuracy | ROC-AUC | Confusion matrix |
|---|---|---|---|
| Epoch-level (n = 3009 epochs) | 0.645 | 0.661 | `[[965, 555], [534, 955]]` |
| **Subject-level** (majority vote, n = 31) | **0.645** | — | `[[10, 6], [5, 10]]` |

Subject-level: 10/16 controls and 10/15 PD subjects correctly classified
(62.5% specificity, 66.7% sensitivity). An AUC of ~0.66 under honest
LOSO validation on n = 31 is a modest but real result — not evidence of
anything close to diagnostic reliability (see the Limitations section
and the dataset curators' own caveat above), but a legitimate signal
that survived a validation scheme designed specifically not to flatter
it.

*Deep model (LSTM / Attention-LSTM) results:*

| Model | Accuracy (subject-level, n=31) | ROC-AUC |
|---|---|---|
| SVM baseline (deterministic) | **0.645** | **0.661** |
| LSTM (mean ± SD, 3 seeds) | 0.516 ± 0.055 | 0.528 ± 0.079 |
| Attention-LSTM (mean ± SD, 3 seeds) | 0.538 ± 0.049 | 0.514 ± 0.012 |

**The classical baseline clearly outperforms both deep sequence
models, which sit essentially at chance (0.50) even after fixing every
bug that surfaced along the way** — a channel-selection bug that
included 9 non-scalp channels in every feature vector, unseeded
training causing results to swing arbitrarily between runs, and severe
overfitting from too much model capacity for the data available. Once
all three were fixed and results were averaged across 3 random seeds
per model, the conclusion held steady: with only 31 subjects, there
isn't enough data for an LSTM to learn anything an SVM on hand-crafted
spectral/Hjorth features doesn't already capture. This is a legitimate
finding, not a failure — reporting "the simple model won, honestly
evaluated" is more credible than chasing a bigger number with a
methodology that doesn't hold up. A fourth architecture (EEGNet) was
considered but skipped: it's unlikely a different deep model changes a
conclusion that's fundamentally about sample size, not architecture
choice.

## Limitations

- **n = 31 (15 PD, 16 control) is too small for statistically reliable
  diagnostic claims — the dataset's own curators say so explicitly**
  (see the callout above). With leave-one-subject-out CV, every subject
  is tested exactly once, which is a fairer evaluation than a single
  fixed train/test split, but it does not fix the fundamental
  small-sample problem. Treat any accuracy/AUC number this project
  produces as an engineering proof-of-concept, not evidence of
  real-world diagnostic validity.
- Single-site, single-recording-protocol data; nothing here has been
  validated on an independent cohort or a different EEG system.
- UPDRS-style clinical ratings referenced in the dataset were collected
  by trained lab personnel, not a board-certified neurologist — treat
  any analysis leaning on disease-severity labels with extra caution.
- Not intended for, and not validated for, actual clinical diagnostic
  use.

## References

- Swann, N.C. (2021). UC San Diego Resting State EEG Data from Patients
  with Parkinson's Disease. OpenNeuro. https://doi.org/10.18112/openneuro.ds002778.v1.0.5
- Jackson, N., Cole, S.R., Voytek, B., Swann, N.C. (2019). Characteristics
  of Waveform Shape in Parkinson's Disease Detected with Scalp
  Electroencephalography. *eNeuro*, 6(3). https://doi.org/10.1523/ENEURO.0151-19.2019
- Swann, N.C. et al. (2015). Elevated synchrony in Parkinson disease
  detected with electroencephalography. *Ann Neurol*, 78(5), 742-750.
- Delorme, A. & Makeig, S. (2004). EEGLAB: an open-source toolbox for
  analysis of single-trial EEG dynamics. *Journal of Neuroscience
  Methods*, 134, 9–21.
- Appelhoff, S. et al. (2019). MNE-BIDS: Organizing electrophysiological
  data into the BIDS format. *JOSS*, 4(44), 1896.

*If you ever submit anything based on this dataset for formal
publication (not just a portfolio/school project), the dataset
maintainers ask to be emailed first — see the callout in the Dataset
section above.*

## License

MIT — see [LICENSE](LICENSE).
