# Roadmap

A long-term project needs visible checkpoints — this is as much for you
as for anyone reviewing the repo.

## Phase 1 — Foundation
- [x] Download ds002778, verify it loads via `src/data_loading.py`
- [x] Run `preprocess_raw` + `make_epochs` on a couple of subjects, sanity-check by eye (plot raw vs. cleaned)
- [x] Extract features for the full dataset, save to `data/processed/`
- [x] Get the baseline (StandardScaler → PCA → SVM) running end-to-end via `scripts/run_experiment.py`

**Result: 64.5% subject-level accuracy, AUC 0.661, LOSO-validated (n=31,
32 scalp channels).**

## Phase 2 — Deep models + real validation
- [x] Implement a PyTorch `Dataset`/`DataLoader` around the per-subject epoch sequences
- [x] Train `LSTMClassifier`, then `AttentionLSTMClassifier`
- [x] ~~Add EEGNet (or another CNN baseline) as a fourth point of comparison~~ — skipped; the conclusion below is about sample size, not architecture, so a 4th deep model is unlikely to change it
- [x] Run everything through `loso_cross_validate` — this is the number that goes in the README, not a random-split number

**Result: the SVM baseline (0.645 acc / 0.661 AUC) clearly beats both
LSTM (0.516 ± 0.055 acc / 0.528 ± 0.079 AUC) and Attention-LSTM
(0.538 ± 0.049 acc / 0.514 ± 0.012 AUC), averaged over 3 seeds each.
Both deep models sit at chance. Legitimate finding: n=31 isn't enough
data for these architectures to beat hand-crafted features + SVM.**

## Phase 3 — Explainability
- [x] SHAP values on the SVM baseline — `scripts/explain_baseline.py`, ready to run
- [ ] ~~Attention-weight visualization on the Attention-LSTM~~ — descoped; Phase 2 found that model performs at chance, so its attention weights aren't expected to reflect a meaningful signal. Documented as a limitation instead of pursued as a result.
- [ ] Run `scripts/explain_baseline.py` on real data, add the resulting band/channel importance + plots to the README

## Phase 4 — Rigor & polish
- [ ] Get CI green (already scaffolded — just needs the real data-dependent tests added)
- [ ] Docstrings + type hints pass on every public function
- [ ] Results table + key plots in the README, replacing the placeholder
- [ ] Write the limitations section honestly (n=31 is small — say so)

## Phase 5 — Ship it
- [ ] Gradio or Streamlit demo: upload an EEG segment, see prediction + explanation
- [ ] Deploy the demo on Hugging Face Spaces
- [ ] Optional: write it up properly — this could become a real extended essay / competition submission, not just a repo
