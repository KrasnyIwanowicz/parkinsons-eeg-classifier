# Roadmap

A long-term project needs visible checkpoints — this is as much for you
as for anyone reviewing the repo.

## Phase 1 — Foundation
- [x] Download ds002778, verify it loads via `src/data_loading.py`
- [x] Run `preprocess_raw` + `make_epochs` on a couple of subjects, sanity-check by eye (plot raw vs. cleaned)
- [x] Extract features for the full dataset, save to `data/processed/`
- [x] Get the baseline (StandardScaler → PCA → SVM) running end-to-end via `scripts/run_experiment.py`

**Result: 67.7% subject-level accuracy, AUC 0.653, LOSO-validated (n=31).**

## Phase 2 — Deep models + real validation
- [ ] Implement a PyTorch `Dataset`/`DataLoader` around the per-subject epoch sequences
- [ ] Train `LSTMClassifier`, then `AttentionLSTMClassifier`
- [ ] Add EEGNet (or another CNN baseline) as a fourth point of comparison
- [ ] Run everything through `loso_cross_validate` — this is the number that goes in the README, not a random-split number

## Phase 3 — Explainability
- [ ] SHAP values on the baseline model — which bands/channels matter most?
- [ ] Attention-weight visualization on the attention-LSTM — which time windows matter most?
- [ ] Cross-check: do both explanation methods agree on anything? That's a more interesting finding than either one alone

## Phase 4 — Rigor & polish
- [ ] Get CI green (already scaffolded — just needs the real data-dependent tests added)
- [ ] Docstrings + type hints pass on every public function
- [ ] Results table + key plots in the README, replacing the placeholder
- [ ] Write the limitations section honestly (n=31 is small — say so)

## Phase 5 — Ship it
- [ ] Gradio or Streamlit demo: upload an EEG segment, see prediction + explanation
- [ ] Deploy the demo on Hugging Face Spaces
- [ ] Optional: write it up properly — this could become a real extended essay / competition submission, not just a repo
