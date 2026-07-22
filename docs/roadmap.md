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

**Result (superseded once — see below): initially the SVM baseline
appeared to beat both deep models. That conclusion was wrong: SHAP
analysis in Phase 3 uncovered a feature-scaling bug (band power and
Hjorth activity are right-skewed and were wrecking StandardScaler,
fixed with a log-transform). Once fixed and the SVM was given a fair
hyperparameter search (PCA components: 0, 10, 30, 50 all tried), the
final result reversed: LSTM (0.667 ± 0.074 acc / 0.686 ± 0.075 AUC,
3 seeds) clearly beats the SVM baseline (0.581 acc / 0.580 AUC, best of
4 PCA settings) and the Attention-LSTM (0.538 ± 0.049 acc /
0.514 ± 0.108 AUC, 3 seeds). The LSTM's ability to model temporal
structure across each subject's full recording is genuine signal a
per-epoch SVM can't access. Kept the wrong intermediate conclusion here
deliberately, crossed out, as a record of how the debugging actually went.**

## Phase 3 — Explainability
- [x] SHAP values on the SVM baseline — `scripts/explain_baseline.py`
- [x] Found and fixed a real bug via this analysis: band power / Hjorth
      activity are right-skewed, outlier-dominated distributions that
      were wrecking `StandardScaler` — every band showed exactly 0.0
      SHAP importance before the fix. Fixed with a log-transform. This
      is also what flipped the Phase 2 conclusion above.
- [x] Final result: beta/gamma power lead, all 8 feature types
      contribute (no more zeros); top channels are occipital (O1/O2/Oz)
      and frontal (Fp2/AF3), with some temporal/parietal too
- [x] Stated the honest interpretive caveat: those electrode sites are
      also exactly where eye-movement and tremor artifacts concentrate,
      so this can't fully rule out artifact-driven signal — see README
- [ ] ~~Attention-weight visualization on the Attention-LSTM~~ — descoped; that model underperforms the SVM baseline even after the fix, so its attention weights aren't expected to reflect a meaningful signal

## Phase 4 — Rigor & polish
- [ ] Get CI green (already scaffolded — just needs the real data-dependent tests added)
- [ ] Docstrings + type hints pass on every public function
- [ ] Results table + key plots in the README, replacing the placeholder
- [ ] Write the limitations section honestly (n=31 is small — say so)

## Phase 5 — Ship it
- [ ] Gradio or Streamlit demo: upload an EEG segment, see prediction + explanation
- [ ] Deploy the demo on Hugging Face Spaces
- [ ] Optional: write it up properly — this could become a real extended essay / competition submission, not just a repo
