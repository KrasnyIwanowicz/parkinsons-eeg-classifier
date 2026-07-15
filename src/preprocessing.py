"""
EEG preprocessing pipeline: band-pass + notch filtering, average
referencing, ICA-based artifact removal, and fixed-length epoching.
"""
from __future__ import annotations

import mne


def preprocess_raw(
    raw: "mne.io.Raw",
    l_freq: float = 1.0,
    h_freq: float = 45.0,
    notch_freq: float = 50.0,
    n_ica_components: int = 15,
) -> "mne.io.Raw":
    """Band-pass filter, remove line noise, and clean via ICA.

    Returns a new, cleaned Raw object — does not modify `raw` in place.
    """
    raw = raw.copy().load_data()
    raw.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    raw.notch_filter(freqs=notch_freq, verbose=False)
    raw.set_eeg_reference("average", verbose=False)

    n_components = min(n_ica_components, len(raw.ch_names) - 1)
    ica = mne.preprocessing.ICA(
        n_components=n_components, random_state=42, max_iter="auto"
    )
    ica.fit(raw, verbose=False)

    # ds002778 has no dedicated EOG channel, so frontal electrodes (Fp1/Fp2)
    # are used as an eye-blink proxy. Inspect ica.plot_components() /
    # ica.plot_sources() manually the first few times — don't trust this
    # heuristic blindly on a new dataset or montage.
    if _has_eog_proxy(raw):
        eog_indices, _ = ica.find_bads_eog(raw, verbose=False)
        ica.exclude = eog_indices

    return ica.apply(raw.copy(), verbose=False)


def _has_eog_proxy(raw: "mne.io.Raw") -> bool:
    return any(ch in raw.ch_names for ch in ("Fp1", "Fp2"))


def make_epochs(raw: "mne.io.Raw", duration: float = 2.0, overlap: float = 0.0) -> "mne.Epochs":
    """Cut a continuous resting-state recording into fixed-length epochs."""
    events = mne.make_fixed_length_events(raw, duration=duration, overlap=overlap)
    return mne.Epochs(
        raw, events, tmin=0, tmax=duration, baseline=None, preload=True, verbose=False
    )
