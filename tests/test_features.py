import numpy as np

from src.features import band_power, extract_features, hjorth_parameters


def test_band_power_shapes():
    rng = np.random.default_rng(0)
    epoch = rng.normal(size=(4, 512))  # 4 channels, 512 samples
    powers = band_power(epoch, sfreq=256.0)
    assert set(powers.keys()) == {"delta", "theta", "alpha", "beta", "gamma"}
    for band_powers in powers.values():
        assert band_powers.shape == (4,)


def test_hjorth_parameters_shapes():
    rng = np.random.default_rng(0)
    epoch = rng.normal(size=(4, 512))
    activity, mobility, complexity = hjorth_parameters(epoch)
    assert activity.shape == mobility.shape == complexity.shape == (4,)


def test_extract_features_length():
    rng = np.random.default_rng(0)
    epoch = rng.normal(size=(4, 512))
    features = extract_features(epoch, sfreq=256.0)
    # 5 bands + 3 Hjorth params, each per channel (4 channels)
    assert features.shape == (8 * 4,)
