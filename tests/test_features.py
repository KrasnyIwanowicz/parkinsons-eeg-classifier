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


def test_extract_features_log_transform_survives_outlier_epochs():
    """Regression test for a real bug: band power and Hjorth activity
    are raw power-like quantities with heavily right-skewed
    distributions in real EEG (a handful of high-power/artifact epochs
    dominate the scale). Without a log-transform, StandardScaler's
    mean/std gets wrecked by those outliers and crushes normal
    epoch-to-epoch variation into a near-invisible range -- which is
    exactly what happened: an earlier SHAP run showed exactly 0.0
    importance for every band and for hjorth_activity. This test builds
    the same outlier-dominated distribution and checks the 'normal'
    epochs still span a usable range after standardization."""
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    n_epochs, n_channels, sfreq = 200, 4, 512.0

    is_outlier = rng.random(n_epochs) < 0.05  # 5% of epochs are ~100x higher power
    feature_matrix = np.array(
        [
            extract_features(
                rng.normal(scale=1e-3 if outlier else 1e-5, size=(n_channels, 1024)), sfreq
            )
            for outlier in is_outlier
        ]
    )

    scaled = StandardScaler().fit_transform(feature_matrix)
    delta_block = scaled[~is_outlier, :n_channels]  # delta power, normal epochs only

    # Without the log-transform this range collapses to ~0.006 (values
    # essentially indistinguishable) -- with it, normal epochs should
    # span a meaningfully wide chunk of the standardized distribution.
    normal_range = np.percentile(delta_block, 95) - np.percentile(delta_block, 5)
    assert normal_range > 0.3
