"""
Pytest suite for the NaN handling in transfer_function_helpers.

Tests cover:
- drop_nans: dropping the observations with a value that is not finite in X, Y
  or RR, and returning the inputs when every value is finite
- process_transfer_functions_with_weights: the EDF weights of the output
  channels without channel weights, computed once per band
"""

import numpy as np
import pytest
import xarray as xr
from mt_metadata.common.band import Band
from mt_metadata.processing.aurora.decimation_level import DecimationLevel

from aurora.pipelines.transfer_function_helpers import (
    apply_weights,
    drop_nans,
    get_estimator_class,
    process_transfer_functions_with_weights,
    set_up_iter_control,
    stack_fcs,
)
from aurora.time_series.frequency_band_helpers import get_band_for_tf_estimate
from aurora.time_series.xarray_helpers import handle_nan
from aurora.transfer_function.weights.edf_weights import (
    effective_degrees_of_freedom_weights,
)


def _band_fcs(channels: list, seed: int, n_frequency: int = 3, n_time: int = 40):
    """Fourier coefficients of one band: complex data on frequency and time."""
    rng = np.random.default_rng(seed)
    times = np.datetime64("2023-09-23T00:00:00", "ns") + np.arange(
        n_time
    ) * np.timedelta64(64, "s")
    coords = {"frequency": np.linspace(0.10, 0.12, n_frequency), "time": times}
    shape = (n_frequency, n_time)
    data_vars = {
        ch: (
            ("frequency", "time"),
            rng.normal(size=shape) + 1j * rng.normal(size=shape),
        )
        for ch in channels
    }
    return xr.Dataset(data_vars, coords=coords)


def _append_observation(ds: xr.Dataset, value=None) -> xr.Dataset:
    """ds with one more observation, a copy of the first, every variable set to value if given."""
    extra = ds.isel(observation=[0]).copy(deep=True)
    if value is not None:
        for name in extra.data_vars:
            extra[name].values[:] = value
    return xr.concat([ds, extra], dim="observation")


@pytest.fixture
def stacked_band():
    """X, Y and RR of one band, stacked to observations as in the regression."""
    X = _band_fcs(["hx", "hy"], seed=1)
    Y = _band_fcs(["ex", "ey"], seed=2)
    RR = _band_fcs(["hx", "hy"], seed=3)
    return stack_fcs(X, Y, RR)


# =============================================================================
# Test Classes
# =============================================================================


class TestDropNansAllFinite:
    """Test drop_nans when every value is finite."""

    def test_returns_inputs(self, stacked_band):
        """X, Y and RR are returned as they are, not copied."""
        X, Y, RR = stacked_band
        X_out, Y_out, RR_out = drop_nans(X, Y, RR)
        assert X_out is X
        assert Y_out is Y
        assert RR_out is RR

    def test_returns_inputs_without_remote(self, stacked_band):
        """Without a remote reference, X and Y are returned as they are."""
        X, Y, _ = stacked_band
        X_out, Y_out, RR_out = drop_nans(X, Y, None)
        assert X_out is X
        assert Y_out is Y
        assert RR_out is None

    def test_unstacked_inputs_raise(self):
        """X, Y and RR not stacked to observations raise ValueError from isel, as
        they do when a value is not finite."""
        X = _band_fcs(["hx", "hy"], seed=1)
        Y = _band_fcs(["ex", "ey"], seed=2)
        RR = _band_fcs(["hx", "hy"], seed=3)
        with pytest.raises(ValueError):
            drop_nans(X, Y, RR)

    def test_other_observation_count_raises(self, stacked_band):
        """Y on fewer observations than X raises ValueError from the mask, as it
        does when a value is not finite."""
        X, Y, RR = stacked_band
        with pytest.raises(ValueError):
            drop_nans(X, Y.isel(observation=slice(0, -2)), RR)

    def test_other_observation_labels_raise(self, stacked_band):
        """RR indexed by other observation labels than X raises IndexError from
        isel, as it does when a value is not finite."""
        X, Y, RR = stacked_band
        labels = np.arange(X.sizes["observation"])
        X = X.assign_coords(observation=labels)
        Y = Y.assign_coords(observation=labels)
        RR = RR.assign_coords(observation=labels + 2)
        with pytest.raises(IndexError):
            drop_nans(X, Y, RR)

    def test_other_observation_count_without_coordinates_raises(self, stacked_band):
        """Y on fewer observations than X, with no coordinate on observation to
        tell them apart, raises ValueError from the mask, as it does when a
        value is not finite."""
        X, Y, RR = (ds.drop_vars(list(ds.coords)) for ds in stacked_band)
        with pytest.raises(ValueError):
            drop_nans(X, Y.isel(observation=slice(0, -2)), RR)

    def test_y_none_raises(self, stacked_band):
        """Y None raises AttributeError from the mask, as it does when a value is
        not finite."""
        X, _, RR = stacked_band
        with pytest.raises(AttributeError):
            drop_nans(X, None, RR)

    @pytest.mark.parametrize("empty", ["X", "Y", "RR"])
    def test_dataset_without_data_variables_raises(self, stacked_band, empty):
        """X, Y or RR without data variables raises TypeError from the mask, as
        it does when a value is not finite."""
        datasets = dict(zip(("X", "Y", "RR"), stacked_band))
        datasets[empty] = datasets[empty].drop_vars(list(datasets[empty].data_vars))
        with pytest.raises(TypeError):
            drop_nans(datasets["X"], datasets["Y"], datasets["RR"])

    @pytest.mark.parametrize("value", [np.nan, np.inf, complex(0.0, np.nan)])
    def test_same_as_masked_path(self, stacked_band, value):
        """The masked path, made to run by one more observation that is not finite
        in Y, gives the same X, Y and RR."""
        X, Y, RR = stacked_band
        X_ref, Y_ref, RR_ref = drop_nans(
            _append_observation(X),
            _append_observation(Y, value),
            _append_observation(RR),
        )
        X_out, Y_out, RR_out = drop_nans(X, Y, RR)
        assert X_ref.sizes["observation"] == X.sizes["observation"]
        xr.testing.assert_identical(X_out, X_ref)
        xr.testing.assert_identical(Y_out, Y_ref)
        xr.testing.assert_identical(RR_out, RR_ref)


class TestDropNansNotFinite:
    """Test drop_nans when some values are NaN or infinite."""

    def test_drops_observations(self, stacked_band):
        """Each observation with a NaN or infinite value in X, Y or RR is dropped."""
        X, Y, RR = stacked_band
        X["hx"].values[3] = np.nan
        Y["ey"].values[7] = np.inf
        RR["hy"].values[11] = complex(0.0, np.nan)
        keep = np.ones(X.sizes["observation"], dtype=bool)
        keep[[3, 7, 11]] = False

        X_out, Y_out, RR_out = drop_nans(X, Y, RR)

        for out, ds in ((X_out, X), (Y_out, Y), (RR_out, RR)):
            assert out.sizes["observation"] == keep.sum()
            for name in ds.data_vars:
                np.testing.assert_array_equal(out[name].values, ds[name].values[keep])
            np.testing.assert_array_equal(out.time.values, ds.time.values[keep])


# =============================================================================
# EDF weights once per band
# =============================================================================


def _stft(data: dict, n_time: int) -> xr.Dataset:
    """An STFT on time and frequency, the frequencies k / 128 Hz."""
    times = np.datetime64("2023-09-23T00:00:00", "ns") + np.arange(
        n_time
    ) * np.timedelta64(64, "s")
    coords = {"time": times, "frequency": np.arange(33) / 128.0}
    return xr.Dataset(
        {ch: (("time", "frequency"), values) for ch, values in data.items()},
        coords=coords,
    )


@pytest.fixture
def remote_reference_problem():
    """A decimation level with two bands, ex and ey out of hx and hy with a
    remote hx and hy, and the local and remote STFTs."""
    dec_level = DecimationLevel()
    dec_level.input_channels = ["hx", "hy"]
    dec_level.output_channels = ["ex", "ey"]
    dec_level.reference_channels = ["hx", "hy"]
    dec_level.estimator.engine = "RME_RR"
    for index_min, index_max in ((13, 16), (18, 25)):
        dec_level.add_band(
            Band(
                decimation_level=0,
                index_min=index_min,
                index_max=index_max,
                frequency_min=index_min / 128.0,
                frequency_max=index_max / 128.0,
            )
        )

    rng = np.random.default_rng(5)
    n_time = 200

    def noise(scale=1.0):
        shape = (n_time, 33)
        return scale * (rng.normal(size=shape) + 1j * rng.normal(size=shape))

    # windows of larger fields: leverage points, so that some EDF weights are
    # neither 0 nor 1
    scale = np.ones((n_time, 1))
    scale[[10, 40, 41, 120, 160], 0] = [3.0, 4.0, 5.0, 8.0, 12.0]
    hx, hy = scale * noise(), scale * noise()
    local = _stft(
        {
            "hx": hx,
            "hy": hy,
            "ex": 2.0 * hx + 5.0 * hy + noise(0.3),
            "ey": -4.0 * hx + 0.5 * hy + noise(0.3),
        },
        n_time,
    )
    remote = _stft({"hx": hx + noise(0.2), "hy": hy + noise(0.2)}, n_time)
    return dec_level, local, remote


class _RecordingTF:
    """Stands in for TTFZ: keeps the regression of each band and output channel."""

    def __init__(self, frequency_bands):
        self.frequency_bands = frequency_bands
        self.estimates = []

    def set_tf(self, regression_estimator, period):
        output_channel = list(regression_estimator._Y.data_vars)[0]
        self.estimates.append((period, output_channel, regression_estimator.b.copy()))


def _per_channel(dec_level, local, remote) -> list:
    """The EDF weights and the regression of each band and output channel, each
    computed from the channel's own stack of X, Y and RR."""
    estimator_class = get_estimator_class(dec_level.estimator.engine)
    iter_control = set_up_iter_control(dec_level)
    out = []
    for band in dec_level.frequency_bands_obj().bands():
        X_band, Y_band, RR_band = get_band_for_tf_estimate(
            band, dec_level, local, remote
        )
        for ch in dec_level.output_channels:
            X, Y, RR = stack_fcs(X_band, Y_band[ch].to_dataset(), RR_band)
            X, Y, RR = drop_nans(X, Y, RR)
            W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
            weights = W.copy()
            X, Y, RR = apply_weights(X, Y, RR, W, segment=False, dropna=True)
            X, Y, RR = handle_nan(X, Y, RR, drop_dim="observation")
            regression = estimator_class(X=X, Y=Y, Z=RR, iter_control=iter_control)
            regression.estimate()
            out.append((band.center_period, ch, weights, regression.b.copy()))
    return out


def _run_recorded(monkeypatch, dec_level, local, remote) -> tuple:
    """process_transfer_functions_with_weights, counting the EDF weight
    computations and keeping the EDF weights each output channel gets."""
    calls = {"edf": 0, "weights": []}

    def counting_edf(X, R=None, edf_obj=None):
        calls["edf"] += 1
        return effective_degrees_of_freedom_weights(X, R, edf_obj=edf_obj)

    def recording_apply_weights(X, Y, RR, W, segment=False, dropna=False):
        if not segment:
            calls["weights"].append((list(Y.data_vars)[0], np.array(W, copy=True)))
        return apply_weights(X, Y, RR, W, segment=segment, dropna=dropna)

    module = "aurora.pipelines.transfer_function_helpers"
    monkeypatch.setattr(f"{module}.effective_degrees_of_freedom_weights", counting_edf)
    monkeypatch.setattr(f"{module}.apply_weights", recording_apply_weights)
    tf = _RecordingTF(dec_level.frequency_bands_obj())
    process_transfer_functions_with_weights(dec_level, local, remote, tf)
    return calls, tf.estimates


def _assert_same_as_per_channel(reference, calls, estimates):
    assert len(calls["weights"]) == len(reference)
    assert len(estimates) == len(reference)
    for (period, ch, weights, b), (ch_w, W), (period_tf, ch_tf, b_tf) in zip(
        reference, calls["weights"], estimates
    ):
        assert ch_w == ch
        assert (period_tf, ch_tf) == (period, ch)
        np.testing.assert_array_equal(W, weights)
        np.testing.assert_array_equal(b_tf, b)


class TestEdfWeightsOncePerBand:
    """Test the EDF weights of the output channels without channel weights."""

    def test_weights_match_per_channel(self, monkeypatch, remote_reference_problem):
        """Both output channels get the weights, and the regression, of the
        per-channel computation."""
        dec_level, local, remote = remote_reference_problem
        reference = _per_channel(dec_level, local, remote)
        assert any(np.any((w > 0) & (w < 1)) for _, _, w, _ in reference)
        calls, estimates = _run_recorded(monkeypatch, dec_level, local, remote)
        _assert_same_as_per_channel(reference, calls, estimates)

    def test_edf_weights_computed_once_per_band(
        self, monkeypatch, remote_reference_problem
    ):
        """With every value finite, the weights are computed once per band, not
        once per output channel."""
        dec_level, local, remote = remote_reference_problem
        calls, _ = _run_recorded(monkeypatch, dec_level, local, remote)
        assert calls["edf"] == len(dec_level.bands)

    def test_nan_in_one_output_channel(self, monkeypatch, remote_reference_problem):
        """A NaN in ex drops observations for ex alone: ex gets weights of its
        own and ey those of the full band, as in the per-channel computation."""
        dec_level, local, remote = remote_reference_problem
        local["ex"].values[[3, 50], 14] = np.nan
        reference = _per_channel(dec_level, local, remote)
        assert reference[0][2].size < reference[1][2].size
        calls, estimates = _run_recorded(monkeypatch, dec_level, local, remote)
        _assert_same_as_per_channel(reference, calls, estimates)
