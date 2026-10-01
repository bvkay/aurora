"""
Pytest suite for the NaN handling in transfer_function_helpers.

Tests cover:
- drop_nans: dropping the observations with a value that is not finite in X, Y
  or RR, and returning the inputs when every value is finite
"""

import numpy as np
import pytest
import xarray as xr

from aurora.pipelines.transfer_function_helpers import drop_nans, stack_fcs


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
