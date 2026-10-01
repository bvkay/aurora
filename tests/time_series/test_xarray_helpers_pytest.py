"""
Pytest suite for testing xarray_helpers module.

Tests cover:
- nan_to_mean: Replacing NaN values with channel means
- handle_nan: Dropping NaN values across multiple datasets
- time_axis_match: Checking time coordinate alignment
- Edge cases and parameter variations

Optimized for pytest-xdist parallel execution.
"""

import numpy as np
import pytest
import xarray as xr

from aurora.time_series.xarray_helpers import handle_nan, nan_to_mean, time_axis_match


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def basic_times():
    """Basic time coordinate array."""
    return np.array([0, 1, 2, 3])


@pytest.fixture
def extended_times():
    """Extended time coordinate array for edge case testing."""
    return np.array([0, 1, 2, 3, 4])


@pytest.fixture
def single_channel_dataset_with_nan(basic_times):
    """Dataset with single channel containing NaN values."""
    data = np.array([1.0, np.nan, 3.0, 4.0])
    return xr.Dataset({"hx": ("time", data)}, coords={"time": basic_times})


@pytest.fixture
def multi_channel_dataset_with_nan(basic_times):
    """Dataset with multiple channels containing NaN values in different locations."""
    data_hx = np.array([1.0, np.nan, 3.0, 4.0])
    data_hy = np.array([np.nan, 2.0, 3.0, 4.0])
    return xr.Dataset(
        {
            "hx": ("time", data_hx),
            "hy": ("time", data_hy),
        },
        coords={"time": basic_times},
    )


@pytest.fixture
def dataset_no_nan(basic_times):
    """Dataset without any NaN values."""
    data = np.array([1.0, 2.0, 3.0, 4.0])
    return xr.Dataset({"hx": ("time", data)}, coords={"time": basic_times})


@pytest.fixture
def dataset_all_nan(basic_times):
    """Dataset with all NaN values."""
    data = np.array([np.nan, np.nan, np.nan, np.nan])
    return xr.Dataset({"hx": ("time", data)}, coords={"time": basic_times})


# =============================================================================
# Test Classes
# =============================================================================


class TestNanToMean:
    """Test nan_to_mean function."""

    def test_nan_to_mean_basic(self, single_channel_dataset_with_nan):
        """Test nan_to_mean replaces NaNs with mean per channel."""
        ds_filled = nan_to_mean(single_channel_dataset_with_nan.copy())

        # The mean ignoring NaN is (1+3+4)/3 = 2.666...
        expected = np.array([1.0, 2.66666667, 3.0, 4.0])
        assert np.allclose(ds_filled.hx.values, expected)

        # No NaNs should remain
        assert not np.any(np.isnan(ds_filled.hx.values))

    def test_nan_to_mean_multiple_channels(self, multi_channel_dataset_with_nan):
        """Test nan_to_mean with multiple channels and NaNs in different places."""
        ds_filled = nan_to_mean(multi_channel_dataset_with_nan.copy())

        expected_hx = np.array([1.0, 2.66666667, 3.0, 4.0])
        expected_hy = np.array([3.0, 2.0, 3.0, 4.0])

        assert np.allclose(ds_filled.hx.values, expected_hx)
        assert np.allclose(ds_filled.hy.values, expected_hy)
        assert not np.any(np.isnan(ds_filled.hx.values))
        assert not np.any(np.isnan(ds_filled.hy.values))

    def test_nan_to_mean_no_nans(self, dataset_no_nan):
        """Test nan_to_mean with dataset containing no NaN values."""
        original_data = dataset_no_nan.hx.values.copy()
        ds_filled = nan_to_mean(dataset_no_nan.copy())

        # Data should remain unchanged
        assert np.allclose(ds_filled.hx.values, original_data)
        assert not np.any(np.isnan(ds_filled.hx.values))

    def test_nan_to_mean_all_nans(self, dataset_all_nan):
        """Test nan_to_mean with dataset containing all NaN values."""
        ds_filled = nan_to_mean(dataset_all_nan.copy())

        # Should replace with 0 (from np.nan_to_num of nanmean)
        assert np.allclose(ds_filled.hx.values, 0.0)

    def test_nan_to_mean_preserves_structure(self, multi_channel_dataset_with_nan):
        """Test that nan_to_mean preserves dataset structure."""
        ds_filled = nan_to_mean(multi_channel_dataset_with_nan.copy())

        # Check that coordinates are preserved
        assert np.allclose(
            ds_filled.time.values, multi_channel_dataset_with_nan.time.values
        )

        # Check that channels are preserved
        assert set(ds_filled.data_vars) == set(multi_channel_dataset_with_nan.data_vars)

    def test_nan_to_mean_single_nan_at_edges(self, subtests):
        """Test nan_to_mean with NaN at beginning and end."""
        times = np.array([0, 1, 2, 3, 4])

        test_cases = [
            (
                "nan_at_start",
                np.array([np.nan, 2.0, 3.0, 4.0, 5.0]),
                np.array([3.5, 2.0, 3.0, 4.0, 5.0]),
            ),
            (
                "nan_at_end",
                np.array([1.0, 2.0, 3.0, 4.0, np.nan]),
                np.array([1.0, 2.0, 3.0, 4.0, 2.5]),
            ),
            (
                "nan_at_both",
                np.array([np.nan, 2.0, 3.0, 4.0, np.nan]),
                np.array([3.0, 2.0, 3.0, 4.0, 3.0]),
            ),
        ]

        for name, data, expected in test_cases:
            with subtests.test(case=name):
                ds = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
                ds_filled = nan_to_mean(ds.copy())
                assert np.allclose(ds_filled.hx.values, expected)


class TestHandleNanBasic:
    """Test basic handle_nan functionality."""

    def test_handle_nan_basic(self, extended_times):
        """Test basic functionality of handle_nan with NaN values."""
        data_x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        data_y = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": extended_times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": extended_times})

        # Test with X and Y only
        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Check that NaN values were dropped
        assert len(X_clean.time) == 3
        assert len(Y_clean.time) == 3
        assert not np.any(np.isnan(X_clean.hx.values))
        assert not np.any(np.isnan(Y_clean.ex.values))

        # Check that correct values remain
        expected_times = np.array([0, 3, 4])
        assert np.allclose(X_clean.time.values, expected_times)
        assert np.allclose(Y_clean.time.values, expected_times)

    def test_handle_nan_x_only(self):
        """Test handle_nan with only X dataset (Y empty, RR None)."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        # Empty dataset with matching time coordinate
        Y = xr.Dataset(coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, None, drop_dim="time")

        # Check that NaN was dropped from X
        assert len(X_clean.time) == 3
        assert not np.any(np.isnan(X_clean.hx.values))

        # Y and RR should be empty datasets
        assert len(Y_clean.data_vars) == 0
        assert len(RR_clean.data_vars) == 0

    def test_handle_nan_no_nans(self):
        """Test handle_nan with datasets containing no NaN values."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # All data should be preserved
        assert len(X_clean.time) == 4
        assert len(Y_clean.time) == 4
        assert np.allclose(X_clean.hx.values, data_x)
        assert np.allclose(Y_clean.ex.values, data_y)


class TestHandleNanRemoteReference:
    """Test handle_nan with remote reference data."""

    def test_handle_nan_with_remote_reference(self):
        """Test handle_nan with remote reference data."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, 2.0, np.nan, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})
        RR = xr.Dataset({"hx": ("time", data_rr)}, coords={"time": times})

        # Test with all datasets
        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # Check that NaN values were dropped
        assert len(X_clean.time) == 2
        assert len(Y_clean.time) == 2
        assert len(RR_clean.time) == 2
        assert not np.any(np.isnan(X_clean.hx.values))
        assert not np.any(np.isnan(Y_clean.ex.values))
        assert not np.any(np.isnan(RR_clean.hx.values))

        # Check that the values are correct
        expected_times = np.array([0, 3])
        assert np.allclose(X_clean.time.values, expected_times)
        assert np.allclose(Y_clean.time.values, expected_times)
        assert np.allclose(RR_clean.time.values, expected_times)
        assert np.allclose(X_clean.hx.values, np.array([1.0, 4.0]))
        assert np.allclose(Y_clean.ex.values, np.array([1.0, 4.0]))
        assert np.allclose(RR_clean.hx.values, np.array([1.0, 4.0]))

    def test_handle_nan_remote_reference_only(self):
        """Test handle_nan with only remote reference having NaN."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, np.nan, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data_rr)}, coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # Only time index 1 should be dropped
        assert len(X_clean.time) == 3
        assert len(Y_clean.time) == 3
        assert len(RR_clean.time) == 3

        expected_times = np.array([0, 2, 3])
        assert np.allclose(X_clean.time.values, expected_times)

    def test_handle_nan_channel_name_preservation(self):
        """Test that channel names are preserved correctly with RR."""
        times = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data)}, coords={"time": times})
        RR = xr.Dataset(
            {"hx": ("time", data), "hy": ("time", data)}, coords={"time": times}
        )

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # Check channel names
        assert "hx" in X_clean.data_vars
        assert "ex" in Y_clean.data_vars
        assert "hx" in RR_clean.data_vars
        assert "hy" in RR_clean.data_vars

        # RR channels should not have "remote_" prefix in output
        assert "remote_hx" not in RR_clean.data_vars


class TestHandleNanTimeMismatch:
    """Test handle_nan with time coordinate mismatches."""

    def test_handle_nan_time_mismatch(self):
        """Test handle_nan with time coordinate mismatches."""
        times_x = np.array([0, 1, 2, 3])
        times_rr = times_x + 0.1  # Small offset
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times_x})
        RR = xr.Dataset({"hx": ("time", data_rr)}, coords={"time": times_rr})

        # Test handling of time mismatch
        X_clean, _, RR_clean = handle_nan(X, None, RR, drop_dim="time")

        # Check that data was preserved despite time mismatch
        assert len(X_clean.time) == 4
        assert "hx" in RR_clean.data_vars
        assert np.allclose(RR_clean.hx.values, data_rr)

        # Check that the time values match X's time values
        assert np.allclose(RR_clean.time.values, X_clean.time.values)

    def test_handle_nan_partial_time_mismatch(self):
        """Test handle_nan when only some time coordinates mismatch."""
        times_x = np.array([0.0, 1.0, 2.0, 3.0])
        times_rr = np.array([0.0, 1.0, 2.0001, 3.0])  # Slight mismatch at index 2
        data_x = np.array([1.0, 2.0, 3.0, 4.0])
        data_rr = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times_x})
        RR = xr.Dataset({"hy": ("time", data_rr)}, coords={"time": times_rr})

        # Should handle this with left join
        X_clean, _, RR_clean = handle_nan(X, None, RR, drop_dim="time")

        assert len(X_clean.time) == 4
        assert len(RR_clean.time) == 4


class TestTimeAxisMatch:
    """Test time_axis_match function."""

    def test_time_axis_match_exact(self):
        """Test time_axis_match when all axes match exactly."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times})

        assert time_axis_match(X, Y, RR) is True

    def test_time_axis_match_xy_only(self):
        """Test time_axis_match with only X and Y."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data)}, coords={"time": times})

        assert time_axis_match(X, Y, None) is True

    def test_time_axis_match_x_rr_only(self):
        """Test time_axis_match with only X and RR."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times})

        assert time_axis_match(X, None, RR) is True

    def test_time_axis_match_mismatch(self):
        """Test time_axis_match when axes do not match."""
        times_x = np.array([0, 1, 2, 3])
        times_rr = np.array([0, 1, 2, 4])  # Different last value
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times_x})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times_rr})

        assert time_axis_match(X, None, RR) is False

    def test_time_axis_match_different_lengths(self):
        """Test time_axis_match with different length time axes."""
        times_x = np.array([0, 1, 2, 3])
        times_y = np.array([0, 1, 2])

        X = xr.Dataset(
            {"hx": ("time", np.array([1.0, 2.0, 3.0, 4.0]))}, coords={"time": times_x}
        )
        Y = xr.Dataset(
            {"ex": ("time", np.array([1.0, 2.0, 3.0]))}, coords={"time": times_y}
        )
        RR = xr.Dataset(
            {"hy": ("time", np.array([1.0, 2.0, 3.0, 4.0]))}, coords={"time": times_x}
        )

        # Use RR instead of None to avoid AttributeError
        assert time_axis_match(X, Y, RR) is False

    def test_time_axis_match_float_precision(self):
        """Test time_axis_match with floating point precision issues."""
        times_x = np.array([0.0, 0.1, 0.2, 0.3])
        times_rr = times_x + 1e-10  # Very small difference
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times_x})
        RR = xr.Dataset({"hy": ("time", data)}, coords={"time": times_rr})

        # Should not match due to precision difference
        assert time_axis_match(X, None, RR) is False


class TestHandleNanMultipleChannels:
    """Test handle_nan with multiple channels in each dataset."""

    def test_handle_nan_multiple_channels_x_y(self):
        """Test handle_nan with multiple channels in X and Y."""
        times = np.array([0, 1, 2, 3])
        data_hx = np.array([1.0, np.nan, 3.0, 4.0])
        data_hy = np.array([1.0, 2.0, np.nan, 4.0])
        data_ex = np.array([np.nan, 2.0, 3.0, 4.0])
        data_ey = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset(
            {
                "hx": ("time", data_hx),
                "hy": ("time", data_hy),
            },
            coords={"time": times},
        )

        Y = xr.Dataset(
            {
                "ex": ("time", data_ex),
                "ey": ("time", data_ey),
            },
            coords={"time": times},
        )

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Only time index 3 has no NaN in any channel
        assert len(X_clean.time) == 1
        assert len(Y_clean.time) == 1
        assert X_clean.time.values[0] == 3

    def test_handle_nan_preserves_all_channels(self):
        """Test that all channels are preserved after NaN handling."""
        times = np.array([0, 1, 2])
        data = np.array([1.0, 2.0, 3.0])

        X = xr.Dataset(
            {
                "hx": ("time", data),
                "hy": ("time", data),
                "hz": ("time", data),
            },
            coords={"time": times},
        )

        Y = xr.Dataset(
            {
                "ex": ("time", data),
                "ey": ("time", data),
            },
            coords={"time": times},
        )

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # All channels should be preserved
        assert set(X_clean.data_vars) == {"hx", "hy", "hz"}
        assert set(Y_clean.data_vars) == {"ex", "ey"}


class TestHandleNanEdgeCases:
    """Test edge cases for handle_nan."""

    def test_handle_nan_empty_dataset(self):
        """Test handle_nan with empty Y and RR."""
        times = np.array([0, 1, 2, 3])
        data = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data)}, coords={"time": times})
        # Empty dataset with matching time coordinate
        Y = xr.Dataset(coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, None, drop_dim="time")

        # X should be unchanged
        assert len(X_clean.time) == 4
        assert np.allclose(X_clean.hx.values, data)

        # Y and RR should be empty
        assert len(Y_clean.data_vars) == 0
        assert len(RR_clean.data_vars) == 0

    def test_handle_nan_all_nans_dropped(self):
        """Test handle_nan when all rows have at least one NaN."""
        times = np.array([0, 1, 2])
        data_x = np.array([np.nan, 2.0, 3.0])
        data_y = np.array([1.0, np.nan, 3.0])
        data_rr = np.array([1.0, 2.0, np.nan])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})
        RR = xr.Dataset({"hy": ("time", data_rr)}, coords={"time": times})

        X_clean, Y_clean, RR_clean = handle_nan(X, Y, RR, drop_dim="time")

        # No rows should remain
        assert len(X_clean.time) == 0
        assert len(Y_clean.time) == 0
        assert len(RR_clean.time) == 0

    def test_handle_nan_different_drop_dim(self):
        """Test handle_nan still works when drop_dim is specified (even though time_axis_match assumes 'time')."""
        # Note: time_axis_match function assumes 'time' dimension exists, so we use 'time' here
        # but test that drop_dim parameter is respected
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # NaN at index 1 should be dropped
        assert len(X_clean.time) == 3
        assert len(Y_clean.time) == 3

        expected_times = np.array([0, 2, 3])
        assert np.allclose(X_clean.time.values, expected_times)


class TestHandleNanDataIntegrity:
    """Test that handle_nan preserves data integrity."""

    def test_handle_nan_values_correctness(self):
        """Test that correct values are preserved after dropping NaNs."""
        times = np.array([0, 1, 2, 3, 4])
        data_x = np.array([10.0, np.nan, 30.0, np.nan, 50.0])
        data_y = np.array([100.0, 200.0, np.nan, 400.0, 500.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Only times 0 and 4 have no NaN in any channel
        expected_times = np.array([0, 4])
        expected_x = np.array([10.0, 50.0])
        expected_y = np.array([100.0, 500.0])

        assert np.allclose(X_clean.time.values, expected_times)
        assert np.allclose(X_clean.hx.values, expected_x)
        assert np.allclose(Y_clean.ex.values, expected_y)

    def test_handle_nan_original_unchanged(self):
        """Test that original datasets are not modified by handle_nan."""
        times = np.array([0, 1, 2, 3])
        data_x = np.array([1.0, np.nan, 3.0, 4.0])
        data_y = np.array([1.0, 2.0, 3.0, 4.0])

        X = xr.Dataset({"hx": ("time", data_x)}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", data_y)}, coords={"time": times})

        # Store original values
        original_x_len = len(X.time)
        original_y_len = len(Y.time)

        # Call handle_nan
        X_clean, Y_clean, _ = handle_nan(X, Y, None, drop_dim="time")

        # Original datasets should be unchanged
        assert len(X.time) == original_x_len
        assert len(Y.time) == original_y_len
        assert np.isnan(X.hx.values[1])  # NaN still present


def _stacked_fcs(channels, seed, n_observation=60):
    """Stacked Fourier coefficients: complex data on an observation dimension
    with the frequency and time of each observation as coordinates."""
    rng = np.random.default_rng(seed)
    n_time = n_observation // 2
    coords = {
        "frequency": ("observation", np.repeat([0.10, 0.11], n_time)),
        "time": ("observation", np.tile(np.arange(n_time), 2)),
    }
    data_vars = {
        ch: (
            "observation",
            rng.normal(size=n_observation) + 1j * rng.normal(size=n_observation),
        )
        for ch in channels
    }
    return xr.Dataset(data_vars, coords=coords)


class TestHandleNanAllFinite:
    """Test handle_nan on finite data, and that it keeps infinite values."""

    def test_handle_nan_all_finite_returns_inputs(self):
        """With Y and RR, matching time axes and no NaN, the inputs are returned as they are."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["ex"], seed=2)
        RR = _stacked_fcs(["hx", "hy"], seed=3)

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        assert X_out is X
        assert Y_out is Y
        assert RR_out is RR

    def test_handle_nan_all_finite_same_as_merge_path(self):
        """The merge path, made to run by one more observation with a NaN in Y, gives the same X, Y and RR."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["ex", "ey"], seed=2)
        RR = _stacked_fcs(["hx", "hy"], seed=3)

        def append_observation(ds, value=None):
            extra = ds.isel(observation=[0]).copy(deep=True)
            if value is not None:
                for name in extra.data_vars:
                    extra[name].values[:] = value
            return xr.concat([ds, extra], dim="observation")

        X_ref, Y_ref, RR_ref = handle_nan(
            append_observation(X),
            append_observation(Y, np.nan),
            append_observation(RR),
            drop_dim="observation",
        )
        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        assert X_ref.sizes["observation"] == X.sizes["observation"]
        xr.testing.assert_identical(X_out, X_ref)
        xr.testing.assert_identical(Y_out, Y_ref)
        xr.testing.assert_identical(RR_out, RR_ref)

    def test_handle_nan_all_finite_mismatched_time_axes_merged(self):
        """With Y and RR given and every value finite, but RR on other times, the
        merge runs: RR comes back on X's times with its own values."""
        times = np.array([0.0, 1.0, 2.0, 3.0])
        X = xr.Dataset({"hx": ("time", [1.0, 2.0, 3.0, 4.0])}, coords={"time": times})
        Y = xr.Dataset({"ex": ("time", [5.0, 6.0, 7.0, 8.0])}, coords={"time": times})
        RR = xr.Dataset(
            {"hx": ("time", [9.0, 10.0, 11.0, 12.0])}, coords={"time": times + 0.1}
        )

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="time")

        assert RR_out is not RR
        np.testing.assert_array_equal(RR_out.time.values, times)
        np.testing.assert_array_equal(RR_out.hx.values, RR.hx.values)
        np.testing.assert_array_equal(X_out.hx.values, X.hx.values)
        np.testing.assert_array_equal(Y_out.ex.values, Y.ex.values)

    @pytest.mark.parametrize("drop_dim", ["", "time"])
    def test_handle_nan_all_finite_drop_dim_not_a_dimension_raises(self, drop_dim):
        """A drop_dim that is not a dimension of the data raises ValueError, as it
        does when there is a NaN to drop."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["ex"], seed=2)
        RR = _stacked_fcs(["hx", "hy"], seed=3)

        with pytest.raises(ValueError):
            handle_nan(X, Y, RR, drop_dim=drop_dim)

    def test_handle_nan_all_finite_mixed_dtypes_cast_as_merge(self):
        """A real Y with complex X and RR comes back complex, as from the merge."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["ex"], seed=2)
        Y["ex"] = Y["ex"].real
        RR = _stacked_fcs(["hx", "hy"], seed=3)

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        assert Y_out["ex"].dtype == np.complex128
        np.testing.assert_array_equal(Y_out["ex"].values, Y["ex"].values)
        xr.testing.assert_identical(X_out, X)
        xr.testing.assert_identical(RR_out, RR)

    def test_handle_nan_infinite_values_kept(self):
        """Only NaN is dropped: an infinite value stays, as before."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["ex"], seed=2)
        RR = _stacked_fcs(["hx", "hy"], seed=3)
        X["hx"].values[5] = np.inf
        RR["hy"].values[9] = np.nan

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        keep = np.ones(X.sizes["observation"], dtype=bool)
        keep[9] = False
        assert X_out.sizes["observation"] == keep.sum()
        assert np.isinf(X_out["hx"].values[5])
        np.testing.assert_array_equal(X_out["hx"].values, X["hx"].values[keep])
        np.testing.assert_array_equal(Y_out["ex"].values, Y["ex"].values[keep])
        np.testing.assert_array_equal(RR_out["hx"].values, RR["hx"].values[keep])


def _stft_fcs(channels, seed, frequencies, dims=("time", "frequency")):
    """Unstacked Fourier coefficients: complex data on time and frequency."""
    rng = np.random.default_rng(seed)
    times = np.arange(5).astype("datetime64[s]").astype("datetime64[ns]")
    sizes = {"time": len(times), "frequency": len(frequencies)}
    shape = tuple(sizes[d] for d in dims)
    data_vars = {
        ch: (dims, rng.normal(size=shape) + 1j * rng.normal(size=shape))
        for ch in channels
    }
    return xr.Dataset(data_vars, coords={"time": times, "frequency": frequencies})


class TestHandleNanAllFiniteOtherLayouts:
    """Test that all-finite inputs in another layout than the regression's still
    take the merge, which raises or rewrites them as before."""

    FREQUENCIES = np.array([0.1, 0.2, 0.3])

    @pytest.mark.parametrize(
        "Y_frequencies, RR_frequencies",
        [(FREQUENCIES + 0.05, FREQUENCIES), (FREQUENCIES, FREQUENCIES[:2])],
        ids=["Y", "RR"],
    )
    def test_other_frequencies_raise(self, Y_frequencies, RR_frequencies):
        """3D STFTs with Y or RR on other frequencies than X, dropped on time,
        raise ValueError from the exact join of the merge."""
        X = _stft_fcs(["hx", "hy"], 1, self.FREQUENCIES)
        Y = _stft_fcs(["ex"], 2, Y_frequencies)
        RR = _stft_fcs(["hx", "hy"], 3, RR_frequencies)

        with pytest.raises(ValueError):
            handle_nan(X, Y, RR, drop_dim="time")

    def test_transposed_y_in_x_dimension_order(self):
        """A 3D Y on frequency and time comes back in X's order, time and frequency."""
        X = _stft_fcs(["hx", "hy"], 1, self.FREQUENCIES)
        Y = _stft_fcs(["ex"], 2, self.FREQUENCIES, dims=("frequency", "time"))
        RR = _stft_fcs(["hx", "hy"], 3, self.FREQUENCIES)

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="time")

        assert Y_out["ex"].dims == ("time", "frequency")
        np.testing.assert_array_equal(Y_out["ex"].values, Y["ex"].values.T)

    def test_y_name_of_x_takes_x_values(self):
        """A data variable of Y named as one of X's comes back with X's values."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["hx"], seed=2)
        RR = _stacked_fcs(["hx", "hy"], seed=3)

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        np.testing.assert_array_equal(Y_out["hx"].values, X["hx"].values)

    def test_variable_attributes_dropped(self):
        """A data variable with attributes, under the same dataset attributes,
        comes back without them, as from the merge."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["ex"], seed=2)
        RR = _stacked_fcs(["hx", "hy"], seed=3)
        X["hx"].attrs["units"] = "nT"

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        assert X_out["hx"].attrs == {}
        np.testing.assert_array_equal(X_out["hx"].values, X["hx"].values)

    def test_dataset_attributes_of_x(self):
        """Y and RR with other dataset attributes than X, and no data variable
        with attributes, come back with X's dataset attributes."""
        X = _stacked_fcs(["hx", "hy"], seed=1).assign_attrs(station="x")
        Y = _stacked_fcs(["ex"], seed=2).assign_attrs(station="y")
        RR = _stacked_fcs(["hx", "hy"], seed=3)

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        assert Y_out.attrs == {"station": "x"}
        assert RR_out.attrs == {"station": "x"}

    def test_coordinates_of_x(self):
        """Y and RR come back with X's coordinates, here without the attributes
        of their time coordinates, which time_axis_match does not compare."""
        X = _stacked_fcs(["hx", "hy"], seed=1)
        Y = _stacked_fcs(["ex"], seed=2)
        RR = _stacked_fcs(["hx", "hy"], seed=3)
        Y["time"].attrs["units"] = "s"
        RR["time"].attrs["units"] = "s"

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        assert Y_out["time"].attrs == {}
        assert RR_out["time"].attrs == {}

    def test_same_attributes_return_inputs(self):
        """In the regression's layout, with the same dataset attributes on X, Y
        and RR, the inputs are still returned as they are."""
        X = _stacked_fcs(["hx", "hy"], seed=1).assign_attrs(station="x")
        Y = _stacked_fcs(["ex"], seed=2).assign_attrs(station="x")
        RR = _stacked_fcs(["hx", "hy"], seed=3).assign_attrs(station="x")

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        assert X_out is X
        assert Y_out is Y
        assert RR_out is RR

    @pytest.mark.parametrize(
        "channel",
        [("observation", np.arange(60)), "a"],
        ids=["on_observation", "scalar"],
    )
    def test_channel_coordinate_dropped(self, channel):
        """A coordinate named "channel", the dimension of the merge's to_array,
        is dropped from X, Y and RR, as by the merge."""
        X = _stacked_fcs(["hx", "hy"], seed=1).assign_coords(channel=channel)
        Y = _stacked_fcs(["ex"], seed=2).assign_coords(channel=channel)
        RR = _stacked_fcs(["hx", "hy"], seed=3).assign_coords(channel=channel)

        X_out, Y_out, RR_out = handle_nan(X, Y, RR, drop_dim="observation")

        for ds in (X_out, Y_out, RR_out):
            assert "channel" not in ds.coords

    @pytest.mark.parametrize("drop_on_channel", [True, False])
    def test_channel_dimension_raises(self, drop_on_channel):
        """A dimension named "channel", the data's own or a coordinate's, clashes
        with the merge's to_array and raises ValueError."""
        datasets = [
            _stacked_fcs(chs, seed)
            for chs, seed in ((["hx", "hy"], 1), (["ex"], 2), (["hx", "hy"], 3))
        ]
        if drop_on_channel:
            dim = "channel"
            datasets = [ds.rename_dims(observation=dim) for ds in datasets]
        else:
            dim = "observation"
            datasets = [ds.assign_coords(extra=("channel", [1, 2])) for ds in datasets]

        with pytest.raises(ValueError):
            handle_nan(*datasets, drop_dim=dim)

    @pytest.mark.parametrize("all_integer", [True, False], ids=["all", "Y"])
    def test_integer_channel_names_raise(self, all_integer):
        """Integer channel names, of all datasets or of Y only, make the split
        of the merge raise KeyError."""
        X = _stacked_fcs([0, 1] if all_integer else ["hx", "hy"], seed=1)
        Y = _stacked_fcs([2], seed=2)
        RR = _stacked_fcs([0, 1] if all_integer else ["hx", "hy"], seed=3)

        with pytest.raises(KeyError):
            handle_nan(X, Y, RR, drop_dim="observation")

    def test_coordinate_named_as_remote_channel_raises(self):
        """A coordinate named as a renamed RR channel ("remote_hx") clashes with
        it in the merge and raises ValueError."""
        remote_hx = ("observation", np.arange(60))
        X = _stacked_fcs(["hx", "hy"], seed=1).assign_coords(remote_hx=remote_hx)
        Y = _stacked_fcs(["ex"], seed=2).assign_coords(remote_hx=remote_hx)
        RR = _stacked_fcs(["hx", "hy"], seed=3).assign_coords(remote_hx=remote_hx)

        with pytest.raises(ValueError):
            handle_nan(X, Y, RR, drop_dim="observation")
