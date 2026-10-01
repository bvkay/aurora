"""
Pytest suite for prototype_decimate in time_series_helpers.

Tests cover:
- the decimated channels are those of scipy.signal.decimate on each channel in
  turn, in the order of the input's data variables
- the channels are decimated in worker threads, at most DECIMATE_MAX_WORKERS
  at once, and fewer, or in turn, when three copies of a channel per thread
  exceed DECIMATE_MAX_TRANSIENT_BYTES
"""

import threading
import time

import numpy as np
import pytest
import scipy.signal as ssig
import xarray as xr
from mt_metadata.processing import TimeSeriesDecimation

from aurora.pipelines import time_series_helpers
from aurora.pipelines.time_series_helpers import prototype_decimate


@pytest.fixture
def run_xrds():
    """Six channels of 1 Hz noise with a sine, 4000 samples."""
    rng = np.random.default_rng(7)
    n_samples = 4000
    times = np.datetime64("2023-09-23T00:00:00", "ns") + np.arange(
        n_samples
    ) * np.timedelta64(1, "s")
    t = np.arange(n_samples)
    data_vars = {
        ch: ("time", np.sin(2 * np.pi * t / (50 + 10 * i)) + rng.normal(size=n_samples))
        for i, ch in enumerate(["hx", "hy", "hz", "ex", "ey", "temperature"])
    }
    return xr.Dataset(
        data_vars, coords={"time": times}, attrs={"sample_rate": 1.0, "station": "x"}
    )


@pytest.fixture
def ts_decimation():
    """Decimation by 4, to 0.25 Hz."""
    decimation = TimeSeriesDecimation()
    decimation.factor = 4
    decimation.sample_rate = 0.25
    return decimation


@pytest.fixture
def concurrency(monkeypatch):
    """Replace scipy.signal.decimate in time_series_helpers by a slow wrapper that
    records the threads it runs in and the most calls running at once."""
    decimate = ssig.decimate
    lock = threading.Lock()
    state = {"running": 0, "most": 0, "threads": set()}

    def slow_decimate(*args, **kwargs):
        with lock:
            state["running"] += 1
            state["most"] = max(state["most"], state["running"])
            state["threads"].add(threading.get_ident())
        time.sleep(0.2)
        try:
            return decimate(*args, **kwargs)
        finally:
            with lock:
                state["running"] -= 1

    monkeypatch.setattr(time_series_helpers.ssig, "decimate", slow_decimate)
    return state


# The filter memory of one thread on the 4000 samples of the run_xrds fixture,
# counted as decimate_num_workers counts it.
ONE_THREAD_BYTES = 3 * 8 * 4000


class TestPrototypeDecimate:
    """Test prototype_decimate."""

    def test_same_as_serial_decimate(self, run_xrds, ts_decimation):
        """Each channel is scipy.signal.decimate of the channel, bit for bit, on
        every fourth time stamp."""
        decimated = prototype_decimate(ts_decimation, run_xrds.copy())

        assert list(decimated.data_vars) == list(run_xrds.data_vars)
        expected_times = run_xrds.time.values[::4]
        np.testing.assert_array_equal(decimated.time.values, expected_times)
        for ch in run_xrds.data_vars:
            expected = ssig.decimate(run_xrds[ch].values, 4)
            np.testing.assert_array_equal(decimated[ch].values, expected)
        assert decimated.attrs["sample_rate"] == 0.25

    def test_channels_decimated_in_bounded_thread_pool(
        self, concurrency, run_xrds, ts_decimation
    ):
        """The channels are decimated in worker threads, more than one at once
        and never more than DECIMATE_MAX_WORKERS."""
        prototype_decimate(ts_decimation, run_xrds)

        max_workers = getattr(time_series_helpers, "DECIMATE_MAX_WORKERS", 1)
        assert threading.get_ident() not in concurrency["threads"]
        assert 1 < concurrency["most"] <= max_workers
        assert len(concurrency["threads"]) <= max_workers

    def test_threads_bounded_by_transient_memory(
        self, monkeypatch, concurrency, run_xrds, ts_decimation
    ):
        """With room for the filter memory of two threads, no more than two
        channels are decimated at once."""
        monkeypatch.setattr(
            time_series_helpers,
            "DECIMATE_MAX_TRANSIENT_BYTES",
            2 * ONE_THREAD_BYTES,
            raising=False,
        )
        decimated = prototype_decimate(ts_decimation, run_xrds.copy())

        assert concurrency["most"] == 2
        assert len(concurrency["threads"]) == 2
        for ch in run_xrds.data_vars:
            expected = ssig.decimate(run_xrds[ch].values, 4)
            np.testing.assert_array_equal(decimated[ch].values, expected)

    def test_record_over_transient_memory_decimated_in_turn(
        self, monkeypatch, concurrency, run_xrds, ts_decimation
    ):
        """When the filter memory of one thread exceeds
        DECIMATE_MAX_TRANSIENT_BYTES, the channels are decimated in the calling
        thread, one after the other."""
        monkeypatch.setattr(
            time_series_helpers,
            "DECIMATE_MAX_TRANSIENT_BYTES",
            ONE_THREAD_BYTES - 1,
            raising=False,
        )
        prototype_decimate(ts_decimation, run_xrds)

        assert concurrency["threads"] == {threading.get_ident()}
        assert concurrency["most"] == 1

    @pytest.mark.parametrize(
        "num_channels, hours, expected",
        [(4, 6.0, 4), (6, 6.0, 4), (4, 44.9, 2), (4, 72.0, 1), (2, 0.1, 2)],
    )
    def test_default_workers_by_record_length(
        self, monkeypatch, num_channels, hours, expected
    ):
        """With the default settings and 20 CPUs, a 1000 Hz record of 6 h is
        decimated in 4 threads, one of 44.9 h in 2 and one of 72 h in turn."""
        monkeypatch.setattr(time_series_helpers.os, "cpu_count", lambda: 20)
        num_samples = int(hours * 3600 * 1000)

        assert (
            time_series_helpers.decimate_num_workers(num_channels, num_samples)
            == expected
        )

    @pytest.mark.parametrize("cpus, expected", [(2, 2), (None, 1)])
    def test_workers_bounded_by_cpus(self, monkeypatch, cpus, expected):
        """With 2 CPUs, the 4 channels of a 6 h record at 1000 Hz are decimated in
        2 threads; when the CPUs are unknown, in turn."""
        monkeypatch.setattr(time_series_helpers.os, "cpu_count", lambda: cpus)
        num_samples = int(6.0 * 3600 * 1000)

        assert time_series_helpers.decimate_num_workers(4, num_samples) == expected

    def test_one_worker_decimates_in_turn(self, monkeypatch, run_xrds, ts_decimation):
        """With DECIMATE_MAX_WORKERS = 1 the channels are decimated in the calling
        thread, with the same result."""
        threaded = prototype_decimate(ts_decimation, run_xrds.copy())
        monkeypatch.setattr(
            time_series_helpers, "DECIMATE_MAX_WORKERS", 1, raising=False
        )
        threads = set()
        decimate = ssig.decimate

        def recording_decimate(*args, **kwargs):
            threads.add(threading.get_ident())
            return decimate(*args, **kwargs)

        monkeypatch.setattr(time_series_helpers.ssig, "decimate", recording_decimate)
        serial = prototype_decimate(ts_decimation, run_xrds.copy())

        assert threads == {threading.get_ident()}
        xr.testing.assert_identical(serial, threaded)
