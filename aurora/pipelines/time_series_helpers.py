"""
    Collection of modules used in processing pipeline that operate on time series
"""
import os
from concurrent.futures import ThreadPoolExecutor

import mt_metadata
import numpy as np
import pandas as pd
import scipy.signal as ssig
import xarray as xr

from loguru import logger
from aurora.time_series.windowing_scheme import window_scheme_from_decimation
from mt_metadata.processing import TimeSeriesDecimation
from mt_metadata.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)
from mt_metadata.processing.fourier_coefficients import Decimation as FCDecimation

from mth5.groups import RunGroup
from typing import Union


# The most threads prototype_decimate decimates channels in at once; 1 decimates
# them in turn.
DECIMATE_MAX_WORKERS = 4
# The most memory, in bytes, the decimating threads may hold in filter
# intermediates at once, a thread counted as three float64 copies of its channel.
# A long record is decimated in fewer threads, and in turn once three copies of
# a channel exceed it. 8 GiB allows 4 threads on a 6 h record at 1000 Hz and 2 on
# a 44.9 h record, whose decimation then stays under the level-0 regression peak.
DECIMATE_MAX_TRANSIENT_BYTES = 8 * 2**30


def decimate_num_workers(num_channels: int, num_samples: int) -> int:
    """
    The number of threads prototype_decimate decimates the channels of a run in.

    No more than DECIMATE_MAX_WORKERS, the channels or the CPUs, and no more than
    DECIMATE_MAX_TRANSIENT_BYTES allows when each thread holds three float64
    copies of its channel; at least 1.

    Parameters
    ----------
    num_channels : int
        The number of channels to decimate.
    num_samples : int
        The number of samples of each channel.

    Returns
    -------
    int
        The number of threads; 1 decimates the channels in turn.
    """
    per_thread = 3 * np.dtype(np.float64).itemsize * max(num_samples, 1)
    by_memory = DECIMATE_MAX_TRANSIENT_BYTES // per_thread
    return max(
        1, min(DECIMATE_MAX_WORKERS, num_channels, os.cpu_count() or 1, by_memory)
    )


def truncate_to_clock_zero(
    decimation_obj: Union[AuroraDecimationLevel, FCDecimation],
    run_xrds: RunGroup,
):
    """
    Compute the time interval between the first data sample and the clock zero
    Identify the first sample in the xarray time series that corresponds to a
    window start sample.

    Parameters
    ----------
    decimation_obj: Union[AuroraDecimationLevel, FCDecimation]
        Information about how the decimation level is to be processed
    run_xrds : xarray.core.dataset.Dataset
        normally extracted from mth5.RunTS


    Returns
    -------
    run_xrds : xarray.core.dataset.Dataset
        same as the input time series, but possibly slightly shortened
    """
    if decimation_obj.stft.window.clock_zero_type == "ignore":
        pass
    else:
        clock_zero = pd.Timestamp(decimation_obj.stft.window.clock_zero)
        clock_zero = clock_zero.to_datetime64()
        delta_t = clock_zero - run_xrds.time[0]
        assert delta_t.dtype == "<m8[ns]"  # expected in nanoseconds
        delta_t_seconds = int(delta_t) / 1e9
        if delta_t_seconds == 0:
            pass  # time series start is already clock zero
        else:
            windowing_scheme = window_scheme_from_decimation(decimation_obj)
            number_of_steps = delta_t_seconds / windowing_scheme.duration_advance
            n_partial_steps = number_of_steps - np.floor(number_of_steps)
            n_clip = n_partial_steps * windowing_scheme.num_samples_advance
            n_clip = int(np.round(n_clip))
            t_clip = run_xrds.time[n_clip]
            cond1 = run_xrds.time >= t_clip
            msg = (
                f"dropping {n_clip} samples to agree with "
                f"{decimation_obj.stft.window.clock_zero_type} clock zero {clock_zero}"
            )
            logger.info(msg)
            run_xrds = run_xrds.where(cond1, drop=True)
    return run_xrds


def prototype_decimate(
    ts_decimation: TimeSeriesDecimation,
    run_xrds: xr.Dataset,
) -> xr.Dataset:
    """
    Basically a wrapper for scipy.signal.decimate.  Takes input timeseries (as xarray
     Dataset) and a TimeSeriesDecimation object and returns a decimated version of the
     input time series.

    TODO: Consider moving this function into time_series/decimate.py
    TODO: Consider Replacing the downsampled_time_axis with rolling mean, or somthing that takes the average value of the time, not the window start
    TODO: Compare outputs with scipy resample_poly, which also has an FIR AAF and appears faster
    TODO: Add handling for case that could occur when sliced time axis has a different length than the decimated data -- see mth5 issue #217 https://github.com/kujaku11/mth5/issues/217

    The channels are decimated in up to DECIMATE_MAX_WORKERS threads at once (scipy's
    filters release the GIL), each channel by the same scipy.signal.decimate call as
    one after the other. A thread holds its channel's filter intermediates
    (sosfiltfilt keeps about three copies of the channel) until the channel is done,
    so the threads are also bounded by DECIMATE_MAX_TRANSIENT_BYTES
    (decimate_num_workers): a long record is decimated in fewer threads, or in turn.

    Parameters
    ----------
    ts_decimation : AuroraDecimationLevel
    run_xrds: xr.Dataset
        Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
        multiple times

    Returns
    -------
    xr_ds: xr.Dataset
        Decimated version of the input run_xrds
    """
    # downsample the time axis
    slicer = slice(None, None, int(ts_decimation.factor))  # decimation.factor
    downsampled_time_axis = run_xrds.time.data[slicer]

    # decimate the time series
    num_observations = len(downsampled_time_axis)
    channel_labels = list(run_xrds.data_vars.keys())  # run_ts.channels
    num_channels = len(channel_labels)
    new_data = np.full((num_observations, num_channels), np.nan)
    channels = [run_xrds[ch_label].data for ch_label in channel_labels]

    def decimate_channel(i_ch: int) -> None:
        # TODO: add check here for ts_decimation.anti_alias_filter
        new_data[:, i_ch] = ssig.decimate(channels[i_ch], int(ts_decimation.factor))

    num_workers = decimate_num_workers(num_channels, run_xrds.sizes["time"])
    if num_workers > 1:
        with ThreadPoolExecutor(max_workers=num_workers) as pool:
            list(pool.map(decimate_channel, range(num_channels)))
    else:
        for i_ch in range(num_channels):
            decimate_channel(i_ch)

    xr_da = xr.DataArray(
        new_data,
        dims=["time", "channel"],
        coords={"time": downsampled_time_axis, "channel": channel_labels},
    )
    attr_dict = run_xrds.attrs
    attr_dict["sample_rate"] = ts_decimation.sample_rate
    xr_da.attrs = attr_dict
    xr_ds = xr_da.to_dataset("channel")
    return xr_ds


# Here are some other ways to downsample/decimate that were experimented with.
# TODO: Jupyter notebook tutorial showing how these differ (cf standard method).
# def prototype_decimate_2(config, run_xrds):
#     """
#     Uses the built-in xarray coarsen method.   Not clear what AAF effects are.
#     Method is fast.  Might be non-linear.  Seems to give similar performance to
#     prototype_decimate for synthetic data.
#
#     N.B. config.decimation.factor must be integer valued
#
#     Parameters
#     ----------
#     config : mt_metadata.processing.aurora.Decimation
#     run_xrds: xr.Dataset
#         Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
#         multiple times
#
#     Returns
#     -------
#     xr_ds: xr.Dataset
#         Decimated version of the input run_xrds
#     """
#     new_xr_ds = run_xrds.coarsen(time=int(config.decimation.factor), boundary="trim").mean()
#     attr_dict = run_xrds.attrs
#     attr_dict["sample_rate"] = config.sample_rate
#     new_xr_ds.attrs = attr_dict
#     return new_xr_ds
#
#
# def prototype_decimate_3(config, run_xrds):
#     """
#     Uses scipy's resample method.   Not clear what AAF effects are.
#     Method is fast.
#
#     Parameters
#     ----------
#     config : mt_metadata.processing.aurora.Decimation
#     run_xrds: xr.Dataset
#         Originally from mth5.timeseries.run_ts.RunTS.dataset, but possibly decimated
#         multiple times
#
#     Returns
#     -------
#     xr_ds: xr.Dataset
#         Decimated version of the input run_xrds
#     """
#     dt = run_xrds.time.diff(dim="time").median().values
#     dt_new = config.decimation.factor * dt
#     dt_new = dt_new.__str__().replace("nanoseconds", "ns")
#     new_xr_ds = run_xrds.resample(time=dt_new).mean(dim="time")
#     attr_dict = run_xrds.attrs
#     attr_dict["sample_rate"] = config.sample_rate
#     new_xr_ds.attrs = attr_dict
#     return new_xr_ds
