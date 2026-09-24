"""
This module contains aurora methods associated with spectrograms or "STFTs".
In future these tools should be moved to MTH5 and made methods of the Spectrogram class.
For now, we can use this module as a place to aggregate functions to migrate.
"""

from typing import Literal, Optional

import mth5.timeseries.spectre as spectre
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mt_metadata.processing.aurora import DecimationLevel as AuroraDecimationLevel
from mth5.groups import RunGroup
from mth5.processing.spectre.prewhitening import apply_prewhitening, apply_recoloring

from aurora.config.metadata.processing import Processing as AuroraProcessing
from aurora.pipelines.time_series_helpers import truncate_to_clock_zero
from aurora.pipelines.transfer_function_kernel import (
    station_obj_from_row,
    TransferFunctionKernel,
)
from aurora.sandbox.triage_metadata import triage_run_id
from aurora.time_series.windowed_time_series import WindowedTimeSeries
from aurora.time_series.windowing_scheme import window_scheme_from_decimation
from aurora.time_series.xarray_helpers import nan_to_mean, time_axis_match


def make_stft_objects(
    processing_config: AuroraProcessing,
    i_dec_level: int,
    run_obj: RunGroup,
    run_xrds: xr.Dataset,
    units: Literal["MT", "SI"] = "MT",
    calibration_filters: Optional[dict] = None,
) -> xr.Dataset:
    """
    Applies STFT to all channel time series in the input run.

    This method could be modified in a multiple station code so that it doesn't care
    if the station is "local" or "remote" but rather uses scale factors keyed by
    station_id (WIP - issue #329)

    Parameters
    ----------
    processing_config: mt_metadata.processing.aurora.Processing
        Metadata about the processing to be applied
    i_dec_level: int
        The decimation level to process
    run_obj: mth5.groups.master_station_run_channel.RunGroup
        The run to transform to stft
    run_xrds: xarray.core.dataset.Dataset
        The data time series from the run to transform
    units: Literal["MT", "SI"] = "MT",
        expects "MT".  May change so that this is the only accepted set of units
    calibration_filters: Optional[dict]
        The run's calibration filters by channel, passed to calibrate_stft_obj.

    Returns
    -------
    stft_obj: xarray.core.dataset.Dataset
        Time series of calibrated Fourier coefficients per each channel in the run

    Development Notes:
    Here are the parameters that are defined via the mt_metadata fourier coefficients structures:

    "bands",
    "decimation.anti_alias_filter": "default",
    "decimation.factor": 4.0,
    "decimation.level": 2,
    "decimation.method": "default",
    "decimation.sample_rate": 0.0625,
    "stft.per_window_detrend_type": "linear",
    "stft.prewhitening_type": "first difference",
    "stft.window.clock_zero_type": "ignore",
    "stft.window.num_samples": 128,
    "stft.window.overlap": 32,
    "stft.window.type": "boxcar"

    Creating the decimations config requires a decision about decimation factors and the number of levels.
    We have been getting this from the EMTF band setup file by default.  It is desirable to continue supporting this,
    however, note that the EMTF band setup is really about a time series operation, and not about making STFTs.

    For the record, here is the legacy decimation config from EMTF, a.k.a. decset.cfg:
    ```
    4     0      # of decimation level, & decimation offset
    128  32.   1   0   0   7   4   32   1
    1.0
    128  32.   4   0   0   7   4   32   4
    .2154  .1911   .1307   .0705
    128  32.   4   0   0   7   4   32   4
    .2154  .1911   .1307   .0705
    128  32.   4   0   0   7   4   32   4
    .2154  .1911   .1307   .0705
    ```

    This essentially corresponds to a "Decimations Group" which is a list of decimations.
    Related to the generation of FCs is the ARMA prewhitening (Issue #60) which was controlled in
    EMTF with pwset.cfg
    4    5             # of decimation levels, # of channels
    3 3 3 3 3
    3 3 3 3 3
    3 3 3 3 3
    3 3 3 3 3

    """
    stft_config = processing_config.get_decimation_level(i_dec_level)
    spectrogram = run_ts_to_stft(stft_config, run_xrds)
    run_id = run_obj.metadata.id
    if run_obj.station_metadata.id == processing_config.stations.local.id:
        scale_factors = processing_config.stations.local.run_dict[
            run_id
        ].channel_scale_factors
    elif run_obj.station_metadata.id == processing_config.stations.remote[0].id:
        scale_factors = (
            processing_config.stations.remote[0].run_dict[run_id].channel_scale_factors
        )

    stft_obj = calibrate_stft_obj(
        spectrogram.dataset,
        run_obj,
        units=units,
        channel_scale_factors=scale_factors,
        calibration_filters=calibration_filters,
    )

    return stft_obj


def shape_check_spectrograms(local_stfts: list, remote_stfts: list):
    """
    Performs a shape check on the local and remote STFTs.
    Seems associated with getting one fewer sample than expected from the edge of a run,
    WIP: Aurora Issue #289.

    If a shape mismatch is detected, compute the greatest lower bound (glb) and least upper bound (lub)
    of the time axis of the local and remote STFTs, and then trim both STFTs to the intersection of these bounds.

    Parameters
    ----------
    local_stfts: list
        A list of local STFT objects, each is an xr.Dataset
    remote_stfts: list
        A list of remote STFT objects, each is an xr.Dataset

    returns: Tuple
        (local_stfts, remote_stfts)
        The original input arguments, shape-matched

    """
    n_chunks = len(local_stfts)
    for i_chunk in range(n_chunks):
        ok = local_stfts[i_chunk].time.shape == remote_stfts[i_chunk].time.shape
        if not ok:
            logger.warning("Mismatch in FC array lengths detected -- Issue #289")
            glb = max(
                local_stfts[i_chunk].time.min(),
                remote_stfts[i_chunk].time.min(),
            )
            lub = min(
                local_stfts[i_chunk].time.max(),
                remote_stfts[i_chunk].time.max(),
            )

            cond1 = local_stfts[i_chunk].time >= glb
            cond2 = local_stfts[i_chunk].time <= lub
            local_stfts[i_chunk] = local_stfts[i_chunk].where(cond1 & cond2, drop=True)

            cond1 = remote_stfts[i_chunk].time >= glb
            cond2 = remote_stfts[i_chunk].time <= lub
            remote_stfts[i_chunk] = remote_stfts[i_chunk].where(
                cond1 & cond2, drop=True
            )
            assert local_stfts[i_chunk].time.shape == remote_stfts[i_chunk].time.shape

    return local_stfts, remote_stfts


def time_axis_check_spectrograms(local_stfts: list, remote_stfts: list):
    """
    Checks that the time axes of local and remote STFTs match.
    If they do not match, raises an error.

    Parameters
    ----------
    local_stfts: list
        A list of local STFT objects, each is an xr.Dataset
    remote_stfts: list
        A list of remote STFT objects, each is an xr.Dataset

    Raises
    ------
    ValueError
        If the time axes of local and remote STFTs do not match.
    """
    for i_chunk in range(len(local_stfts)):
        ok = time_axis_match(local_stfts[i_chunk], remote_stfts[i_chunk])
        if not ok:
            raise ValueError(
                "Time axes of local and remote STFTs do not match. "
                "Please check the data."
            )


def concatenate_stfts(stfts: list) -> xr.Dataset:
    """
    Concatenates the STFTs of several runs along the time axis, emptying the list.

    A single STFT is returned as it is. With several, each data variable is
    concatenated in turn and dropped from the runs' STFTs once it has been copied.
    STFTs with differing data variables, or with coordinates other than time and
    frequency, are concatenated with xr.concat in one step.

    Parameters
    ----------
    stfts: list
        The STFTs (xr.Dataset) of the runs, in time order.

    Returns
    -------
    merged: xr.Dataset
        The STFTs concatenated along time.
    """
    if len(stfts) == 1:
        return stfts.pop()
    names = list(stfts[0].data_vars)
    simple = all(
        list(ds.data_vars) == names and set(ds.coords) <= {"time", "frequency"}
        for ds in stfts
    )
    if not simple:
        merged = xr.concat(stfts, "time")
        stfts.clear()
        return merged
    attrs = stfts[0].attrs
    coord_names = list(stfts[0].coords)
    data_vars = {}
    coords = {}
    for name in names:
        merged = xr.concat([ds[name] for ds in stfts], "time")
        data_vars[name] = merged.variable
        coords = {c: merged[c].variable for c in coord_names}
        for i, ds in enumerate(stfts):
            stfts[i] = ds.drop_vars(name)
    stfts.clear()
    return xr.Dataset(data_vars, coords=coords, attrs=attrs)


def merge_stfts(stfts: dict, tfk: TransferFunctionKernel):
    """

    Applies concatenation along the time axis to multiple arrays of STFTs from different runs.
    At the TF estimation level we treat all the FCs in one array.
    This builds the array for both the local and the remote STFTs.
    The lists of stfts are emptied (see concatenate_stfts).

    Parameters
    ----------
    stfts: dict
        The dict is keyed by "local" and "remote".
        Each value is a list of STFTs (one list for local and one for remote)
    tfk: TransferFunctionKernel
        Just here to let us know if there is a remote reference to merge or not.

    Returns
    -------
    local_merged_stft_obj, remote_merged_stft_obj: Tuple
        Both are xr.Datasets
    """
    local_stfts = stfts["local"]
    remote_stfts = stfts["remote"]
    if tfk.config.stations.remote:
        local_stfts, remote_stfts = shape_check_spectrograms(local_stfts, remote_stfts)
        time_axis_check_spectrograms(local_stfts, remote_stfts)
        remote_merged_stft_obj = concatenate_stfts(remote_stfts)
    else:
        remote_merged_stft_obj = None

    local_merged_stft_obj = concatenate_stfts(local_stfts)

    return local_merged_stft_obj, remote_merged_stft_obj


def append_chunk_to_stfts(stfts: dict, chunk: xr.Dataset, remote: bool) -> dict:
    """
    Aggregate one STFT into a larger dictionary that tracks all the STFTs

    Parameters
    ----------
    stfts: dict
        has keys "local" and "remote".
    chunk: xr.Dataset
        The data to append to the dictionary
    remote: bool
        If True, append the chunk to stfts["remote"], else append to stfts["local"]

    Returns
    -------
    stfts: dict
        Same as input but now has new chunk appended to it.
    """
    if remote:
        stfts["remote"].append(chunk)
    else:
        stfts["local"].append(chunk)
    return stfts


# def load_spectrogram_from_station_object(station_obj, fc_group_id, fc_decimation_id):
#     """
#     Placeholder.  This could also be a method in mth5
#     Returns
#     -------
#
#     """
#     return station_obj.fourier_coefficients_group.get_fc_group(fc_group_id).get_decimation_level(fc_decimation_id)


def load_stft_obj_from_mth5(
    i_dec_level: int,
    row: pd.Series,
    run_obj: RunGroup,
    channels: Optional[list] = None,
) -> xr.Dataset:
    """
    Load stft_obj from mth5 (instead of compute)

    Note #1: See note #1 in mth5.timeseries.spectre.spectrogram.py in extract_band function.

    Parameters
    ----------
    i_dec_level: int
        The decimation level where the data are stored within the Fourier Coefficient group
    row: pandas.core.series.Series
        A row of the TFK.dataset_df
    run_obj: mth5.groups.run.RunGroup
        The original time-domain run associated with the data to load

    Returns
    -------
    stft_chunk: xr.Dataset
        An STFT from mth5.
    """
    station_obj = station_obj_from_row(row)
    fc_group = station_obj.fourier_coefficients_group.get_fc_group(run_obj.metadata.id)
    fc_decimation_level = fc_group.get_decimation_level(f"{i_dec_level}")
    stft_obj = fc_decimation_level.to_xarray(channels=channels)

    cond1 = stft_obj.time >= row.start.tz_localize(None)
    cond2 = stft_obj.time <= row.end.tz_localize(None)
    try:
        stft_chunk = stft_obj.where(cond1 & cond2, drop=True)
    except TypeError:  # see Note #1
        tmp = stft_obj.to_array()
        tmp = tmp.where(cond1 & cond2, drop=True)
        stft_chunk = tmp.to_dataset("variable")
    return stft_chunk


def save_fourier_coefficients(
    dec_level_config: AuroraDecimationLevel, row: pd.Series, run_obj, stft_obj
) -> None:
    """
    Optionally saves the stft object into the MTH5.
    Note that the dec_level_config must have its save_fcs attr set to True to actually save the data.
    WIP

    Note #1: Logic for building FC layers:
    If the processing config decimation_level.save_fcs_type = "h5" and fc_levels_already_exist is False, then open
    in append mode, else open in read mode.  We should support a flag: force_rebuild_fcs, normally False.  This flag
    is only needed when save_fcs_type=="h5".  If True, then we open in append mode, regarless of fc_levels_already_exist
    The task of setting mode="a", mode="r" can be handled by tfk (maybe in tfk.validate())

    Parameters
    ----------
    dec_level_config: mt_metadata.processing.aurora.decimation_level.DecimationLevel
        The information about decimation level associated with row, run, stft_obj
    row: pd.Series
         A row of the TFK.dataset_df
    run_obj: mth5.groups.run.RunGroup
        The run object associated with the STFTs.
    stft_obj: xr.Dataset
        The data to pack intp the mth5.

    """

    if not dec_level_config.save_fcs:
        msg = "Skip saving FCs. dec_level_config.save_fc = "
        msg = f"{msg} {dec_level_config.save_fcs}"
        logger.info(f"{msg}")
        return
    i_dec_level = dec_level_config.decimation.level
    if dec_level_config.save_fcs_type == "csv":
        msg = "Unless debugging or testing, saving FCs to csv unexpected"
        logger.warning(msg)
        csv_name = f"{row.station}_dec_level_{i_dec_level}.csv"
        stft_df = stft_obj.to_dataframe()
        stft_df.to_csv(csv_name)
    elif dec_level_config.save_fcs_type == "h5":
        logger.info(("Saving FC level"))
        station_obj = station_obj_from_row(row)

        if not row.mth5_obj.h5_is_write():
            msg = "See Note #1: Logic for building FC layers"
            raise NotImplementedError(msg)

        # Get FC group (create if needed)
        if run_obj.metadata.id in station_obj.fourier_coefficients_group.groups_list:
            fc_group = station_obj.fourier_coefficients_group.get_fc_group(
                run_obj.metadata.id
            )
        else:
            fc_group = station_obj.fourier_coefficients_group.add_fc_group(
                run_obj.metadata.id
            )

        decimation_level_metadata = dec_level_config.to_fc_decimation()

        # Get FC Decimation Level (create if needed)
        dec_level_name = f"{i_dec_level}"
        if dec_level_name in fc_group.groups_list:
            fc_decimation_level = fc_group.get_decimation_level(dec_level_name)
            fc_decimation_level.metadata = decimation_level_metadata
        else:
            fc_decimation_level = fc_group.add_decimation_level(
                dec_level_name,
                decimation_level_metadata=decimation_level_metadata,
            )
        fc_decimation_level.from_xarray(
            stft_obj, decimation_level_metadata.decimation.sample_rate
        )
        fc_decimation_level.update_metadata()
        fc_group.update_metadata()
    else:
        msg = f"dec_level_config.save_fcs_type {dec_level_config.save_fcs_type} not recognized"
        msg = f"{msg} \n Skipping save fcs"
        logger.warning(msg)
    return


def get_spectrograms(
    tfk: TransferFunctionKernel, i_dec_level: int, units: str = "MT"
) -> dict:
    """
    Given a decimation level id, loads a dictianary of all spectragrams from information in tfk.
    TODO: Make this a method of TFK
    TODO: Modify this to be able to yield Spectrogram objects.

    Parameters
    ----------
    tfk: TransferFunctionKernel

    i_dec_level: integer
        The decimation level of the spectrograms.

    units: str
        "MT" or "SI", likely to be deprecated

    Returns
    -------
    stfts: dict
        The short time fourier transforms for the decimation level as a dictionary.
        Keys are "local" and "remote".  Values are lists, one (element) xr.Dataset per run
    """

    stfts = {}
    stfts["local"] = []
    stfts["remote"] = []
    # calibration filters of each run, kept on the kernel for all decimation levels
    run_calibration_filters = getattr(tfk, "calibration_filters", None)

    # Check first if TS processing or accessing FC Levels
    for i, row in tfk.dataset_df.iterrows():
        # TODO: Consider updating this to iterate over row-pairs corresponding to simultaneous data.
        #  Grouping by (sorted) starttime should accomplish this.

        if not tfk.is_valid_dataset(row, i_dec_level):
            continue

        run_obj = row.mth5_obj.from_reference(row.run_hdf5_reference)
        if row.fc:
            stft_obj = load_stft_obj_from_mth5(i_dec_level, row, run_obj)
            # TODO: Cast stft_obj to a Spectrogram here
            stfts = append_chunk_to_stfts(stfts, stft_obj, row.remote)
            continue

        run_xrds = row["run_dataarray"].to_dataset("channel")

        # Musgraves workaround for old MT data
        triage_run_id(row.run, run_obj)

        calibration_filters = None
        if run_calibration_filters is not None:
            key = (str(row.mth5_path), row.survey, row.station, row.run)
            calibration_filters = run_calibration_filters.setdefault(key, {})
        stft_obj = make_stft_objects(
            tfk.config,
            i_dec_level,
            run_obj,
            run_xrds,
            units,
            calibration_filters=calibration_filters,
        )
        # TODO: Cast stft_obj to a Spectrogram here or in make_stft_objects

        # Pack FCs into h5
        dec_level_config = tfk.config.decimations[i_dec_level]
        save_fourier_coefficients(dec_level_config, row, run_obj, stft_obj)
        # TODO: cast stft_obj to a Spectrogram here

        stfts = append_chunk_to_stfts(stfts, stft_obj, row.remote)

    return stfts


def run_ts_to_stft(
    decimation_obj: AuroraDecimationLevel, run_xrds_orig: xr.Dataset
) -> spectre.Spectrogram:
    """
    Converts a runts object into a time series of Fourier coefficients.
    Similar to run_ts_to_stft_scipy, but in this implementation operations on individual
    windows are possible (for example pre-whitening per time window via ARMA filtering).

    TODO: Make the output of this function a Spectrogram object

    Parameters
    ----------
    decimation_obj : AuroraDecimationLevel
        Information about how the decimation level is to be processed
    run_xrds_orig: xarray.core.dataset.Dataset
        normally extracted from mth5.RunTS

    Returns
    -------
    stft_obj: xarray.core.dataset.Dataset
        Note that the STFT object may have inf/nan in the DC harmonic, introduced by
        recoloring. This really doesn't matter since we don't use the DC harmonic for
        anything.
    """
    # need to remove any nans before windowing, or else if there is a single
    # nan then the whole channel becomes nan.
    run_xrds = nan_to_mean(run_xrds_orig)
    run_xrds = apply_prewhitening(decimation_obj.stft.prewhitening_type, run_xrds)
    run_xrds = truncate_to_clock_zero(decimation_obj, run_xrds)
    windowing_scheme = window_scheme_from_decimation(decimation_obj)
    windowed_obj = windowing_scheme.apply_sliding_window(
        run_xrds, dt=1.0 / decimation_obj.decimation.sample_rate
    )
    del run_xrds  # the windows are copies
    if not windowed_obj.data_vars or not all(
        v.size for v in windowed_obj.data_vars.values()
    ):
        raise ValueError

    windowed_obj = WindowedTimeSeries.detrend(data=windowed_obj, detrend_type="linear")
    tapered_obj = WindowedTimeSeries.apply_taper(
        data=windowed_obj, taper=windowing_scheme.taper
    )
    del windowed_obj
    stft_obj = WindowedTimeSeries.apply_fft(
        data=tapered_obj,
        sample_rate=windowing_scheme.sample_rate,
        spectral_density_correction=windowing_scheme.linear_spectral_density_calibration_factor,
        detrend_type=decimation_obj.stft.per_window_detrend_type,
    )
    del tapered_obj

    if decimation_obj.stft.recoloring:
        stft_obj = apply_recoloring(decimation_obj.stft.prewhitening_type, stft_obj)

    spectrogram = spectre.Spectrogram(dataset=stft_obj)
    return spectrogram


def get_calibration_filters(run_obj: RunGroup, channel_id: str) -> tuple:
    """
    Returns the channel response of a channel and the filters calibration removes.

    Parameters
    ----------
    run_obj : mth5.groups.master_station_run_channel.RunGroup
        The run the channel belongs to.
    channel_id : str
        The channel.

    Returns
    -------
    channel_response : mt_metadata.timeseries.filters.ChannelResponse
        The channel's response (that of hx for an hy without filters).
    filters_to_remove : list
        The applied filters of the response, without decimation and delay filters.
    """
    channel = run_obj.get_channel(channel_id)
    channel_response = channel.channel_response
    if not channel_response.filters_list:
        msg = f"Channel {channel_id} with empty filters list detected"
        logger.warning(msg)
        if channel_id == "hy":
            msg = "Channel hy has no filters, try using filters from hx"
            logger.warning(msg)
            channel_response = run_obj.get_channel("hx").channel_response

    indices_to_flip = channel_response.get_indices_of_filters_to_remove(
        include_decimation=False, include_delay=False
    )
    indices_to_flip = [
        i for i in indices_to_flip if channel.metadata.filters[i].applied
    ]
    filters_to_remove = [channel_response.filters_list[i] for i in indices_to_flip]
    if not filters_to_remove:
        logger.warning("No filters to remove")
    return channel_response, filters_to_remove


def calibrate_stft_obj(
    stft_obj: xr.Dataset,
    run_obj: RunGroup,
    units: Literal["MT", "SI"] = "MT",
    channel_scale_factors: Optional[dict] = None,
    calibration_filters: Optional[dict] = None,
) -> xr.Dataset:
    """
    Calibrates frequency domain data into MT units.

    Development Notes:
     The calibration often raises a runtime warning due to DC term in calibration response = 0.
     TODO: It would be nice to suppress this, maybe by only calibrating the non-dc terms and directly assigning np.nan to the dc component when DC-response is zero.

    Parameters
    ----------
    stft_obj : xarray.core.dataset.Dataset
        Time series of Fourier coefficients to be calibrated
    run_obj : mth5.groups.master_station_run_channel.RunGroup
        Provides information about filters for calibration
    units : string
        usually "MT", contemplating supporting "SI"
    channel_scale_factors : Optional[dict]
        Keyed by channel, supports a single scalar to apply to that channels data
        Useful for debugging.  Should not be used in production and should throw a
        warning if it is not None
    calibration_filters : Optional[dict]
        Keyed by channel, the (channel_response, filters_to_remove) of
        get_calibration_filters for this run; a missing channel is added. When None,
        they are read from run_obj for every channel.

    Returns
    -------
    stft_obj : xarray.core.dataset.Dataset
        Time series of calibrated Fourier coefficients
    """
    for channel_id in stft_obj.keys():

        if calibration_filters is None:
            channel_response, filters_to_remove = get_calibration_filters(
                run_obj, channel_id
            )
        else:
            if channel_id not in calibration_filters:
                calibration_filters[channel_id] = get_calibration_filters(
                    run_obj, channel_id
                )
            channel_response, filters_to_remove = calibration_filters[channel_id]

        calibration_response = channel_response.complex_response(
            stft_obj.frequency.data, filters_list=filters_to_remove
        )

        if channel_scale_factors:
            try:
                channel_scale_factor = channel_scale_factors[channel_id]
            except KeyError:
                channel_scale_factor = 1.0
            if channel_scale_factor != 1.0:
                calibration_response /= channel_scale_factor

        if units == "SI":
            logger.warning("Warning: SI Units are not robustly supported issue #36")

        # Handle case where DC term in calibration response = 0 (Sometimes raises a runtime warning)
        zero_dc_term = calibration_response[0] == 0
        if zero_dc_term:
            calibration_response[0] = 1.0
        stft_obj[channel_id].data /= calibration_response
        if zero_dc_term:
            stft_obj[channel_id].data[:, 0] = np.nan + 1j * np.nan

    return stft_obj
