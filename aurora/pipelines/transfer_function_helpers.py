"""
This module contains helper methods that are used during transfer function processing.

Development Notes:
 Note #1: repeatedly applying edf_weights seems to have no effect at all.
 tested 20240118 and found that test_compare in synthetic passed whether this was commented
 or not.  TODO confirm this is a one-and-done add doc about why this is so.

"""

from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from mt_metadata.processing.aurora.decimation_level import (
    DecimationLevel as AuroraDecimationLevel,
)

from aurora.time_series.frequency_band_helpers import get_band_for_tf_estimate
from aurora.time_series.xarray_helpers import (
    all_finite,
    handle_nan,
    same_coordinates,
)
from aurora.transfer_function.regression.base import RegressionEstimator
from aurora.transfer_function.regression.iter_control import IterControl
from aurora.transfer_function.regression.RME import RME
from aurora.transfer_function.regression.RME_RR import RME_RR
from aurora.transfer_function.weights.edf_weights import (
    effective_degrees_of_freedom_weights,
)


ESTIMATOR_LIBRARY = {"OLS": RegressionEstimator, "RME": RME, "RME_RR": RME_RR}
SUPPORTED_REGRESSION_ESTIMATOR = Literal["OLS", "RME", "RME_RR"]


def get_estimator_class(
    estimation_engine: SUPPORTED_REGRESSION_ESTIMATOR,
) -> RegressionEstimator:
    """

    Parameters
    ----------
    estimation_engine: Literal["OLS", "RME", "RME_RR"]
        One of the keys in the ESTIMATOR_LIBRARY, designates the method that will be
        used to estimate the transfer function

    Returns
    -------
    estimator_class: aurora.transfer_function.regression.base.RegressionEstimator
        The class that will do the TF estimation
    """
    try:
        estimator_class = ESTIMATOR_LIBRARY[estimation_engine]
    except KeyError:
        logger.error(f"processing_scheme {estimation_engine} not supported")
        logger.error(
            f"processing_scheme must be one of {list(ESTIMATOR_LIBRARY.keys())}"
        )
        raise Exception
    return estimator_class


def set_up_iter_control(config: AuroraDecimationLevel):
    """
    Initializes an IterControl object based on values in the processing config.

    Development Notes:
    TODO: Review: maybe better to just make this the __init__ method of the IterControl object, iter_control = IterControl(config)


    Parameters
    ----------
    config: AuroraDecimationLevel
        metadata about the decimation level processing.

    Returns
    -------
    iter_control: aurora.transfer_function.regression.iter_control.IterControl
        Object with parameters about iteration control in regression
    """
    if config.estimator.engine in ["RME", "RME_RR"]:
        iter_control = IterControl(
            max_number_of_iterations=config.regression.max_iterations,
            max_number_of_redescending_iterations=config.regression.max_redescending_iterations,
            r0=config.regression.r0,
            u0=config.regression.u0,
            tolerance=config.regression.tolerance,
        )
    elif config.estimator.engine in [
        "OLS",
    ]:
        iter_control = None

    return iter_control


def drop_nans(X: xr.Dataset, Y: xr.Dataset, RR: Union[xr.Dataset, None]) -> tuple:
    """
    Drops any observation where any variable in X, Y, or RR is NaN.

    When Y is given, X, Y and RR (if given) have data variables, Y and RR
    have the "observation" dimension of X with its length, X's coordinates
    and X's dataset attributes (same_coordinates), and every value is
    finite, isel would return copies identical to the inputs, so X, Y and
    RR are returned as they are. Any other input takes isel as before.
    Data without an "observation" dimension still raise ValueError. Y or
    RR on other observations, Y None or a dataset without data variables
    still raise. Y or RR with other dataset attributes are copied with
    their own.
    """
    import numpy as np

    datasets = [ds for ds in (X, Y, RR) if ds is not None]
    stacked = (
        Y is not None
        and all(ds.data_vars for ds in datasets)
        and same_coordinates(X, Y, RR, dim="observation")
    )
    if stacked and all_finite(X, Y, RR):
        return X, Y, RR

    def get_obs_mask(ds):
        """
        Generate a boolean mask indicating which 'observation' entries are finite across all data variables in an xarray Dataset.

        This function iterates over all data variables in the provided xarray Dataset `ds`, checks for finite values (i.e., not NaN or infinite)
        along all axes except the 'observation' axis, and combines the results to produce a single boolean mask. The resulting mask is True
        for each 'observation' index where all data variables are finite, and False otherwise.


        Parameters
        ds : xarray.Dataset
            The input dataset containing data variables with an 'observation' dimension.

        Returns
        numpy.ndarray
            A boolean array with shape matching the 'observation' dimension, where True indicates all data variables are finite
        """
        mask = None
        for v in ds.data_vars.values():
            # Reduce all axes except 'observation'
            axes = tuple(i for i, d in enumerate(v.dims) if d != "observation")
            this_mask = np.isfinite(v)
            if axes:
                this_mask = this_mask.all(axis=axes)
            mask = this_mask if mask is None else mask & this_mask
        return mask

    mask = get_obs_mask(X)
    mask = mask & get_obs_mask(Y)
    if RR is not None:
        mask = mask & get_obs_mask(RR)

    X = X.isel(observation=mask)
    Y = Y.isel(observation=mask)
    if RR is not None:
        RR = RR.isel(observation=mask)
    return X, Y, RR


def stack_frequency_time(ds: xr.Dataset) -> xr.Dataset:
    """
    Ravels the frequency and time dimensions of a dataset into one "observation" dimension.

    The observations are frequency-major (the time index varies fastest), the order
    of ``ds.stack(observation=("frequency", "time"))``. The frequency and time of
    each observation are kept as coordinates on the observation dimension, which
    has no index.

    Parameters
    ----------
    ds: xarray.core.dataset.Dataset
        Every data variable has the dimensions frequency and time.

    Returns
    -------
    stacked: xarray.core.dataset.Dataset
        Same data variables as the input with the frequency and time dimensions
        replaced by observation.
    """
    n_frequency = ds.sizes["frequency"]
    n_time = ds.sizes["time"]
    data_vars = {}
    for name, var in ds.data_vars.items():
        var = var.variable.transpose(..., "frequency", "time")
        data = np.reshape(var.data, var.shape[:-2] + (n_frequency * n_time,))
        data_vars[name] = xr.Variable(
            var.dims[:-2] + ("observation",), data, attrs=var.attrs
        )
    coords = {
        "frequency": ("observation", np.repeat(ds.frequency.values, n_time)),
        "time": ("observation", np.tile(ds.time.values, n_frequency)),
    }
    return xr.Dataset(data_vars, coords=coords, attrs=ds.attrs)


def stack_fcs(X, Y, RR):
    """
    Reshape 2D arrays of frequency and time to 1D.

    Notes: When the data for a frequency band are extracted from the Spectrogram, each
    channel is a 2D array, one axis is time (the time of the window that was FFT-ed) and the
    other axis is frequency. However if we make no distinction between the harmonics
    (bins) within a band in regression, then all the FCs for each channel can be
    put into a 1D array.  This method performs that reshaping (ravelling) operation.
    It is not important how we unravel the FCs but it is important that the same indexing
    scheme is used for X, Y and RR.

    The stacked observation dimension has no index (see stack_frequency_time).

    TODO: Consider this take a list and return a list rather than X,Y,RR

    TODO: Consider decorate this with @dataset_or_dataarray

    Parameters
    ----------
    X: xarray.core.dataset.Dataset
    Y: xarray.core.dataset.Dataset
    RR: xarray.core.dataset.Dataset or None

    Returns
    -------
    X, Y, RR: Same as input but with stacked time and frequency dimensions
    """
    X = stack_frequency_time(X)
    Y = stack_frequency_time(Y)
    if RR is not None:
        RR = stack_frequency_time(RR)
    return X, Y, RR


def stack_and_edf_weights(
    X: xr.Dataset,
    Y: xr.Dataset,
    RR: Optional[xr.Dataset],
    shared: Optional[dict] = None,
) -> tuple:
    """
    Stacks the FCs of a band, drops the observations that are not finite and
    computes the effective degrees of freedom weights of the rest.

    The output channels of a band that have no channel weights are regressed
    on the same X and RR. With `shared`, a dict kept for the band and passed
    for each of those channels, X and RR are stacked and checked for values
    that are not finite once per band, and when Y is finite too the weights
    are computed once per band: each channel gets shallow copies of the
    stacked X and RR, which apply_weights scales in place, and a copy of the
    weights, in which apply_weights sets zero weights to NaN; the shared
    weights stay as they are. When X, RR or Y holds a value that is not
    finite, the observations are dropped and the weights computed for the
    channel, as without `shared`.

    Parameters
    ----------
    X: xarray.core.dataset.Dataset
    Y: xarray.core.dataset.Dataset
    RR: xarray.core.dataset.Dataset or None
        The FCs of the band, on frequency and time.
    shared: dict or None
        An empty dict for the first output channel of a band, then the same
        dict for the band's other output channels with the same X and RR.

    Returns
    -------
    X, Y, RR: tuple
        Stacked, without the observations that are not finite.
    W: numpy array
        The effective degrees of freedom weights of the observations.
    """
    if shared is None:
        X, Y, RR = stack_fcs(X, Y, RR)
        # Should only be needed if weights were applied
        X, Y, RR = drop_nans(X, Y, RR)
        W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
        return X, Y, RR, W

    if "X" not in shared:
        shared["X"] = stack_frequency_time(X)
        shared["RR"] = None if RR is None else stack_frequency_time(RR)
        shared["finite"] = all_finite(shared["X"], shared["RR"])
    X, RR = shared["X"], shared["RR"]
    Y = stack_frequency_time(Y)
    if not (shared["finite"] and all_finite(Y)):
        X, Y, RR = drop_nans(X, Y, RR)
        W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
        return X, Y, RR, W

    if "W" not in shared:
        shared["W"] = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
    X = X.copy(deep=False)
    if RR is not None:
        RR = RR.copy(deep=False)
    return X, Y, RR, shared["W"].copy()


def apply_weights(
    X: xr.Dataset,
    Y: xr.Dataset,
    RR: xr.Dataset,
    W: np.ndarray,
    segment: bool = False,
    dropna: bool = False,
) -> tuple:
    """
    Applies data weights (W) to each of X, Y, RR.
    If weight is zero, we set to nan and optionally dropna.

    Parameters
    ----------
    X: xarray.core.dataset.Dataset
    Y: xarray.core.dataset.Dataset
    RR: xarray.core.dataset.Dataset or None
    W: numpy array
        The Weights to apply to the data
    segment: bool
        If True the weights may need to be reshaped.
    dropna: bool
        Whether or not to drop zero-weighted data.  If true, we drop the nans.

    Returns
    -------
    X, Y, RR: tuple
        Same as input but with weights applied and (optionally) nan dropped.
    """
    W[W == 0] = np.nan
    if segment:
        W = np.atleast_2d(W).T
    X *= W
    Y *= W
    if RR is not None:
        RR *= W

    if dropna:
        X, Y, RR = drop_nans(X, Y, RR)

    return X, Y, RR


def _utc_naive(value) -> np.datetime64:
    """A time (ISO string, Timestamp, datetime64; naive is taken as UTC) as naive UTC datetime64[ns]."""
    ts = pd.Timestamp(value)
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return np.datetime64(ts.to_datetime64(), "ns")


def window_mask_for_band(
    dec_level_config: AuroraDecimationLevel,
    band,
    times: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Returns a boolean array over the STFT windows, True for the windows to keep in this band.

    dec_level_config.window_masks (optional; None or [] means no masking) is a list of
    [start, end, pmin_s, pmax_s]: a time interval (UTC; ISO strings or timestamps, naive
    is UTC) and a period range in seconds. A mask applies to a band whose center period
    lies in [pmin_s, pmax_s], and removes from that band every STFT window that overlaps
    [start, end): window k spans [times[k], times[k] + num_samples / sample_rate), times
    being the STFT's time axis (the first sample of each window). The windows are removed
    before the regression, so no weighting ever sees them. A band the masks would leave
    with fewer than max(1, stft.min_num_stft_windows) windows is left unmasked, with a
    warning.

    Parameters
    ----------
    dec_level_config: AuroraDecimationLevel
        Processing parameters for the active decimation level.
    band: mt_metadata.common.band.Band
        The frequency band being estimated.
    times: np.ndarray
        The time axis of the band's STFT (datetime64).

    Returns
    -------
    keep: np.ndarray of bool, or None
        None when no mask removes a window from this band.
    """
    masks = getattr(dec_level_config, "window_masks", None)
    if not masks:
        return None
    period = band.center_period
    t = np.asarray(times).astype("datetime64[ns]")
    window_ns = int(
        round(
            1e9
            * dec_level_config.stft.window.num_samples
            / dec_level_config.decimation.sample_rate
        )
    )
    window = np.timedelta64(window_ns, "ns")
    drop = np.zeros(t.size, dtype=bool)
    for start, end, pmin_s, pmax_s in masks:
        if not min(pmin_s, pmax_s) <= period <= max(pmin_s, pmax_s):
            continue
        drop |= (t < _utc_naive(end)) & (t + window > _utc_naive(start))
    if not drop.any():
        return None
    minimum = max(1, int(dec_level_config.stft.min_num_stft_windows or 0))
    if t.size - int(drop.sum()) < minimum:
        msg = (
            f"window masks would leave band {period:.6f}s {t.size - int(drop.sum())} of "
            f"{t.size} STFT windows (minimum {minimum}): band left unmasked"
        )
        logger.warning(msg)
        return None
    logger.info(
        f"window masks: band {period:.6f}s drops {int(drop.sum())} of {t.size} STFT windows"
    )
    return ~drop


def apply_window_mask(
    X: xr.Dataset,
    Y: xr.Dataset,
    RR: Optional[xr.Dataset],
    keep: Optional[np.ndarray],
) -> tuple:
    """Drops the STFT windows `keep` marks False from X, Y and RR (all three, same time axis)."""
    if keep is None:
        return X, Y, RR
    X = X.isel(time=keep)
    Y = Y.isel(time=keep)
    if RR is not None:
        RR = RR.isel(time=keep)
    return X, Y, RR


def get_band_weights(chws, band, all_frequencies, band_frequencies) -> np.ndarray:
    """
    One weight per STFT window for `band`, from a channel weight spec's weights.

    Weights with a frequency dimension (xarray) are taken through
    chws.get_weights_for_band(band), the bin nearest the band's center frequency.
    A 2D numpy array is (time, frequency) on the STFT's frequency axis (the layout the
    previous ``weights.mean(axis=1)`` assumed): it is averaged over the band's own
    harmonics only. A 1D array is already one weight per window.
    (mt_metadata's get_weights_for_band takes axis 0 of a numpy array as frequency, which
    is the time axis here, so a plain array does not go through it.)

    Parameters
    ----------
    chws: ChannelWeightSpec
        The spec whose .weights were set by calculate_weights.
    band: mt_metadata.common.band.Band
    all_frequencies: np.ndarray
        The frequency axis of the level's STFT.
    band_frequencies: np.ndarray
        The harmonics the band was extracted with (X.frequency).

    Returns
    -------
    band_weights: np.ndarray
        Shape (n_windows,).
    """
    weights = chws.weights
    if hasattr(weights, "dims") and any("freq" in str(d) for d in weights.dims):
        return np.asarray(chws.get_weights_for_band(band)).squeeze()
    weights = np.asarray(weights)
    if weights.ndim == 1:
        return weights
    in_band = np.isin(np.asarray(all_frequencies), np.asarray(band_frequencies))
    if weights.ndim == 2 and weights.shape[1] == in_band.size and in_band.any():
        return weights[:, in_band].mean(axis=1)
    msg = (
        f"cannot take band weights for band {band.center_period:.6f}s from weights of "
        f"shape {weights.shape} (STFT has {in_band.size} frequencies)"
    )
    raise ValueError(msg)


def process_transfer_functions(
    dec_level_config: AuroraDecimationLevel,
    local_stft_obj: xr.Dataset,
    remote_stft_obj: xr.Dataset,
    transfer_function_obj,
):
    """
    This is the main tf_processing method.  It is based on the Matlab legacy code TTFestBand.m.

    Note #1: Although it is advantageous to execute the regression channel-by-channel
    vs. all-at-once, we need to keep the all-at-once to get residual covariances (see aurora issue #87)

    TODO: Consider push the nan-handling into the band extraction as a kwarg.

    Parameters
    ----------
    dec_level_config: AuroraDecimationLevel
        Processing parameters for the active decimation level.
    local_stft_obj: xarray.core.dataset.Dataset
    remote_stft_obj: xarray.core.dataset.Dataset or None
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function container ready to receive values in this method.

    Returns
    -------
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
    """
    estimator_class: RegressionEstimator = get_estimator_class(
        dec_level_config.estimator.engine
    )
    iter_control = set_up_iter_control(dec_level_config)
    for band in transfer_function_obj.frequency_bands.bands():

        X, Y, RR = get_band_for_tf_estimate(
            band, dec_level_config, local_stft_obj, remote_stft_obj
        )
        keep = window_mask_for_band(dec_level_config, band, X.time.values)
        X, Y, RR = apply_window_mask(X, Y, RR, keep)

        # Reshape to 2d
        X, Y, RR = stack_fcs(X, Y, RR)

        # Should only be needed if weights were applied
        X, Y, RR = drop_nans(X, Y, RR)

        W = effective_degrees_of_freedom_weights(X, RR, edf_obj=None)
        X, Y, RR = apply_weights(X, Y, RR, W, segment=False, dropna=True)

        if dec_level_config.estimator.estimate_per_channel:
            for ch in dec_level_config.output_channels:

                Y_ch = Y[ch].to_dataset()  # keep as a dataset, maybe not needed

                X_, Y_, RR_ = handle_nan(X, Y_ch, RR, drop_dim="observation")

                # see note #1
                # if RR is not None:
                #     W = effective_degrees_of_freedom_weights(X_, RR_, edf_obj=None)
                #     X_, Y_, RR_ = apply_weights(X_, Y_, RR_, W, segment=False)

                regression_estimator = estimator_class(
                    X=X_, Y=Y_, Z=RR_, iter_control=iter_control
                )
                regression_estimator.estimate()
                transfer_function_obj.set_tf(regression_estimator, band.center_period)
        else:
            X, Y, RR = handle_nan(X, Y, RR, drop_dim="observation")
            regression_estimator = estimator_class(
                X=X, Y=Y, Z=RR, iter_control=iter_control
            )
            regression_estimator.estimate()
            transfer_function_obj.set_tf(regression_estimator, band.center_period)

    return transfer_function_obj


def process_transfer_functions_with_weights(
    dec_level_config: AuroraDecimationLevel,
    local_stft_obj: xr.Dataset,
    remote_stft_obj: xr.Dataset,
    transfer_function_obj,
):
    """
    This is version of process_transfer_functions applies weights to the data.

    Development Notes:
    Note #1: This is only for per-channel estimation, so it does not support the
    dec_level_config.estimator.estimate_per_channel = False
    Note #2: This was adapted from the process_transfer_functions method. Each band
    is extracted once and its regression is done per output channel, with that
    channel's weights.

    Parameters
    ----------
    dec_level_config: AuroraDecimationLevel
        Processing parameters for the active decimation level.
    local_stft_obj: xarray.core.dataset.Dataset
    remote_stft_obj: xarray.core.dataset.Dataset or None
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
        The transfer function container ready to receive values in this method.

    Returns
    -------
    transfer_function_obj: aurora.transfer_function.TTFZ.TTFZ
    """
    if not dec_level_config.estimator.estimate_per_channel:
        msg = (
            "process_transfer_functions_with_weights is only for per-channel estimation"
        )
        logger.error(msg)
        raise ValueError(msg)

    estimator_class: RegressionEstimator = get_estimator_class(
        dec_level_config.estimator.engine
    )
    iter_control = set_up_iter_control(dec_level_config)

    # check if there are channel weights for each output channel
    channel_weights = {}
    for ch in dec_level_config.output_channels:
        weights = None
        channel_spec = None
        for chws in dec_level_config.channel_weight_specs:
            if ch in chws.output_channels:
                weights = chws.weights
                channel_spec = chws  # not the loop variable: that ends on the last spec
        channel_weights[ch] = (weights, channel_spec)

    for band in transfer_function_obj.frequency_bands.bands():

        X_band, Y_band, RR_band = get_band_for_tf_estimate(
            band, dec_level_config, local_stft_obj, remote_stft_obj
        )
        keep = window_mask_for_band(dec_level_config, band, X_band.time.values)
        X_band, Y_band, RR_band = apply_window_mask(X_band, Y_band, RR_band, keep)

        # the stacked X and RR and their EDF weights, shared by the output
        # channels without channel weights
        shared = {}
        for ch in dec_level_config.output_channels:
            weights, channel_spec = channel_weights[ch]
            X, RR = X_band, RR_band
            Y_ch = Y_band[ch].to_dataset()  # keep as a dataset, maybe not needed

            # extract the weights for this band
            if weights is not None:
                band_weights = get_band_weights(
                    channel_spec,
                    band,
                    local_stft_obj.frequency.values,
                    X.frequency.values,
                )
                if keep is not None:
                    band_weights = band_weights[keep]  # the windows the masks kept

                # apply_weights scales X and RR in place: the band is shared
                # by the output channels
                X = X_band.copy(deep=False)
                if RR_band is not None:
                    RR = RR_band.copy(deep=False)
                apply_weights(
                    X, Y_ch, RR, band_weights.squeeze(), segment=True, dropna=False
                )

            # Reshape to 2d, drop NaN and compute the EDF weights
            X, Y_ch, RR, W = stack_and_edf_weights(
                X, Y_ch, RR, shared=shared if weights is None else None
            )

            X, Y_ch, RR = apply_weights(X, Y_ch, RR, W, segment=False, dropna=True)
            X_, Y_, RR_ = handle_nan(X, Y_ch, RR, drop_dim="observation")
            regression_estimator = estimator_class(
                X=X_, Y=Y_, Z=RR_, iter_control=iter_control
            )
            regression_estimator.estimate()
            transfer_function_obj.set_tf(regression_estimator, band.center_period)

    return transfer_function_obj
