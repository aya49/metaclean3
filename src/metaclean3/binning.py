import numpy as np
import pandas as pd
import datetime
import warnings

from .utils import str_to_time, round_to_1

## binning functions ####
def get_time_bin_unit(
    time_values: list | np.ndarray | pd.Series,
    time_bin: str | int | float | None = None,
    bin_size_temp: int = 1000000,
    bin_no_max_temp: int = 10
):# -> str:
    """Determine a relative time bin unit.

    Args:
        time_values (list | numpy.ndarray | pandas.Series): Time channel values.
        time_bin (str | int | float | None, optional): The desired time bin
            size, in seconds or milliseconds. Defaults to None.
        bin_size_temp (int, optional): Default bin frequency.
            Defaults to 1000000.
        bin_no_max_temp (int, optional): Maximum bin size. Defaults to 10.

    Returns:
        str: Time bin unit.
    """
    if isinstance(time_bin, str):
        return time_bin

    c = round_to_1(min(int(np.ceil(
        max(time_values) / bin_size_temp)), bin_no_max_temp))
    if c == 1:
        tb = 'S'
    elif c < 0.001:
        tb = str(c * 1000) + 'ms'
    else:
        tb = str(c) + 'S'
    return tb

def adjust_time_bin(
    time_bin: str,
    frac: float = 0.5,
    reduce: bool = True
):# -> tuple:
    """Reduce a time bin size by a fraction and return the new time bin size and scale.

    Args:
        time_bin (str): The initial time bin size.
        frac (int, optional): The fraction to reduce the time bin size by.
            Defaults to 2.
        Reduce (bool, optional): whether to reduce or increase time bin.
            Defaults to True.

    Returns:
        tuple:
            str: time_bin.
            str: time_unit.
    """
    tb, time_unit = str_to_time(time_bin)
    new_tb = tb * (frac if reduce else (1/frac))
    if time_unit == 'S' and new_tb < 0.001 and reduce:
            time_unit = 'ms'
            new_tb = new_tb * 1000
    elif time_unit == 'ms':
        if reduce and new_tb < 0.001:
            time_unit = 'us'
            new_tb = new_tb * 1000
        elif not reduce and new_tb >= 1000:
            time_unit = 'S'
            new_tb = new_tb / 1000
    elif time_unit == 'us' and new_tb >= 1000 and not reduce:
        time_unit = 'ms'
        new_tb = new_tb / 1000

    time_bin = str(round_to_1(new_tb)) + time_unit
    return time_bin, time_unit

def data_to_flow(
    time_values: np.ndarray | pd.Series,
    min_bin: int = 2000,
    max_time: int = int(1e9), # 1 second
    min_time: int = 1000  # TODO: ???
):# -> pd.DataFrame:
    """Convert a dataset with timestamps into a flow dataframe.

    Args:
        time_values (numpy.ndarray | pandas.Series): Time channel values.
        min_bin (int, optional): Minimum number of bins. Defaults to 2000.
        max_time (int, optional): Maximum time scale to use.
            Defaults to 1e9 (1 second).
        min_time (int, optional): Maximum time scale to use. Defaults to 1000.

    Returns:
        pandas.DataFrame: A flow dataframe with timestamped rows
            and an 'index' column.
    """
    max_t = max(time_values)
    if max_t > max_time:
        type_dt = 'timedelta64[us]'
    elif max_t < min_time:
        type_dt = 'timedelta64[s]'
    else:
        type_dt = 'timedelta64[ms]'

    now = np.datetime64(datetime.datetime(1970, 1, 1)) + np.array(
        time_values, dtype=type_dt)
    flow = pd.DataFrame({'index': np.array(range(len(time_values)))})
    flow.index = pd.DatetimeIndex(now)

    # make time points unique
    time_point, inds, n = np.unique(
        flow.index, return_index=True, return_counts=True)
    if len(time_point) < min_bin:
        if len(time_point) == 1:
            tp = np.array(flow.index) + (max_time * np.array(range(len(flow))))
        else:
            time_interval = time_point[1] - time_point[0]
            time_eps = [time_interval / n[i] for i in range(len(time_point)-1)]
            time_eps.append(np.min(time_eps))
            tpl = [np.array(flow.index[inds[i]:(inds[i] + n[i])]) + np.cumsum(
                np.full((n[i]), time_eps[i])) for i in range(len(time_point))]
            tp = np.concatenate(tpl)

        flow.index = tp

    return flow

def _get_time_bin_while(
    flow_df: pd.DataFrame,
    time_bin: str,
    reduce: bool
):# -> tuple:
    nb = np.array(flow_df.resample(time_bin).agg({"index": "size"}))
    mask = nb > 0
    nb = nb[mask]
    last_time = time_bin
    time_bin, time_unit = adjust_time_bin(time_bin, reduce=reduce)
    return last_time, time_bin, time_unit, nb, mask

def get_time_binned(
    time_values: np.ndarray | pd.Series,
    time_bin: str = '1S',
    time_step: float | None = None,
    min_bin_size: int = 2000,
    max_bin_size: int = 10000,
    min_events_per_bin: int = 50
):# -> tuple:
    """Bin the time channel values.

    Args:
        time_values (numpy.ndarray | pandas.Series): Time channel values.
        time_bin (str, optional): Time bin i.e. duration of each bin.
            Defaults to '1S'.
        time_step (float | None, optional): Time step parameter from FCS file
            e.g. `meta.get('$TIMESTEP')`. Defaults to None.
        min_bin_size (int, optional): Minimum number of bins. Defaults to 2000.
        max_bin_size (int, optional): Maximum number of bins. Defaults to 10000.
        min_events_per_bin (int, optional): Minimum number of events per bin.
            Defaults to 50.

    Returns:
        numpy.ndarray: A 1D numpy array with the same length as `time_values`
            containing integer bin labels.
    """
    if not (time_step is None):
        # count number of events grouped by time
        bin_no = int(len(time_values) / min_events_per_bin)
        min_bin_size = min(min_bin_size, bin_no)
        flow_df = data_to_flow(time_values=time_values, min_bin=min_bin_size)

        # iteratively reduce/increase time bin
        nb = np.array(flow_df.resample(time_bin).agg({"index": "size"}))
        mask = nb > 0
        nb = nb[mask]
        last_time = time_bin
        _, time_unit = str_to_time(time_bin)
        if len(nb) < min_bin_size and time_unit != 'us':
            while len(nb) < min_bin_size and time_unit != 'us':
                last_time, time_bin, time_unit, nb, mask = _get_time_bin_while(
                    flow_df, time_bin, reduce=True)
        elif len(nb) > max_bin_size:
            while len(nb) > max_bin_size:
                last_time, time_bin, time_unit, nb, mask = _get_time_bin_while(
                    flow_df, time_bin, reduce=False)

        if last_time is None:
            em = 'data has too few values: ({})'.format(len(time_values))
            raise ValueError(em)

        time_bin = last_time
        index = np.array(flow_df.resample(time_bin).agg({"index": "max"}))[mask]
        bin_r = np.digitize(
            np.array(flow_df["index"]), np.append([0], index), right=True)
        bin_r[0] = 1
    else:
        wm = '3.X fcs required to compute extreme studentized deviate test.'
        warnings.warn(wm)

        # default time_step = 1.0
        nbin = min(min_bin_size, len(time_values) / min_events_per_bin)
        bins = np.linspace(0, len(time_values) - 1, int(nbin))
        bin_r = np.digitize(np.array(range(len(time_values))), bins)

    return bin_r


