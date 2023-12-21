import numpy as np
import pandas as pd
import warnings

from difflib import get_close_matches
from .utils import is_monotonic, seq_array

## channel functions ####
def get_clean_time_chan(
    data: pd.DataFrame,
    time_chan: str = 'time',
    min_bin_size: int = 2000
):# -> tuple:
    """Finds the time channel in data; if there is no time channel, creates one.

    Args:
        data (pandas.DataFrame): _description_
        time_chan (str, optional): _description_. Defaults to 'time'.
        min_bin_size (int, optional): _description_. Defaults to 2000.

    Returns:
        tuple:
            str: Time channel column name in data.
            pandas.DataFrame: The given data sorted by the time channel.
    """
    if time_chan in data.columns:
        tc = time_chan
    else:
        tc = [tci for tci in data.columns if tci.lower() == time_chan.lower()]
        tc_none = len(tc) == 0
        if len(tc) == 1:
            tc = tc[0]
        elif len(tc) > 1:
            tc = get_close_matches(time_chan, tc)[0]
        elif not isinstance(tc, str):
            tc_none = True

        # if time channel doesn't exist, create one
        if tc_none:
            wm = 'No valid time channel in data; proceeding to create one.'
            warnings.warn(wm)
            data[time_chan] = seq_array(size=len(data), max_val=min_bin_size)
            tc = time_chan

    if not is_monotonic(data[tc], strict=False):
        data.sort_values(by=tc, inplace=True)

    # i.e. data.reset_index(drop=False, inplace=True)
    data.index = np.array(range(len(data)))
    return tc, data

def verify_channels(given_chans: list = [], all_chans: list = []):# -> np.ndarray:
    """Verifies whether elements of `given_chans` are in `all_chans`.

    Args:
        given_chans (list): User given strings.
        all_chans (list): The full string list.

    Returns:
        numpy.ndarray: User given strings that are in the full string list.
    """
    if len(given_chans) == 0 or len(all_chans) == 0:
        return []

    gcs = [gc.lower() for gc in given_chans]
    acs = [ac.lower() for ac in all_chans]
    ch_in_data = np.isin(gcs, acs)
    if any(ch_in_data):
        if not all(ch_in_data):
            wm = 'Dropping channels not found in usable data: {}'.format(
                np.array(given_chans)[~ch_in_data])
            warnings.warn(wm)
        gcs = np.array(given_chans)[ch_in_data]
    else:
        warnings.warn('Channels not found in usable data, to find manually.')
        gcs = []

    return np.array(gcs)

def get_clean_fp_channels(
    data: pd.DataFrame,
    fluo_chans: list = [],
    phys_chans: list = [],
    channel_unique_no: int = 25, # omip 018 has very little unique values
    phys_channel_suffix: list = [
        'fs', 'ss', 'area', 'eccentricity', 'forward', 'side'],
    bad_suffix: list = ['bead', 'event', 'label', 'is_gate',
                        'index', 'index_original', 'bin', 'time']
):# -> tuple:
    """Identify fluorescent channels.

    Args:
        data (pandas.DataFrame): FCS pandas.DataFrame.
        fluo_chans (list): String vector containing fluorescent channel names.
            Defaults to None.
        phys_chans (list): String vector containing physical morphology
            channel names. Defaults to None.
        channel_unique_no (int): Minimum number of rows in each column that
            can be non-unique. Defaults to 25.
        phys_channel_suffix (list): Standard suffixes for phys_chans.
            Defaults to ['fs', 'ss', 'area', 'eccentricity', 'forward', 'side'].
        bad_suffix (list): Suffixes to avoid. Defaults to ['bead', 'event',
            'label', 'is_gate', 'index', 'index_original', 'bin', 'time'].

    Returns:
        tuple:
            numpy.array: fluorescent channels that can be used to clean FCS data.
            numpy.array: physical morphology channels.
    """
    # validate all columns
    cc = data.columns
    cc = [f for f in cc if not np.any([
        f.lower().startswith(bs) for bs in bad_suffix])]
    if len(cc) == 0:
        raise ValueError('No valid channels found in data.')
    cc = [c for c in cc if not (is_monotonic(np.array(data[c]), strict=False))]
    if len(cc) == 0:
        raise ValueError('All channels are monotonically in/decreasing.')
    channel_unique_no = min(channel_unique_no, int(len(data) / 2))
    cc = [c for c in cc if len(np.unique(data[c])) > channel_unique_no]
    if len(cc) == 0:
        em = 'All channels have less than {} unique values.'.format(
            channel_unique_no)
        raise ValueError(em)

    # verify user given physical channels
    pc = verify_channels(given_chans=phys_chans, all_chans=cc)
    if len(pc) == 0:
        pc = [f for f in cc if np.any(
            [f.lower().startswith(ps) for ps in phys_channel_suffix])]

    # verify user given fluorescent channels
    fc = verify_channels(given_chans=fluo_chans, all_chans=cc)
    if len(fc) == 0:
        fc1 = cc if len(pc) == 0 else [f for f in cc if not (f in pc)]
        if len(fc1) > 0:
            fc2 = [f for f in fc1 if f.lower().endswith('-a')]
            if len(fc2) == 0:
                fc2 = fc1
            fc = fc2 if len(fc2) >= len(fc) else fc

    # verify user given physical channels part 2
    if len(pc) > 0:
        pc2 = [p for p in pc if p.lower().endswith('-a')]
        if len(pc2) == 0:
            pc2 = pc
        pc = pc2 if len(pc2) <= len(pc) and len(pc2) > 0 else pc

    return np.array(fc), np.array(pc)

def most_corr_channels(
    data: pd.DataFrame,
    chosen_chans: np.ndarray | pd.Series | None = None,
    bins: np.ndarray | pd.Series | None = None,
    candidate_no: int = 4,
    min_nrows: int = 50,
    corr_type: str = 'max'
):# -> np.ndarray:
    """Verifies chosen channels and finds channels most correlated with time.

    Args:
        data (pd.DataFrame): FCS data matrix.
        chosen_chans (np.ndarray | pd.Series | None, optional):
            User chosen channels. Defaults to None.
        bins (np.ndarray | pd.Series | None, optional): Bin labels.
            Defaults to None.
        candidate_no (int, optional): Number of channels to return.
            Defaults to 4.
        min_nrows (int, optional): Minimum number of rows required to calculate
            correlation. Defaults to 50.
        corr_type (str, optional): Type of summarization to use on bins to
            calculate correlation e.g. `min`, `max`, `median`, `mean`.
            See `pandas.DataFrame.agg`. Defaults to 'max'.

    Returns:
        np.ndarray: Vector of candidate channel names.
    """
    # check if user chosen channels are in data
    if not (chosen_chans is None):
        corr_ch_in_data = np.isin(chosen_chans, data.columns)
        if any(corr_ch_in_data):
            if not all(corr_ch_in_data):
                wm = 'Dropping channels not found in data: {}'.format(
                    chosen_chans[~corr_ch_in_data])
                warnings.warn(wm)
            return chosen_chans[corr_ch_in_data]
        warnings.warn('Chosen channels not found in data, will find manually.')

    # check edge cases
    candidate_chans = np.array(data.columns)
    if len(data) <= min_nrows:
        warnings.warn('Data has too little rows, returning random channels.')
    if len(candidate_chans) <= candidate_no:
        return candidate_chans
    if len(data) <= min_nrows:
        return candidate_chans[:candidate_no]

    # calculate correlation
    bins = np.array(range(len(data))) if bins is None else bins
    corr_bin = data.groupby(bins).agg(corr_type)
    _, time_binned_nb = np.unique(bins, return_index=True)
    # correlation with bin rate
    # time_binned_nb = np.diff(np.concatenate((inds, [len(data)])))
    corrs = np.zeros((len(candidate_chans)))
    for ci in range(len(candidate_chans)):
        chan = candidate_chans[ci]
        if len(np.unique(corr_bin[chan])) == 1:
            corrs[ci] = 0.0
        else:
            corrs[ci] = np.corrcoef(time_binned_nb, corr_bin[chan])[0, 1]

    return np.array(candidate_chans)[np.argsort(0-np.abs(corrs))[:candidate_no]]

