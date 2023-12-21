import numpy as np
import copy
import warnings
import re
import difflib
import pandas as pd
from scipy.stats import gmean

## generic functions ####
def get_close_match(
    word: str,
    list_of_words: list
):# -> str:
    """Returns string from list_of_words that is the closest match to word.

    Args:
        word (str): A string.
        list_of_words (list[str]): A list of strings word will match to.

    Returns:
        str: string from list_of_words that is the closest match to word.
    """
    if len(list_of_words) == 1:
        return list_of_words[0]

    max_ind = 0
    max_score = difflib.SequenceMatcher(None, word, list_of_words[0]).ratio()
    for i in range(1, len(list_of_words)):
        word_ = list_of_words[i]
        score = difflib.SequenceMatcher(
            None, word.lower(), word_.lower()).ratio()
        if score > max_score:
            max_score = score
            max_ind = i

    return list_of_words[max_ind]

def align_list(
    list1: list,
    list2: list
):# -> list:
    """Converts list2 into list1.

    We assume both lists should be the same and we are only cleaning up list2.
    If some element in list2 does not exist in list1, this function finds
    the most similar string in list1 and replaces that element.

    Args:
        list1 (list[str]): List to compare to.
        list2 (list[str]): List to convert into list2.

    Returns:
        list2_ (list[str]): Cleaned-up version of list2.
    """

    list2_ = list2.copy()
    if len(list2_) > 0:
        for mi in range(len(list2_)):
            if list2_[mi] not in list1:
                list2_[mi] = get_close_match(list2_[mi], list1)
        list2_l = set(list2_)
        if len(list2_l) < len(list2):
            msg = 'Elements in list2 not found in list1; they will be removed.'
            warnings.warn(msg)
            list2_ = list(list2_l)

    return list2_

def arg_lim(
    x: int | float,
    xmin: int | float = -np.Inf,
    xmax: int | float = np.Inf
):
    """Sets the lower and upper limit of a given numeric variable.

    Args:
        x (int | float): Numeric variable.
        xmin (int | float, optional): Lower limit. Defaults to -numpy.Inf.
        xmax (int | float, optional): Upper limit. Defaults to numpy.Inf.

    Returns:
        int | float: `x` within the given lower and upper limit.
    """
    if xmin > xmax:
        warnings.warn('`xmin` can\'t be larger than `xmax`, ignoring limits.')
        return x

    return type(x)(max(xmin, min(xmax, x)))

def is_monotonic(x: np.ndarray, strict: bool = True):# -> bool:
    """Determines whether a vector is increasing or decreasing monotonically.

    Args:
        x: Any integer/float vector that can be converted to a 1D numpy array.
        strict (bool, optional): Whether or not to test strict monotonicity.
            Defaults to True.

    Returns:
        bool: Given vector is increasing or decreasing monotonically.
    """
    if len(x) < 2:
        return False

    x = np.array(x)
    if strict:
        increasing = all(y < z for y, z in zip(x, x[1:]))
        decreasing = all(y > z for y, z in zip(x, x[1:]))
    else:
        increasing = all(y <= z for y, z in zip(x, x[1:]))
        decreasing = all(y >= z for y, z in zip(x, x[1:]))

    res = increasing or decreasing
    return res

def mode(x: int | float):# -> int | float:
    """Returns the mode in vector `x`.

    Args:
        x: Any integer/float vector that can be converted to a 1D numpy array.

    Returns:
        int | float: The mode in vector `x`.
    """
    xu, xc = np.unique(x, return_counts=True)
    return xu[np.argmax(xc)]

def str_to_time(time_bin: str):# -> tuple:
    """Converts string to time format.

    Args:
        time_bin (str): Time in string format.

    Returns:
        tuple:
            float: Time value.
            str: Time unit.
    """
    tbs = re.findall(r'\d+', time_bin)
    time_scale = re.sub(r'\d+', '', time_bin)
    time_scale = time_scale.replace('.', '')
    tb = 1.0
    if len(tbs) > 0:
        if len(tbs) == 1:
            tb = float(tbs[0])
        else:
            tb = float(tbs[0]+'.'+''.join(tbs[1:]))

    return tb, time_scale


## remove duplicates in 2D arrays
def randomize_duplicates(
    arr: np.ndarray,
    noise_prop: float = 1 / 50000,
    strict: bool = False,
    lenient_percent: float = 0.99,
    seed: int = 123
):# -> np.ndarray: # 2D only, for rows
    """If there are any duplicate rows, adds random noise to last column.

    Args:
        arr (numpy.ndarray): 2D array
        noise_prop (float, optional): Proportion of noise.
            Defaults to 1/10000.
        strict (bool, optional): Actually find duplicates? This takes
            too long, so if false, we just find duplicate of the row sums.
            Defaults to False.duplicated_rows
        lenient_percent (float): If this percent of rows are unique,
            the function returns the original data. Defaults to 0.99.
        seed (int): Random seed. Defaults to 123.

    Returns:
        numpy.ndarray: 2D array with (almost, if not `strict`) unique rows.
    """
    arr = copy.copy(np.array(arr).astype(np.float32))

    # check if there are duplicates
    duplicated_inds, duplicated_size = duplicated_rows(
        arr, strict=strict, lenient_percent=lenient_percent)
    if duplicated_size < len(arr) * (1 - lenient_percent):
        return arr

    wm = 'There are {}/{} duplicate events ({}), will perturb values.'.format(
        duplicated_size, len(arr), 'rows' if strict else 'row sums')
    warnings.warn(wm)

    # perturb values
    ncol = arr.shape[1]
    p25 = np.quantile(arr[:, 0], .25)
    p75 = np.quantile(arr[:, 0], .75)
    np.random.seed(seed)
    noise = np.random.normal(
        loc=0.0, scale=noise_prop * (p75-p25), size=duplicated_size)
    arr[duplicated_inds, (ncol-1)] = arr[
        duplicated_inds, (ncol-1)] + noise

    return arr

def duplicated_rows(
    arr: np.ndarray,
    strict: bool = False,
    lenient_percent: float = 1.0
):# -> tuple:
    """Identifies duplicate rows in the given 2D array.

    Args:
        arr (numpy.ndarray): A 2D numpy array.
        strict (bool, optional): If set to True, finds rows with duplicate
            value. If set to False, finds rows whose sum is duplicated.
            Defaults to False.
        lenient_percent (float): If this percent of rows are unique,
            the function returns the original data. Defaults to 1.0.

    Returns:
        tuple:
            numpy.ndarray: 1D bool numpy array indicating whether each row in
                `arr` is a duplicate.
            int: Number of duplicate rows in `arr`.
    """
    row_sums = np.sum(arr, axis=1)
    if len(np.unique(row_sums)) >= len(arr) * lenient_percent:
        return np.array([]), 0

    if strict:
        duplicated_inds = pd.DataFrame(arr).duplicated()
    else:
        duplicated_inds = pd.Series(row_sums).duplicated()

    return duplicated_inds, np.sum(duplicated_inds)

def round_to_1(x: int | float):# -> float:
    """Round a number to the nearest power of 10.

    Args:
        x (int | float): The number to round.

    Returns:
        float: The rounded number.
    """
    return round(x, -int(np.floor(np.log10(abs(x)))))

def seq_array(size: int, max_val: int = 2000):# -> np.ndarray:
    """Creates an evenly-spaced integer numpy array.

    Args:
        size (int): Length of desired output.
        max_val (int, optional): Desired frequency of output.
            Defaults to 2000.

    Returns:
        numpy.ndarray: An evenly-spaced integer numpy array.
    """
    freq = int(size / max_val)
    time_seq = int(np.ceil(size / freq))
    time_seq = np.repeat([i + 1 for i in range(time_seq)], freq)
    return time_seq[:size]

def order_value_bins(
    vlist: list | np.ndarray | pd.Series,
    segments: list | np.ndarray | pd.Series,
    percentile_diff: float = 0.9
):# -> np.ndarray:
    """Order segments by their geometric means and quantiles.

    Args:
        vlist (list | numpy.ndarray | pandas.Series): 2D array to be sorted.
        segments (list | numpy.ndarray | pandas.Series): Segment label of values.
        percentile_diff (float, optional): Percentile to be evaluated while
            sorting. For example, if set to 0.9, we evaluate the 10th and 90th
            percentiles. Defaults to 0.9.

    Returns:
        numpy.ndarray: list of sorted indices.
    """
    d = pd.DataFrame()
    value_len = 1
    if isinstance(vlist, np.ndarray):
        if len(vlist.shape) == 1:
            vlist = pd.Series(vlist)
        elif len(vlist.shape) == 2:
            vlist = pd.DataFrame(vlist)
        else:
            raise ValueError('Invalid input. Please provide a 1 or 2d array.')
    if isinstance(vlist, list):
        value_len = len(vlist)
        for vi in range(len(vlist)):
            d['v{}'.format(vi)] = vlist[vi]
    elif isinstance(vlist, pd.DataFrame):
        value_len = len(vlist.columns)
        for vi in range(len(vlist.columns)):
            d['v{}'.format(vi)] = vlist[vlist.columns[vi]]
    elif isinstance(vlist, pd.Series):
        d['v0'] = vlist
    else:
        raise ValueError('Invalid input. Please provide a 1 or 2d array.')
    d.index = range(len(d)) # pandas version reconciliation
    if len(segments) != len(d):
        raise ValueError('Segment and value length differ.')

    d['segments'] = segments
    if len(np.unique(segments)) == 1:
        return np.array(range(len(segments)))

    # order segments by their mean
    dg = d.groupby('segments')
    seg_means_all = np.zeros((value_len, len(dg)))
    for vi in range(value_len):
        xi = 0
        for x in dg.groups:
            dgv = dg.get_group(x)['v{}'.format(vi)]
            neg_diff = min(np.min(dgv), 0)
            gm = gmean(dgv - neg_diff) + neg_diff
            q1 = np.quantile(dgv, q=1-percentile_diff) /2
            q2 = np.quantile(dgv, q=percentile_diff) /2
            seg_means_all[vi, xi] = gm + q1 + q2
            xi += 1
    seg_means = np.sum(seg_means_all, axis=0)
    seg_inds = np.array(list(dg.groups.keys()))[np.argsort(seg_means)]

    # rearrange segment indices
    dg_index = np.array([], dtype=int)
    for i in seg_inds:
        dg_index = np.append(dg_index, dg.get_group(i).index)

    return dg_index

def get_timestep(meta: dict, timestep_key: str = '$TIMESTEP'):# -> float | None:
    """Extract timestep from flow cyotometry standard file meta data.

    Args:
        meta (dict): Flow cytometr standard file meta data.
        timestep_key (str, optional): Timestep key in meta data.
            Defaults to '$TIMESTEP'.

    Returns:
        float | None: Timestep if it exists, otherwise None.
    """
    if timestep_key in meta.keys():
        return float(meta[timestep_key])
    return None

## FCS preprocessing functions ####
# Assign column names to index (from fcmdata_helpers).
def _clean_spil_index(spil_df):
    key = spil_df.index
    value = spil_df.columns
    dico = dict(zip(key, value))
    spil_df = spil_df.rename(dico, axis='index')
    return spil_df

# Clean spillover names according to some known formats (from fcmdata_helpers).
def _clean_spil_names(spil_names, keep_left=True):
    new_names = {}
    for name in spil_names:
        # Replace eventual flowJo addition
        if isinstance(name, str) and ' :: ' in name:
            if keep_left:
                # Keep only first part
                new_names[name] = name[:name.find(' :: ')]
            else:
                # Keep second part
                new_names[name] = name[name.find(' :: ') + 4:]
    return new_names

# Apply cleaning on spillover matrix (from fcmdata_helpers).
def _clean_spillover(spillover, data_col=None, keep_left=True):
    if spillover.shape[0] != spillover.shape[1]:
        raise Exception('Error spillover matrix: spillover should be square')

    if np.sum(np.diag(spillover)) != spillover.shape[0]:
        raise Exception('Error spillover matrix: diag should be 1')

    clean_columns = _clean_spil_names(spillover.columns, keep_left)

    spillover = spillover.rename(
        columns=clean_columns
    )
    spillover = _clean_spil_index(spillover)

    if data_col is not None:
        # clean spillover according to data
        intersection = [list(spillover.columns).index(val)
                        for val in data_col if val in list(spillover.columns)]
        if len(intersection) == len(spillover.columns):
            # Reorder spillover matrix
            spillover = spillover.iloc[intersection, intersection]
        else:
            raise Exception(
                "Spillover matrix error: columns don't match with data.")

    # check duplicates
    chan_corrected = copy.copy(list(spillover.columns))
    chan_to_keep = copy.copy(list(spillover.columns))
    for ii in range(len(chan_to_keep)):
        for jj in range(ii + 1, len(chan_to_keep)):
            if sum(abs(spillover[chan_to_keep[ii]] -
                       spillover[chan_to_keep[jj]]) < 1e-10) == len(spillover):
                if chan_to_keep[jj] in chan_corrected:
                    chan_corrected.remove(chan_to_keep[jj])

    return spillover.loc[chan_corrected, chan_corrected]

def get_spillover_raw(meta: dict, dat_columns: list): #list(dat.columns)
    """Extract spillover matrix from fcs meta data.

    Args:
        meta (dict): FCS meta data.
        dat_columns (list): Column names of data.

    Returns:
        pandas.DataFrame: Spillover matrix.
    """
    try:
        s = meta[get_close_match('spill', list(meta.keys()))].split(',')
        n = int(s[0])
        sd = align_list(list1=dat_columns, list2=s[1:(n+1)])  # just in case
        sm = np.array(s[(n+1):]).astype('float64').reshape(n, n)
        n = len(sm)

        # correct diagonal of spill matrix
        smid = True
        for smi in range(sm.shape[0]):
            if sm[smi,smi] != 1.0:
                smid = False
                break
        if smid:
            sm += np.eye(sm.shape[0]) - np.diag(np.diag(sm))

        return pd.DataFrame(sm, columns=sd)
    except:
        warnings.warn("No spillover matrix found.")
        return None

def apply_compensation_matrix(
    data: pd.DataFrame,
    spillover: pd.DataFrame | None
):
    """Apply spillover compensation on data (from fcmdata_helpers).

    Spillover columns should match some subset of data columns.

    Args:
        data (pandas.DataFrame): An event x feature matrix.
        spillover (pandas.DataFrame): spillover matrix to compensate data.

    Returns:
        pandas.DataFrame: Compensated data for columns specified in spillover.
    """
    # TODO: do check on spillover to ensure spillover can be properly applied
    # See https://onlinelibrary.wiley.com/doi/full/10.1002/cyto.a.22018
    if spillover is None:
        return data

    if not isinstance(spillover, pd.DataFrame):
        raise ValueError('Spillover matrix should be a pandas dataframe')

    # Copy data to not overwrite existing dataframe
    data_copy = data.copy()
    spillover = _clean_spillover(spillover, data_copy.columns)
    try:
        spillover_inv = np.linalg.inv(spillover)
    except BaseException:
        spillover_inv = np.linalg.pinv(spillover)
    # X = X * S^(-1)
    data_copy.loc[:, spillover.columns] = np.dot(
        data_copy.loc[:, spillover.columns], spillover_inv)

    return data_copy
