import numpy as np
import pandas as pd
import ruptures as rpt

# TODO: shrinkable
def list_gains(
    chpts0M: list | np.ndarray | pd.Series,
    rpt_class: rpt.Binseg,
    chpts_inds: list | np.ndarray | pd.Series | None = None,
    dif: bool = True
):# -> np.ndarray:
    """List the gains for values fit on a ruptures class on given changepoints.

    Args:
        chpts0M (list | numpy.ndarray | pandas.Series): A vector listing the
            index of the first element in each segment + a last element that
            indicates the length of the fitted values.
        rpt_class (rpt.Binseg): A ruptures.Binseg object that is already fit on
            values we are calculating gains on.
        chpts_inds (list | numpy.ndarray | pandas.Series | None, optional):
            If the user only wants to calculate gains on a subset of the
            given changepoints, `chpts_inds` is a vector that lists the
            indices of `chpts_inds` for which a gain should be calculated.
            Defaults to None.
        dif (bool, optional): Whether to use gain difference (`True`) or to use
            gain ratio  (`False`). Defaults to True.

    Returns:
        numpy.ndarray: A 1D array containing a gain for each given changepoint.
    """
    gain = np.full((len(chpts0M)), -1.0)
    # prepare indices to calculate gain/gains for
    if chpts_inds is None:
        chpts_inds = np.array(range(1, len(chpts0M)-1))

    chpts_inds = np.array(chpts_inds)[
        np.isin(chpts_inds, np.array(range(1, len(chpts0M)-1)))]
    if len(chpts_inds) == 0:
        # warnings.warn('Chosen indices not in chpt list, returning -1s.')
        return gain

    chpts_inds = np.sort(chpts_inds)
    for ci in chpts_inds:
        start = chpts0M[ci-1]
        bkp = chpts0M[ci]
        end = chpts0M[ci+1]
        g = binseg_gain(start, bkp, end, rpt_class=rpt_class, dif=dif)
        gain[ci] = np.squeeze(np.array(g))

    return gain

def binseg_gain(
    start: int,
    bkp: int,
    end: int,
    rpt_class: rpt.Binseg,
    dif: bool = True
):# -> float:
    """Calculates the gain of a single changepoint.

    Args:
        start (int): Start index (first inde of first segment).
        bkp (int): Breakpoint index (first inde of second segment).
        end (int): End index (not inclusive).
        rpt_class (rpt.Binseg): A ruptures.Binseg (for example) object that is
            already fit on values we are calculating the gain on.
        dif (bool, optional): Whether to use gain difference (`True`) or to use
            gain ratio  (`False`). Defaults to True.

    Returns:
        float: Gain value.
    """
    segment_cost0 = rpt_class.cost.error(start, end)
    try:
        d = (rpt_class.cost.error(start, bkp) + rpt_class.cost.error(bkp, end))
        if dif:
            return segment_cost0 - d
        else:
            return segment_cost0 / d
    except rpt.exceptions.NotEnoughPoints:
        return 0

def chpts0M_to_segments(chpts0M: list | np.ndarray | pd.Series):# -> np.ndarray:
    """Converts changepoint indices to an integer segment label array.

    Args:
        chpts0M (list | numpy.ndarray | pandas.Series): Changepoint indices
            where the first value is 0 and the last value is the length of the
            desired segment label array.

    Returns:
        numpy.ndarray: A 1D integer segment label array.
    """
    if len(chpts0M) == 2:
        return np.full((chpts0M[-1]), 0)

    segments = np.full((chpts0M[-1]), -1)
    for i, ch in enumerate(chpts0M[:-1]):
        segments[ch:chpts0M[i+1]] = i
    return segments

def segments_to_chpts0M(segments: list | np.ndarray | pd.Series):# -> np.ndarray:
    """Converts an integer segment label array to changepoint indices.

    Args:
        segments (list | numpy.ndarray | pandas.Series):
            A 1D integer segment label array.

    Returns:
        numpy.ndarray: A 1D integer changepoint indices array.
    """
    chpts = np.where(np.diff(segments) != 0)[0] + 1
    chpts0M = np.concatenate(([0], chpts, [len(segments)]))
    return chpts0M

