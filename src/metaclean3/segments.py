import pandas as pd
import numpy as np
from typing import Callable, Literal
from operator import gt, lt
import ruptures as rpt

from scipy.stats import gmean, ranksums, ttest_ind

from kneed import KneeLocator

from .utils import mode
from .changepoint import segments_to_chpts0M, list_gains, chpts0M_to_segments
# from .plot import plot_scat

# TODO longterm: shorten functions

# checks arguments chpts0M, r, and returns rl if it is none (using segments).
def _check_chpts0M_segments(
    chpts0M = None, segments = None, r = None, rl = None):# -> int | None:
    if not (chpts0M is None):
        if not (len(chpts0M) > 1 and chpts0M[0] == 0 and chpts0M[-1] > 0):
            raise ValueError('Invalid changepoints (chpts0M).')
        if not (r is None) and not (r >= 0 and r < chpts0M[-1]):
            raise ValueError('Invalid reference changepoint index (r).')
    if not (segments is None) and len(segments) > 0:
        if rl is None:
            return mode(segments)
        elif not (rl in segments):
            raise ValueError('Invalid reference segment label (rl).')
        else:
            return rl

def get_ref_ranges(
    ref_values: list,
    ignore_no: int = 5,
    percent_diff: float = 0.05,
    percent_shift: float = 0.1
):# -> tuple:
    """
    Get ranges (quantiles) for reference values.
    Intended for internal use; made public for external QA.

    Args:
        ref_values (list): A list of numeric vector(s) that the
            function will create ranges for.
        ignore_no (int, optional): See `in_percentiles()`. Defaults to 5.
        percent_diff (float, optional): See `in_percentiles()`. Defaults to 0.05.
        percent_shift (float, optional): See `in_percentiles()`. Defaults to 0.1.

    Returns:
        tuple:
            list: a list of four floats containing upper and lower bounds of
                maximum and minimum ranges.
            list: A list of floats representing the shift for each element in
                `ref_values`.
    """
    ref_range = [np.max(ref_values[0]), np.min(ref_values[0])] * 2
    ref_shift_all = []
    for ref_value in ref_values:
        if len(ref_value) < ignore_no:
            continue
        ref_percentiles = (
            np.quantile(ref_value, percent_diff),
            np.quantile(ref_value, 1-percent_diff))
        ref_shift = (percent_shift * (ref_percentiles[1] - ref_percentiles[0]))
        ref_shift_all.append(ref_shift)
        ref_range_ = [
            ref_percentiles[0] - ref_shift,
            ref_percentiles[0] + ref_shift,
            ref_percentiles[1] - ref_shift,
            ref_percentiles[1] + ref_shift]

        ref_range[0] = min(ref_range[0], ref_range_[0])
        ref_range[1] = max(ref_range[1], ref_range_[1])
        ref_range[2] = min(ref_range[2], ref_range_[2])
        ref_range[3] = max(ref_range[3], ref_range_[3])

    return ref_range, ref_shift_all

def get_test_ranges(
    ref_values: int,
    test_value: list | np.ndarray | pd.Series,
    small_no_per_seg: int = 50,
    percent_diff: float = 0.05,
    percent_shift: float = 0.1,
    lenient: bool = True
):# -> list:
    """
    Get ranges (quantiles) for test values.
    Intended for internal use only; made public for external QA.

    Args:
        ref_lens (int): The maximum length of value vectors in the
            reference segment; most of the time, this is the length of the
            reference segment, but an arbitrary length can be given.
        test_value (list | numpy.ndarray | pandas.Series): A numeric vector that
            the function will create ranges for.
        small_no_per_seg (int, optional): Vectors with a smaller length than
            this value is considered a "small" vector;
            small vectors are treated differently because
            smaller segments can contain more sporadic values. Defaults to 50.
        percent_diff (float, optional): See `in_percentiles()`. Defaults to 0.05.
        percent_shift (float, optional): See `in_percentiles()`. Defaults to 0.1.
        lenient (bool, optional): See `in_percentiles()`. Defaults to True.

    Returns:
        list: a list of four floats containing upper and lower bounds of
                maximum and minimum ranges.
    """
    if len(test_value) <= small_no_per_seg:
        test_range = [
            np.min(test_value),
            np.quantile(test_value, percent_diff), # * 2 to be 2x lenient
            np.quantile(test_value, 1-percent_diff),
            np.max(test_value)]
        # TODO: put into ref_func
        if lenient:
            ref_len = np.argmax([len(r) for r in ref_values])
            in_range = np.where(
                (test_range[0] <= ref_values[ref_len]) &
                (ref_values[ref_len] > test_range[3]))[0]
            in_m1 = np.diff(in_range) >= len(test_value) * (percent_diff * 2)
            if np.sum(in_m1) > 0:
                r0 = ref_values[ref_len]
                gi = np.where(in_m1)[0]
                ref_values.append(
                    np.concatenate([r0[in_range[g]:in_range[g+1]] for g in gi]))
    else:
        test_range = [
            np.quantile(test_value, percent_diff * (1 - percent_shift)),
            np.quantile(test_value, percent_diff * (1 + percent_shift)),
            np.quantile(test_value, 1 - (percent_diff * (1 + percent_shift))),
            np.quantile(test_value, 1 - (percent_diff * (1 - percent_shift)))]

    return test_range

def apply_ranges(
    ref_range: list | np.ndarray,
    test_range: list | np.ndarray
):# -> bool:
    """
    Test whether test ranges fall within reference ranges.
    Intended for internal use only; made public for external QA.

    Args:
        ref_range (list | numpy.ndarray): A reference ranges vector of length 4.
        test_range (list | numpy.ndarray): A test ranges vector of length 4.

    Returns:
        bool: Whether or not test ranges fall within reference ranges.
    """
    if not (len(test_range) == 4) or not (len(ref_range) == 4):
        raise ValueError('Ranges/quantiles must be of length 4.')
    cond = (
        ((test_range[0] >= ref_range[0] and ref_range[1] > test_range[0]) or
        (test_range[1] >= ref_range[0] and ref_range[1] > test_range[1]) or
        (test_range[0] < ref_range[0] and ref_range[1] <= test_range[1])) and
        ((test_range[2] >= ref_range[2] and ref_range[3] > test_range[2]) or
        (test_range[3] >= ref_range[2] and ref_range[3] > test_range[3]) or
        (test_range[2] <= ref_range[2] and ref_range[3] <= test_range[3]))
    )
    return cond

# calculate statistics from and merge/refine segments
def in_percentiles(
    ref_values: list,
    test_value: list | np.ndarray | pd.Series,
    check_mean: bool = False,
    check_var: bool = False,
    switch: bool = True,
    ignore_no: int = 5,
    small_no_per_seg: int = 50,
    percent_diff: float = 0.05,
    percent_shift: float = 0.1,
    lenient: bool = True
):# -> bool:
    """
    Test whether test and reference values have a similar distribution.
    Intended for internal use only; made public for external QA.

    Args:
        ref_values (list): A vector of reference values.
        test_value (list | numpy.ndarray | pandas.Series): A vector of test
            values.
        check_mean (bool, optional): Whether or not to check similarity of
            reference and test means. Defaults to False.
        check_var (bool, optional): Whether or not to check similarity of
            reference and test variance. Defaults to False.
        switch (bool, optional): Whether or not to check similarity of reference
            and test mean and variance if test ranges do not fall within
            reference ranges. Defaults to True.
        ignore_no (int, optional): Indicates the minimum length of an acceptable
            vector. Defaults to 5.
        small_no_per_seg (int, optional): Vectors with a smaller length than
            this value is considered a "small" vector;
            small vectors are treated differently because
            smaller segments can contain more sporadic values. Defaults to 50.
        percent_diff (float, optional): Quantile to be calculated from 0 and 1.
            Defaults to 0.05.
        percent_shift (float, optional): Percent above and below mean of values.
            Defaults to 0.1.
        lenient (bool, optional): If `test_value` is a small vector, users can
            choose whether or not the ranges should be lenient or not.
            Defaults to True.

    Returns:
        bool: Indicates whether or not test ranges are within reference ranges.
    """
    # if segment is small enough, just keep it
    rall = all([len(x) <= ignore_no for x in ref_values])
    if len(test_value) <= ignore_no or rall:
        return True

    # create reference segments ranges
    ref_range, ref_shift_all = get_ref_ranges(
        ref_values=ref_values, ignore_no=ignore_no,
        percent_diff=percent_diff, percent_shift=percent_shift
    )
    # create test segment ranges + adjust ref values
    test_range = get_test_ranges(
        ref_values=ref_values, test_value=test_value,
        small_no_per_seg=small_no_per_seg,
        percent_diff=percent_diff, percent_shift=percent_shift, lenient=lenient
    )
    # compare test and reference segment ranges
    cond = apply_ranges(ref_range, test_range)
    # compare test and reference segment variances
    if check_var:
        cond_ = False
        tvar = (np.quantile(test_value, 1-percent_diff) -
                np.quantile(test_value, percent_diff)) # np.var(test_value)
        for ref_value, ref_shift in zip(ref_values, ref_shift_all):
            rvar = (np.quantile(ref_value, 1-percent_diff) -
                    np.quantile(ref_value, percent_diff)) # np.var(test_value)
            cond_ = cond_ or (np.abs(tvar-rvar) < ref_shift)
        cond = cond and cond_
    # compare test and reference segment means
    if check_mean:
        cond_ = False
        neg_diff = min(np.min(test_value), 0)
        tmean = gmean(test_value - neg_diff, nan_policy='omit') + neg_diff
        for ref_value, ref_shift in zip(ref_values, ref_shift_all):
            neg_diff = min(np.min(ref_value), 0)
            rmean = gmean(ref_value - neg_diff, nan_policy='omit') + neg_diff
            cond_ = cond_ or (
                tmean >= rmean - ref_shift and tmean < rmean + ref_shift)
        cond = cond and cond_
    # compare test and reference segment var/means only if range test fails
    if switch and not cond:
        cond = False
        tvar = (np.quantile(test_value, 1-percent_diff) -
                np.quantile(test_value, percent_diff))
        for ref_value, ref_shift in zip(ref_values, ref_shift_all):
            rvar = (np.quantile(ref_value, 1-percent_diff) -
                    np.quantile(ref_value, percent_diff)) # np.var(test_value)
            neg_diff = min(np.min(np.concatenate((test_value, ref_value))), 0)
            tmean = gmean(test_value - neg_diff, nan_policy='omit')
            rmean = gmean(ref_value - neg_diff, nan_policy='omit')
            cond_ = (np.abs(tvar - rvar) < ref_shift) and (
                     tmean >= rmean - ref_shift and tmean < rmean + ref_shift)
            cond = cond or cond_

    return cond

def get_ref_updown(
    r: int,
    chpts0M: list | np.ndarray | pd.Series,
    vlist: list,
    ref_percents: list = [1, 0.5],
    ref_percent: float = 0.4
):# -> list:
    """
    Extracts values for reference segment.
    Intended for internal use only; made public for external QA.

    Args:
        r (int): Index of the reference segment in `chpts0M`.
        chpts0M (list | numpy.ndarray | pandas.Series): A vector containing the
            first index of each segment.
        vlist (list): List of numeric vector of values.
        ref_percents (list, optional): List of percent of reference values
            next to the test values to extract. Defaults to [1, 0.5].
        ref_percent (float, optional): Percent of length of `vlist`;
            a minimum of this value and the length of reference values is taken
            as another alternative to how many reference next to the test values
            to extract. Defaults to 0.4.

    Returns:
        list: List of reference values extracted.
    """
    _check_chpts0M_segments(chpts0M=chpts0M, r=r)
    rps = np.append(
        [int((chpts0M[r+1]-chpts0M[r]) * p) for p in ref_percents],
        [min(int(chpts0M[-1] * ref_percent), chpts0M[r+1] - chpts0M[r])])
    ref_values = [
        [[np.array(v[chpts0M[r]:(chpts0M[r]+rp)])
            for rp in rps] for v in vlist],
        [[np.array(v[(chpts0M[r+1]-rp):chpts0M[r+1]])
            for rp in rps] for v in vlist]]
    return ref_values

def go_updown(
    r: int,
    chpts0M: list | np.ndarray,
    lim: int,
    refs: list,
    vlist: list,
    ip_func: Callable = in_percentiles,
    exception_no: int = 2,
    ignore_no: int = 5
):# -> tuple:
    """
    Tests whether reference segment is similar to its adjacent segment(s).
    Intended for internal use only; made public for external QA.

    Args:
        r (int): Index of the reference segment in `chpts0M`.
        chpts0M (list | numpy.ndarray | pandas.Series): A vector containing the
            first index of each segment.
        lim (int): Upper limit for the index of segments to test i.e. the index
            of the end or the next reference segment to test .
        refs (list): List of reference vectors.
        vlist (list): List of all value vectors.
        ip_func (Callable, optional): Function used to determine whether test
            and reference values are similar. Defaults to `in_percentiles()`.
        exception_no (int, optional): Users can choose to check test values
            `exception_no` segments away from reference segment
            (i.e. non-adjacent test segments). Defaults to 2.
        ignore_no (int, optional): See `in_percentiles()`. Defaults to 5.

    Returns:
        tuple:
            bool: Whether test values and adjacent reference values have similar
                distribution.
            numpy.ndarray: Indices of segments that were skipped in between test
                and similar reference segment.
    """
    # set parameters
    _check_chpts0M_segments(chpts0M=chpts0M, r=r)
    go_up = r < lim
    op = lt if go_up else gt
    ec = 1 if go_up else -1
    df = ec * 2

    # start: check segments before and after segment r for merging
    en = 0
    go = False
    skipped_chpt_ = np.array([], dtype=int)
    while not go and op(r+ec+en, lim) and abs(en) < exception_no:
        while ((en > 0) and op(r+df+en, lim) and (
            chpts0M[r+ec+1+en]-chpts0M[r+ec+en] <= ignore_no)):
            s = chpts0M[r+ec+en]
            if s not in skipped_chpt_:
                skipped_chpt_ = np.concatenate((
                    skipped_chpt_, [s, chpts0M[r+ec+1+en]]))
            en += ec
        if op((r+ec+en), lim):
            go = True
            tests = [v[chpts0M[r+ec+en]:chpts0M[r+ec+1+en]] for v in vlist]
            for test_value, ref_values in zip(tests, refs):
                rli = np.argmax([len(rl) for rl in ref_values])
                new_len = min(len(ref_values[rli]), len(test_value))
                if len(ref_values[rli]) > new_len:
                    if go_up:
                        ref_values.append(ref_values[rli][-new_len:])
                    else:
                        ref_values.append(ref_values[rli][:new_len])
                go_ = ip_func(ref_values=ref_values, test_value=test_value,
                              check_mean=en != 0)
                go = go and go_
            s = chpts0M[r+ec+en]
            if not go and s not in skipped_chpt_:
                skipped_chpt_ = np.concatenate((
                    skipped_chpt_, [s, chpts0M[r+ec+1+en]]))
        if not go:
            en += ec
    return go, skipped_chpt_

def get_ref_func2(
    vlist: list,
    ref_inds: np.ndarray,
    test_i1: int,
    test_i2: int,
    # test_i1_raw: int | None = None,
    # test_i2_raw: int | None = None,
    small_no_per_seg: int = 50,
    small_ref_prop: float = 0.5,
    sorted_index: np.ndarray | None = None
):# -> list:
    """
    Extracts values for reference segment; this function is an alternative to
    `get_ref_updown()` for when the user wants to subset values in
    the reference segment that are closest to some test segment. The reference
    segment indices do not need to be continuous.
    Intended for internal use only; made public for external QA.

    Args:
        vlist (list): List of value vectors.
        ref_inds (numpy.ndarray): Indices of reference values.
        test_i1 (int): Starting index of test segment.
        test_i2 (int): Ending index of test segment.
        small_no_per_seg (int, optional): Vectors with a smaller length than
            this value is considered a "small" vector;
            small vectors are treated differently because
            smaller segments can contain more sporadic values. Defaults to 50.
        small_ref_prop (float, optional): Proportion of reference segment to
            test. Defaults to 0.5.
        sorted_index (numpy.ndarray | None, optional): Indices sorted based on
            segment quantiles; see `utils.order_value_bins()`. Defaults to None.

    Returns:
        list: List of reference value vectors.
    """
    # test_i1_raw = test_i1_raw if not (test_i1_raw is None) else test_i1
    # test_i2_raw = test_i2_raw if not (test_i2_raw is None) else test_i2
    test_len = test_i2 - test_i1
    big_test = test_len > small_no_per_seg and len(ref_inds) > test_len
    refs = [[v[ref_inds]] for v in vlist]
    for vi in range(len(vlist)):
        v = vlist[vi]
        if big_test:
            med_diff = ref_inds - np.median([test_i1, test_i2-1])
            ref_close_inds = np.argsort(np.abs(med_diff))[:test_len]
            refs[vi].append(v[ref_inds[ref_close_inds]])
            if not (sorted_index is None):
                sorted_ref_inds = np.where(np.isin(sorted_index, ref_inds))[0]
                med_diff = sorted_ref_inds - np.median(
                    np.where((sorted_index==test_i1) |
                             (sorted_index==(test_i2-1)))[0])
                goodi = sorted_ref_inds[np.argsort(np.abs(med_diff))[:test_len]]
                ref_close_inds = sorted_index[goodi]
                refs[vi].append(v[ref_close_inds])
        elif test_len <= small_no_per_seg:
            smin = np.min(vlist[vi][test_i1:test_i2])
            smax = np.max(vlist[vi][test_i1:test_i2])
            in_r = np.append([0], np.array(
                (smin <= v[ref_inds]) & (v[ref_inds] <= smax), dtype=int))
            startj = np.where(np.diff(in_r) == 1)[0]
            endj = np.where(np.diff(in_r) == -1)[0]
            lenj = min(len(startj), len(endj))
            sei = np.where(endj[:lenj] - startj[:lenj] >=
                           test_len * small_ref_prop)[0]
            if len(sei) > 0:
                sei_ref = np.concatenate(
                    [np.array(v[ref_inds[startj[j]:endj[j]]]) for j in sei])
                refs[vi].append(sei_ref)
    return refs

def merge_segments_test(
    values: list | np.ndarray | pd.Series,
    segments: list | np.ndarray | pd.Series,
    merge_signif_test: Literal['t', 'wilcox'] = 'wilcox',
    p_thres: float = 0.05,
    ignore_no: int = 5,
    strict: bool = True
):# -> np.ndarray:
    """
    Merge segments that contain values that are not significantly different
    based on some significance test.
    Intended for internal use only; made public for external QA.

    Args:
        values (list | numpy.ndarray | pandas.Series): Vector of values.
        segments (list | numpy.ndarray | pandas.Series): Segment label vector.
        merge_signif_test (str, optional): Type of significance test to used
            i.e. 't' (T-test) or 'wilcox' (Wilcoxan). Defaults to 'wilcox'.
        p_thres (float, optional): P-value threshold. Defaults to 0.05.
        ignore_no (int, optional): See `in_percentiles()`. Defaults to 5.
        strict (bool, optional): If set to `True`, function will only test
            adjacent segments; otherwise, . Defaults to True.

    Returns:
        numpy.ndarray: an updated version of `segments` after merging.
    """
    segments = np.array(segments)
    if len(np.unique(segments)) < 2:
        return segments

    chpts0M = segments_to_chpts0M(segments)

    # determine which changepoints to keep based on significance test
    del_inds = [0]
    w_i = 0
    while len(del_inds) > 0 and len(chpts0M) > 0 and not (strict and w_i > 0):
        del_inds = []
        ps = []
        lengthi = np.diff(chpts0M)
        for i in range(1, len(chpts0M) - 1):
            x = np.array(values[chpts0M[i-1]:chpts0M[i]])
            x = x[~np.isnan(x)]
            y = np.array(values[chpts0M[i]:chpts0M[i+1]])
            y = y[~np.isnan(y)]
            if merge_signif_test == 'wilcox':
                p = ranksums(x, y).pvalue
            else:
                p = ttest_ind(x, y).pvalue
            # sometimes, small segments could be falsely p >= p_thres
            # so we check the segments beside it; even if the next segment is
            # also small, there's less chance of false positive?
            if (lengthi[i-1] <= ignore_no and len(y) > ignore_no and i > 1 and
                chpts0M[i-1] - chpts0M[i-2] > ignore_no and
                ps[-1] > p_thres and p > p_thres):
                x = np.array(values[chpts0M[i-2]:chpts0M[i-1]])
                x = x[~np.isnan(x)]
                if merge_signif_test == 'wilcox':
                    p_ = ranksums(x, y).pvalue
                else:
                    p_ = ttest_ind(x, y).pvalue
                if p_ < p_thres:
                    if ps[-1] < p:
                        ps[-1] = p_
                    else:
                        p = p_
            ps.append(p)
        w_i += 1
        del_inds = np.where(np.array(ps) >= p_thres)[0]
        if len(del_inds) > 0:
            chpts0M = np.delete(chpts0M, del_inds + 1)

    return chpts0M_to_segments(chpts0M)

def merge_segments_gain(
    values: list | np.ndarray | pd.Series,
    segments: list | np.ndarray,
    cost_model: str = 'rank',
    binseg_jump: int = 2,
    binseg_min_size: int = 2,
    knee: bool = False,
    min_ref_percent: float = 0.4
    # png_dir: str = ''
):# -> np.ndarray:
    """
    Merge segments with minimum gain at their changepoints until one segment is
    at least `min_ref_percent` percent of the maximum length.
    Intended for internal use only; made public for external QA.

    Args:
        values (list | numpy.ndarray | pandas.Series): Vector of values.
        segments (list | numpy.ndarray | pandas.Series): Segment label vector.
        cost_model (str, optional): Type of cost calculation method to use;
            see `ruptures.Binseg`. Defaults to 'rank'.
        binseg_jump (int, optional): See argument `jump` in `ruptures.Binseg`.
            Defaults to 2.
        binseg_min_size (int, optional): See argument `min_size` in
            `ruptures.Binseg`. Defaults to 2.
        knee (bool, optional): Use gain value knee as stopping criteria. For
            testing purposes only, do not change. Defaults to False.
        min_ref_percent (float, optional): Stopping criteria; indicates the
            minimum length (proportion of total) of any segment.
            Defaults to 0.4.

    Returns:
        numpy.ndarray: an updated version of `segments` after merging segments.
    """
    segments = np.array(segments)
    if len(np.unique(segments)) == 1:
        return segments

    chpts0M = segments_to_chpts0M(segments)
    if len(chpts0M) < (1 + (1 / min_ref_percent)):
        return segments

    # merge chpts until a segment is at least min_ref_percent long
    rpt_class = rpt.Binseg(
        model=cost_model, jump=binseg_jump, min_size=binseg_min_size)
    rpt_class.fit(signal=np.array(values))
    gains = list_gains(rpt_class=rpt_class, chpts0M=chpts0M)
    thres_met = any(np.diff(chpts0M) / len(values) > min_ref_percent)
    # use gain value knee (i.e. elbow) as the merge limit if specified.
    if knee: # TODO 10
        kneedle = KneeLocator(
            np.array(range(2, len(gains))),
            np.sort(gains[1:-1]),
            S=1.0, curve="convex", direction="increasing")
        if not (kneedle.elbow is None) and not (kneedle.knee is None):
            keep_ind = np.argsort(gains)[min(kneedle.elbow, kneedle.knee):]
            if len(keep_ind) > 0:
                keep_ind = np.sort(keep_ind)
                chpts0M = np.concatenate(
                    ([0], chpts0M[keep_ind], [chpts0M[-1]]))
        else:
            knee = False
    if not knee: # TODO: mean variance must be within something...
        thres_met = any(np.diff(chpts0M) / len(values) > min_ref_percent)
        while len(chpts0M) > 2 and not thres_met:
            # delete chpt with lowest gain
            del_ind = np.argmin(gains[1:-1]) + 1
            chpts0M = np.delete(chpts0M, del_ind)
            gains = np.delete(gains, del_ind)
            thres_met = any(np.diff(chpts0M) / len(values) > min_ref_percent)

            # recalculate gain
            if len(chpts0M) > 2:
                gains_new = list_gains(
                    chpts0M=chpts0M, chpts_inds=[del_ind-1, del_ind],
                    rpt_class=rpt_class)
                gains[gains_new != -1] = gains_new[gains_new != -1]

    segments = chpts0M_to_segments(chpts0M)
    return np.array(segments)

def merge_segments_quant(
    vlist: list,
    segments: list | np.ndarray | pd.Series,
    ip_func: Callable = in_percentiles,
    get_ref_fun: Callable = get_ref_updown,
    small_no_per_seg: int = 50,
    keep_skipped: str = 'all'
    # png_dir: str = ''
):# -> np.ndarray:
    """
    Merge segments if their quantiles are within a range of each other.
    Intended for internal use only; made public for external QA.

    Args:
        vlist (list): List of value vectors.
        segments (list | numpy.ndarray | pandas.Series): Segment label vector.
        ip_func (Callable, optional): Function used to determine whether test
            and reference values are similar. Defaults to `in_percentiles()`.
        get_ref_fun (Callable, optional): A function that extracts a list of
            reference values. Defaults to `get_ref_updown()`.
        small_no_per_seg (int, optional): Vectors with a smaller length than
            this value is considered a "small" vector;
            small vectors are treated differently because
            smaller segments can contain more sporadic values. Defaults to 50.
        keep_skipped (str, optional): Indicates whether to keep all, some, or no
            skipped segments i.e. 'all', 'some', 'none'. Defaults to 'all'.

    Returns:
        numpy.ndarray: an updated version of `segments` after merging segments.
    """
    segments = np.array(segments)
    if len(np.unique(segments)) == 1:
        return segments

    # prepare arguments
    chpts0M = segments_to_chpts0M(segments=segments)
    if len(chpts0M) < 3:
        return segments

    # check adjacent segments against longest (reference) segments
    max_list = np.array([], dtype=int)
    skipped = np.array([], dtype=int)
    # plot_i = 0
    primary_twice = False
    while ((len(chpts0M) > 0) and
           (not all(np.isin(chpts0M[:-1], max_list)) or not primary_twice)):
        r_list = np.argsort(np.diff(chpts0M))
        in_max = np.isin(r_list, np.where(np.isin(chpts0M[:-1], max_list)))
        r = mode(chpts0M_to_segments(chpts0M))
        if all(in_max):
            if primary_twice:
                break
            else:
                primary_twice = True
        # check if segment is large enough to become a reference segment
        elif not primary_twice:
            r = r_list[~in_max][-1]
        if chpts0M[r+1] - chpts0M[r] <= small_no_per_seg:
            max_list = np.append(max_list, chpts0M[r])
            continue
        # set upper/lower bounds for reference segments: half, %, all
        ref_values = get_ref_fun(r=r, chpts0M=chpts0M, vlist=vlist)
        # set up/lower limits for segments to check
        lims = [-1, len(chpts0M) - 1]
        if len(max_list) > 0:
            up_max_ = np.where(np.isin(chpts0M[r:-1], max_list))[0]
            down_min_ = np.where(np.isin(chpts0M[:r], max_list))[-1]
            if len(up_max_) > 0:
                lims[1] = r + up_max_[0]
            if len(down_min_) > 0:
                lims[0] = down_min_[0]
        gos = [r - 1 > lims[0], r + 1 < lims[1]]
        # plot_j = 0
        # merge warrented segments
        while any(gos):
            for i in [1, 0]:
                if gos[i]:
                    # explicitly assign variables for testing
                    gos[i], skipped_ = go_updown(
                        r=r, chpts0M=chpts0M,
                        lim=lims[i], refs=ref_values[i],
                        vlist=vlist, ip_func=ip_func)
                if gos[i]:
                    chpts0M = np.delete(chpts0M, r+1 if i == 1 else r)
                    r = r - 1 if i == 0 else r
                    skipped = np.append(skipped, skipped_)
            if np.any(gos):
                ref_values = get_ref_fun(r=r, chpts0M=chpts0M, vlist=vlist)
                lims[1] -= np.sum(gos)
                gos = [r - 1 > lims[0] and gos[0], r + 1 < lims[1] and gos[1]]

            # # plot combined segments
            # if png_dir:
            #     plot_scat(
            #         np.array(range(len(vlist[0]))),
            #         np.sum(np.stack(vlist), axis=0),
            #         L=chpts0M_to_segments(chpts0M),
            #         plot_title='percentile combined segments SORTED',
            #         out_path='{}/2.while_{}.{}_SORTED.png'.format(
            #             png_dir, plot_i, plot_j))
            # plot_j += 1

        # plot_i += 1
        max_list = np.append(max_list, chpts0M[r])

    # keep or removed skipped segments
    r = np.argmax(np.diff(chpts0M))
    segments = chpts0M_to_segments(chpts0M)
    skipped_chpt = np.array(np.unique(skipped), dtype=int)
    skipped_chpt = skipped_chpt[
        (skipped_chpt >= chpts0M[r]) & (skipped_chpt < chpts0M[r+1])]
    if keep_skipped != 'all':
        for i, ch in enumerate(skipped_chpt[0::2]):
            segments[ch:skipped_chpt[i+1]] = -1

    return segments

def refine_reference_segment(
    vlist: list,
    segments: list | np.ndarray,
    rl: int | None = None,
    sorted_index: np.ndarray | None = None,
    ip_func: Callable = in_percentiles,
    small_no_per_seg: int = 50,
    overlap_no: int = 0,
    split_by_segment: bool = True,
    multi_refine: bool = True
):# -> np.ndarray:
    """
    Check if there are sporadic segments within the reference segments that
    should be merged with the reference segment.
    Intended for internal use only; made public for external QA.

    Args:
        vlist (list): List of value vectors.
        segments (list | numpy.ndarray | pandas.Series): Segment label vector.
        rl (int | None, optional): Reference segment label; if set to `None`,
            sets the reference segment as the largest segment. Defaults to None.
        sorted_index (numpy.ndarray | None, optional): Indices sorted based on
            segment quantiles; see `utils.order_value_bins()`. Defaults to None.
        ip_func (Callable, optional): Function used to determine whether test
            and reference values are similar. Defaults to `in_percentiles()`.
        small_no_per_seg (int, optional): Vectors with a smaller length than
            this value is considered a "small" vector;
            small vectors are treated differently because
            smaller segments can contain more sporadic values. Defaults to 50.
        overlap_no (int, optional): Numer of values allowed to overlap.
            Defaults to 0.
        split_by_segment (bool, optional): If there are multiple segments
            in between the reference segment, choose whether to test them
            separately. Defaults to True.
        multi_refine (bool, optional): Optional, if set to True, function will
            refine repeatedly. Defaults to True.

    Returns:
        numpy.ndarray: an updated version of `segments` after merging segments.
    """
    segments = np.array(segments)
    if len(np.unique(segments)) == 1:
        return segments

    # start: merge segments into reference if similar enough
    rl = _check_chpts0M_segments(segments=segments, rl=rl)
    any_removed = True
    while any_removed:
        # get indices of segments in between reference segment
        ri = np.where(segments == rl)[0]
        chpts0M = segments_to_chpts0M(segments)
        ri_diff = np.diff(ri)
        starti = ri[np.where(np.append(ri_diff, [0]) > 1)[0]] + 1
        if len(starti) == 0 and ri[0] == 0 and ri[-1] == (len(segments)-1):
            break
        lengthi = ri_diff[ri_diff > 1]
        endi = starti + lengthi - 1
        if ri[0] > 0:
            endi = np.append(ri[0], endi)
            start_chpts0M = chpts0M[np.where(chpts0M == ri[0])[0] - 1]
            starti = np.append(start_chpts0M, starti)
        if ri[-1] < (len(segments)-1):
            starti = np.append(starti, ri[-1] + 1)
            end_chpts0M = chpts0M[np.where(chpts0M == ri[-1] + 1)[0] + 1]
            endi = np.append(endi, end_chpts0M)

        # for each segment inside the reference segment...
        any_removed = False
        for i in range(len(starti)):
            chs = np.unique(segments[starti[i]:endi[i]])
            to_keep = len(chs) == 1 or endi[i] - starti[i] <= small_no_per_seg
            # there is only one segment or if the inbetween segment is small,
            # test normally.
            if to_keep:
                refs = get_ref_func2(
                    vlist=vlist, ref_inds=ri,
                    test_i1=starti[i], test_i2=endi[i],
                    small_no_per_seg=small_no_per_seg,
                    sorted_index=sorted_index)
                starti_ = max(0, starti[i] - overlap_no)
                endi_ = min(len(segments), endi[i] + overlap_no)
                tests = [v[starti_:endi_] for v in vlist]
                for test_value, ref_values in zip(tests, refs):
                    to_keep_ = ip_func(
                        ref_values=ref_values, test_value=test_value,
                        check_mean=endi[i] == ri[0] or ri[-1] < starti[i])
                    to_keep = to_keep and to_keep_
                if to_keep:
                    segments[starti[i]:endi[i]] = rl
            elif len(chs) > 1:
                # test each segment inbetween or combine them for testing
                if split_by_segment:
                    ch_inds = [starti[i] + np.where(
                        segments[starti[i]:endi[i]] == c)[0] for c in chs]
                else:
                    chs = np.where(np.diff(segments[starti[i]:endi[i]])!=0)[0]+1
                    chs = np.concatenate(([0], chs, [endi[i] - starti[i]]))
                    ch_inds = [starti[i] + np.array(range(chs[ci], chs[ci+1]))
                                for ci in range(len(chs) - 1)]
                for ci in ch_inds:
                    if ci[0] > 0 and overlap_no > 0:
                        ci = np.concatenate((np.array(range(
                            ci[0] - overlap_no, ci[0]), dtype=int), ci))
                    if ci[-1] < len(segments) and overlap_no > 0:
                        ci = np.concatenate((ci, np.array(range(
                            ci[-1] + 1, ci[-1] + overlap_no + 1), dtype=int)))
                    # TODO: come back to this, maybe not change ref inds?
                    refs = get_ref_func2(
                        vlist=vlist, ref_inds=ri,
                        test_i1=ci[0], test_i2=ci[-1],
                        small_no_per_seg=small_no_per_seg,
                        sorted_index=sorted_index)
                    tests = [v[ci] for v in vlist]
                    to_keep = True
                    for test_value, ref_values in zip(tests, refs):
                        to_keep_ = ip_func(
                            ref_values=ref_values, test_value=test_value,
                            lenient=False)
                        to_keep = to_keep and to_keep_
                    if to_keep:
                        segments[ci] = rl

            if not any_removed and to_keep and multi_refine:
                any_removed = True

    return np.array(segments)

def refine_reference_segment_inbetween(
    segments: list | np.ndarray,
    rl: int | None = None,
    small_no_per_seg: int = 50
):# -> np.ndarray:
    """
    Same as `refine_reference_segment()` but intended for when there are
    sporadic parts of the reference segment between other segments.
    Intended for internal use only; made public for external QA.

    Args:
        segments (list | numpy.ndarray | pandas.Series): Segment label vector.
        rl (int | None, optional): Reference segment label; if set to `None`,
            sets the reference segment as the largest segment. Defaults to None.
        small_no_per_seg (int, optional): Vectors with a smaller length than
            this value is considered a "small" vector;
            small vectors are treated differently because
            smaller segments can contain more sporadic values. Defaults to 50.

    Returns:
        numpy.ndarray: an updated version of `segments` after merging segments.
    """
    segments = np.array(segments)
    if len(np.unique(segments)) == 1:
        return segments

    rl = _check_chpts0M_segments(segments=segments, rl=rl)

    # get indices of reference segments in between other segments
    chpts0M = segments_to_chpts0M(segments)
    ref_inds = np.where(segments == rl)[0]
    ref_inds = np.concatenate(([-1], ref_inds, [len(segments)]))
    ri_diff = np.diff(ref_inds)
    endi = ref_inds[np.where(np.append(ri_diff, [0]) > 1)[0]] + 1
    if len(endi) == 0:
        return np.array(segments)
    starti = chpts0M[np.where(np.isin(chpts0M, endi))[0] - 1]
    lengthi = endi - starti
    chi = np.where(lengthi <= small_no_per_seg)[0]
    if len(chi) == 0:
        return segments

    # TODO: temporary
    for i in chi:
        around_ref = np.concatenate((
            segments[max(0, starti[i] - small_no_per_seg):starti[i]],
            segments[endi[i]:min(chpts0M[-1], endi[i] + small_no_per_seg)]))
        if all(around_ref != rl):
            segments[starti[i]:endi[i]] = -1

    return np.array(segments)

def remove_reference_ends(
    values: list | np.ndarray | pd.Series,
    segments: list | np.ndarray | pd.Series,
    rl: int | None = None,
    ip_func: Callable = in_percentiles,
    small_no_per_seg: int = 50
):# -> np.ndarray:
    """
    Check ends of the reference segment to see if there are any sporacidity.
    Merge ends if applicable.
    Intended for internal use only; made public for external QA.

    Args:
        values (list | numpy.ndarray | pandas.Series): Vector of values.
        segments (list | numpy.ndarray | pandas.Series): Segment label vector.
        rl (int | None, optional): Reference segment label; if set to `None`,
            sets the reference segment as the largest segment. Defaults to None.
        ip_func (Callable, optional): Function used to determine whether test
            and reference values are similar. Defaults to `in_percentiles()`.
        small_no_per_seg (int, optional): Vectors with a smaller length than
            this value is considered a "small" vector;
            small vectors are treated differently because
            smaller segments can contain more sporadic values. Defaults to 50.

    Returns:
        numpy.ndarray: an updated version of `segments` after merging segments.
    """
    segments = np.array(segments)
    if len(np.unique(segments)) == 1:
        return np.array(segments)

    rl = _check_chpts0M_segments(segments=segments, rl=rl)
    ri = segments == rl
    ref_inds = np.where(ri)[0]
    ri_diff = np.diff(ref_inds)
    starti = ref_inds[np.where(np.append(ri_diff, [0]) > 1)[0]] + 1
    if len(starti) == 0:
        return segments
    lengthi = ri_diff[ri_diff > 1]
    endi = starti + lengthi - 1
    # check if there are inserts into the reference segment at ends
    cond_ = [starti[0] <= small_no_per_seg,
                endi[-1] >= (len(values) - small_no_per_seg - 1)]
    refs_ = [values[starti[0]:endi[0]], values[starti[-1]:endi[-1]]]
    tests_ = [values[np.where(ri[:starti[0]])[0]],
                values[endi[-1] + np.where(ri[endi[-1]:])[0]]]
    inds_ = [np.array(range(starti[0])),
                np.array(range(endi[-1], len(segments)))]
    for cond, refs, tests, inds in zip(cond_, refs_, tests_, inds_):
        if cond:
            rm_values = ip_func(ref_values=[refs], test_value=tests)
            if rm_values:
                segments[inds] = -1

    return segments




