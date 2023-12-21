from functools import partial
from typing import Type
import time
import copy
import warnings
import pandas as pd
import numpy as np
import ruptures as rpt
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score, pairwise_distances

from metaclean3.utils import mode, order_value_bins, randomize_duplicates, arg_lim
from metaclean3.plot import plot_scat
from metaclean3.segments import (
    in_percentiles,
    go_updown,
    get_ref_updown,
    remove_reference_ends,
    merge_segments_test,
    merge_segments_gain,
    merge_segments_quant,
    refine_reference_segment,
    refine_reference_segment_inbetween
)
from metaclean3.changepoint import chpts0M_to_segments, segments_to_chpts0M, list_gains
from metaclean3.outliers import IsoForestDetector, drop_outliers
from metaclean3.features import get_bin_density, get_bin_moments, scale_values
from metaclean3.fcs import FCSfile
from metaclean3.channels import most_corr_channels

class MetaClean:
    """Cleans a given data set.
    """
    def __init__(
        self,
        # for rupture: changepoint
        seg_method: str = 'pelt',
        cost_model_seg: str = 'rbf',
        jump_per_seg: int = 2,
        min_seg_size: int = 2,
        # for binseg
        seg_no: int = 40, # over-estimate segments please > 0
        mean_no_per_seg: int = 50,
        min_no_per_seg_limit: int = 10,
        # for pelt
        pelt_penalty: float = 5.0,
        # for combining segments (significance test) / determining best segment
        merge_method: str = 'sequential',
        merge_signif_test: str = 'wilcox', # wilcoxan, ...
        p_thres: float = 0.05, # > 0, < 1
        signif_strict: bool = True,
        # for combining segments (gain)
        min_ref_percent: float = 0.5,
        cost_model_gain: str = 'rank',
        gain_diff: bool = True, # ratio if false
        # for combining segments (quantile ranges)
        percent_diff: float = 0.05,
        percent_shifts: float = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4],
        small_no_per_seg0: int = 50,
        ignore_no: int = 10,
        ref_percents: list = [1, 0.5],
        ref_percent: float = 0.1,
        exception_no: int = 2,
        keep_skipped: str = 'all',
        # be lenient when ordering segments
        pd_lenient: int = 2,
        # if there are non-reference segments that are large, keep it
        min_ref_percent_to_keep: float = 0.4,
        # output options
        random_seed: int = 623,
        verbose: bool = True,
        png_dir: str = ''
    ):
        """
        Args:
            seg_method (str, optional): Changepoint detection method; currently
                support options 'pelt' and 'binseg'. Defaults to 'pelt'.
            cost_model_seg (str, optional): See cost models in `ruptures.costs`.
                Used for finding changepoints. Defaults to 'rbf'.
            jump_per_seg (int, optional): See `jump` argument in
                `ruptures.Binseg`. Defaults to 2.
            min_seg_size (int, optional): See `min_size` argument in
                `ruptures.Binseg`. Defaults to 2.
            seg_no (int, optional): Number of segments to find; see `n_bkps`
                argument in `ruptures.base` (used only if `seg_method`
                is set to `binseg`). Defaults to 40.
            min_no_per_seg_limit (int, optional): The maximum length of a
                "small" segment; small segments are treated more leniently when
                deciding whether or not it should be merged with adjacent
                segments. Defaults to 10.
            pelt_penalty (float, optional): See `pen` argument in
                `ruptures.KernelCPD`. Defaults to 5.0.
            merge_method (str, optional): 'sequential' or 'sd'. 'sequential'
                merges adjacent segments that are not significantly different.
                'sd' removes changepoints that have a gain within
                3 standard deviations. Defaults to 'sequential'.
            merge_signif_test (str, optional): 'wilcox' or 't'.
                Significance test to use for testing and merging segments.
                Defaults to 'wilcox'.
            min_ref_percent (float, optional): Stopping criteria; indicates the
                minimum length (proportion of total) of any segment.
                Defaults to 0.5.
            cost_model_gain (str, optional): See cost models in `ruptures.costs`.
                Used for calculating changepoint gain. Defaults to 'rank'.
            gain_diff (bool, optional): Wether to use the difference (`True`)
                or the ratio (`False`) to calculate gain change.
                Defaults to True.
            percent_shifts (list, optional): Quantile ranges used to test
                two segments' quantile range.
                Defaults to [0.15, 0.2, 0.25, 0.3, 0.35, 0.4].
            small_no_per_seg0 (int, optional): An upper limit for and used to
                refine `min_no_per_seg_limit`. Defaults to 50.
            ignore_no (int, optional): The maximum length of a segment that has
                negligible length. Defaults to 10.
            ref_percents (list, optional): The size of reference segments to
                test (a proportion of the reference segment length).
                Defaults to [1, 0.5].
            ref_percent (float, optional): The size of reference segments to
                test (a proportion of the whole length). Defaults to 0.1.
            exception_no (int, optional): Users can choose to check test values
                `exception_no` segments away from reference segment
                (i.e. non-adjacent test segments). Defaults to 2.
            keep_skipped (str, optional): Remove 'all', 'some', or 'none'
                skipped segments. Defaults to 'all'.
            min_ref_percent_to_keep (float, optional): If there are
                non-reference segments larger than this percentage, we keep it.
                Defaults to 0.4.
            pd_lenient (int, optional): How many times as lenient should
                the function be when ordering (not merging) segments
                based on their geometric mean and quantiles. Defaults to 2.
            random_seed (int, optional): Random seed. Defaults to 623.
            verbose (bool, optional): Whether to display progress messages.
                Defaults to True.
            png_dir (str, optional): .png path of where users want to save
                plots; no plot will be made if set to ''. Defaults to ''.
        """
        # segmentation arguments
        # for binseg
        self.seg_no = seg_no # over-estimate segments please > 0
        self.mean_no_per_seg = mean_no_per_seg
        if seg_method not in ['pelt', 'binseg']: # Literal['binseg', 'bayes', 'kmeans2']
            raise ValueError('seg_method chosen is not available.')
        self.seg_method = seg_method

        # for binseg
        bms = [
            'rbf', 'normal', 'linear', 'l1', 'l2', 'ar', 'mi', 'rank', 'cosine']
        if cost_model_seg not in bms:
            raise ValueError('cost_model_seg chosen is not available.')
        if cost_model_gain not in bms:
            raise ValueError('cost_model_gain chosen is not available.')
        self.cost_model_seg = cost_model_seg
        self.cost_model_gain = cost_model_gain
        self.jump_per_seg = arg_lim(jump_per_seg, xmin=1)
        min_no_per_seg_limit = min(
            min_no_per_seg_limit, np.round(small_no_per_seg0 / 2))
        # HARDCODE because segment merging WILL error if less than 5
        self.min_no_per_seg_limit = arg_lim(min_no_per_seg_limit, xmin=5)
        self.min_seg_size = arg_lim(min_seg_size, xmin=1)
        self.pelt_penalty = pelt_penalty
        self.gain_diff = gain_diff # ratio if false

        # combine segment / determine best segment

        if merge_method not in ['sequential', 'sequential']: # Literal['sequential', 'sd']
            raise ValueError('{}{}'.format(
                'merge_method chosen is not available, ',
                'please choose from \'sequential\' or \'sequential\''))
        self.merge_method = merge_method
        if merge_signif_test not in ['t', 'cohen', 'wilcox']: # Literal['t', 'cohen']
            raise ValueError('{}{}'.format(
                'merge_signif_test chosen is not available, ',
                'please choose from \'t\', \'cohen\', or \'wilcox\''))
        self.merge_signif_test = merge_signif_test
        self.p_thres = arg_lim(p_thres, xmin=0, xmax=1)
        self.signif_strict = signif_strict

        if keep_skipped not in ['all', 'some', 'none']: # Literal['all', 'some', 'none']
            raise ValueError('{}{}'.format(
                'merge_signif_test chosen is not available, ',
                'please choose from \'all\', \'some\', or \'none\''))
        self.keep_skipped = keep_skipped
        self.percent_diff = arg_lim(percent_diff, xmin=0, xmax=1)
        self.ignore_no = arg_lim(ignore_no, xmin=0)
        self.percent_shifts = [
            arg_lim(p, xmin=0, xmax=1) for p in percent_shifts]
        self.in_percentiles_ind = 0 # which percent_shifts to use
        self.exception_no = arg_lim(exception_no, xmin=0)

        self.min_ref_percent = arg_lim(min_ref_percent, xmin=0, xmax=1)
        self.small_no_per_seg0 = arg_lim(small_no_per_seg0, xmin=0, xmax=1)
        self.small_no_per_seg = arg_lim(small_no_per_seg0, xmin=0, xmax=1)
        self.ref_percents = [arg_lim(r, xmin=0, xmax=1) for r in ref_percents]
        self.ref_percent = arg_lim(ref_percent, xmin=0, xmax=1)

        # functions
        self.in_percentiles = None
        self.go_updown = None
        self.get_ref_fun = None

        # generic arguments
        self.pd_lenient = arg_lim(pd_lenient, xmin=1)
        self.min_ref_percent_to_keep = arg_lim(
            min_ref_percent_to_keep, xmin=0, xmax=1)
        self.random_seed = random_seed
        self.verbose = verbose
        self.png_dir = png_dir

    # TODO: make methods private after testing
    def clean(
        self,
        data: pd.DataFrame,
        val_cols: list | None = None,
        val_col_final: str | None = None,
    ):# -> tuple:
        """Conduct cleaning of `data` (i.e. remove irregular objects/rows).

        Args:
            data (pandas.DataFrame): An object x feature matrix.
            val_cols (list | None): List of feature column names in `data` to be
                used for cleaning. Defaults to None.
            val_col_final (str | None): A summary feature column name in `data`
                where (e.g. sum of `val_cols` values). `val_cols` and
                `val_col_final` can be overlapping. Defaults to None.

        Returns:
            tuple:
                numpy.ndarray: A 1D boolean array with `True`/`False` labels
                    indicating which rows in `data` to keep/remove.
                numpy.ndarray: A 1D integer array containing refined segment
                    labels.
                numpy.ndarray: A 1D integer array containing raw segment labels.
        """
        start0 = time.time()
        # specify columns to use
        if val_cols is None:
            val_cols = data.columns
        if val_col_final is None:
            if len(val_cols) == 1:
                val_col_final = val_cols[0]
            else:
                val_col_final = '_'.join(val_cols)
                data[val_col_final] = data[val_cols].sum(axis=1)
        # process parameters
        self.small_no_per_seg = min(
            self.small_no_per_seg0,
            int(len(data) / self.min_no_per_seg_limit)
        )
        # prepare functions
        self.in_percentiles_multi = [partial(
            in_percentiles,
            ignore_no=self.ignore_no,
            percent_diff=self.percent_diff,
            percent_shift=p,
            small_no_per_seg=self.small_no_per_seg
        ) for p in self.percent_shifts]
        self.go_updown = partial(
            go_updown,
            exception_no=self.exception_no,
            ignore_no=self.ignore_no
        )
        self.get_ref_fun = partial(
            get_ref_updown,
            ref_percents=self.ref_percents,
            ref_percent=self.ref_percent
        )

        # get initial segments
        start = time.time()
        values = np.array(data[val_col_final])
        vinds = np.array(range(len(values)))
        segments0 = self.get_segments(values=values)
        if self.png_dir:
            op = '{}/0.segments.png'.format(self.png_dir)
            plot_scat(
                x=vinds, y=values, L=segments0,
                plot_title='Raw segments', out_path=op
            )
        if self.verbose:
            print(' + Time to identify segments: {}\'s'.format(
                np.round(time.time() - start, 3)))

        # segments_merged, sorted_index = self.merge_segments(
        start = time.time()
        segments1, _ = self.merge_segments(
            values=values, vlist=data[val_cols], segments=segments0)
        # merge skipped segments into reference
        segments2 = self.refine_segments(
            values=values, segments=segments1, ref_segment=mode(segments1))
        if self.png_dir:
            plot_scat(
                x=vinds, y=values, L=segments2,
                plot_title='Refined segments (Done! Keep the largest segment)',
                out_path='{}/3.refined.png'.format(self.png_dir)
            )
        if self.verbose:
            print(' + Time to merge segments: {}\'s'.format(
                np.round(time.time() - start, 3)))
            print('- Total time used: {}\'s'.format(
                np.round(time.time() - start0, 3)))

        return segments2 == mode(segments2), segments2, segments0

    def get_segments(
        self,
        values: list | np.ndarray | pd.Series,
    ):# -> np.ndarray:
        """
        Given an array, break the array into segments (find changepoints) and
        merge those segments based on significance tests to obtain the initial
        "raw" segments.

        Args:
            values (list | numpy.ndarray | pandas.Series): A 1D array of numeric
                values.

        Returns:
            numpy.ndarray: A 1D integer array containing "raw" segment labels.
        """
        # set up parameters (min 2 points per segment)
        self.seg_no = min(
            max(self.seg_no, 1), int(len(values) / self.min_seg_size))
        if self.seg_no < 2:
            raise ValueError('Given data has < 2 data points')

        if self.seg_method == 'binseg':
            rpt_class = rpt.Binseg(
                model=self.cost_model_seg,
                jump=self.jump_per_seg,
                min_size=self.min_seg_size
            )
            chpts = np.array(
                rpt_class.fit_predict(signal=values, n_bkps=self.seg_no))[:-1]
        elif self.seg_method == 'pelt':
            rpt_class = rpt.KernelCPD(
                kernel=self.cost_model_seg, min_size=self.min_seg_size)
            chpts = np.array(rpt_class.fit_predict(
                signal=values, pen=self.pelt_penalty))[:-1]
        chpts0M = np.concatenate(([0], chpts, [len(values)]))
        rpt_class = rpt.Binseg(
            model=self.cost_model_gain,
            jump=self.jump_per_seg,
            min_size=self.min_seg_size
        ).fit(signal=np.array(values))
        gains = list_gains(
            rpt_class=rpt_class, chpts0M=chpts0M, dif=self.gain_diff)[1:-1]

        return self.merge_segments_raw(values=values, chpts=chpts, gains=gains)

    def merge_segments_raw(
        self,
        values: list | np.ndarray | pd.Series,
        chpts: list | np.ndarray | pd.Series,
        gains: list | np.ndarray | pd.Series
    ):# -> np.ndarray:
        """Merge raw segments using statistical significance tests.

        Args:
            values (list | numpy.ndarray | pandas.Series): A 1D array of numeric
                values.
            chpts (list | numpy.ndarray | pandas.Series): A 1D array of
                changepoints not including 0 and length of `values` as the
                first and last elements.
            gains (list | numpy.ndarray | pandas.Series): List of gains
                corresponding to `chpts`.

        Returns:
            numpy.ndarray: A 1D integer array containing "raw" segment labels.
        """
        if len(chpts) == 0 or (self.merge_method == 'sd' and len(chpts) < 2):
            return np.full((len(values)), 0)

        # merge changepoints
        chpts0M = np.concatenate(([0], chpts, [len(values)]))
        if self.merge_method == 'sd':
            chpts = chpts[np.where(gains > np.mean(gains) + 3 * np.std(gains))]
            chpts0M = np.concatenate(([0], chpts, [len(values)]))
            segments = chpts0M_to_segments(chpts0M)
        elif self.merge_method == 'sequential':
            segments = merge_segments_test(
                values=values,
                segments=chpts0M_to_segments(chpts0M),
                merge_signif_test=self.merge_signif_test,
                p_thres=self.p_thres,
                ignore_no=self.ignore_no,
                strict=self.signif_strict
            )
        # segments = chpts0M_to_segments(np.concatenate(([0], chpts0M[1:-1]+1, [len(values)])))
        return segments

    def merge_segments(
        self,
        values: list | np.ndarray | pd.Series,
        vlist: list,
        segments: list | np.ndarray | pd.Series
    ):# -> tuple:
        """Merge segments based on changepoint gain and segment quantile ranges.

        Args:
            values (list | numpy.ndarray | pandas.Series): A 1D array of numeric
                values.
            vlist (list): A list of 1D arrays with the same length as `values`.
            segments (list | numpy.ndarray | pandas.Series): Segment label
                vector.

        Returns:
            tuple:
                numpy.ndarray: sorted 1D integer segment label array.
                numpy.ndarray: sorted indices.

        """
        # sort segments
        sorted_index = order_value_bins(
            vlist=copy.copy(vlist),
            segments=segments,
            percentile_diff=self.percent_diff * self.pd_lenient
        )

        # merge sorted segments based on gain
        values_sorted = np.array(values)[sorted_index]
        segmentss1 = merge_segments_gain(
            values=values_sorted,
            segments=segments[sorted_index],
            cost_model=self.cost_model_gain,
            binseg_jump=self.jump_per_seg,
            binseg_min_size=self.min_seg_size,
            min_ref_percent=self.min_ref_percent
        )
        if self.png_dir:
            op = '{}/1.merge_segments_gain_SORTED.png'.format(self.png_dir)
            plot_scat(
                x=np.array(range(len(values))), y=values_sorted, L=segmentss1,
                plot_title='Combined segments (gain) SORTED', out_path=op
            )

        # merge sorted segents based on quantiles (best parameter = max sil)
        vlist_sorted = [vlist[v][sorted_index] for v in vlist.columns]
        segmentss2s = [merge_segments_quant(
            vlist=vlist_sorted,
            segments=segmentss1,
            ip_func=partial(
                ip, check_var=False, check_mean=False, switch=True),
            get_ref_fun=self.get_ref_fun,
            small_no_per_seg=self.small_no_per_seg,
            keep_skipped=self.keep_skipped
        ) for ip in self.in_percentiles_multi]
        good_ip = np.full((len(segmentss2s)), True)
        if len(segmentss2s) > 1:
            for i in range(1, len(segmentss2s)):
                if np.array_equal(segmentss2s[i-1], segmentss2s[i]):
                    good_ip[i] = False
        if np.sum(good_ip) > 1:
            good_ip_ind = np.squeeze(np.argwhere(good_ip))
            x = np.expand_dims(values_sorted, axis=1)
            d = pairwise_distances(x)
            sil = [silhouette_score(d, segmentss2s[gip]) if
                   len(np.unique(segmentss2s[gip])) > 1 else 0
                   for gip in good_ip_ind]
            self.in_percentiles_ind = good_ip_ind[np.argmax(sil)]
        segmentss2 = segmentss2s[self.in_percentiles_ind]

        # merge skipped sorted segments into reference
        if self.keep_skipped == 'some' and -1 in segmentss2:
            segmentss2 = refine_reference_segment(
            vlist=[values],
                segments=segmentss2,
                rl=mode(segmentss2),
                sorted_index=sorted_index,
                ip_func=partial(
                    self.in_percentiles_multi[self.in_percentiles_ind],
                    check_mean=False, check_var=True, switch=False),
                small_no_per_seg=self.small_no_per_seg
            )
        if self.png_dir:
            op = '{}/2.merge_segments_quantiles_SORTED.png'.format(self.png_dir)
            plot_scat(
                x=np.array(range(len(values))), y=values_sorted, L=segmentss2,
                plot_title='Combined segments (quantiles) SORTED', out_path=op
            )

        return segmentss2[np.argsort(sorted_index)], sorted_index

    def refine_segments(
        self,
        # vlist: list,
        values: list | np.ndarray | pd.Series,
        segments: list | np.ndarray | pd.Series,
        ref_segment: int
        # sorted_index: list | np.ndarray | None = None
    ):# -> np.ndarray:
        """Refine segment to be kept.

        Args:
            values (list | numpy.ndarray | pandas.Series): A 1D array of numeric
                values.
            segments (list | numpy.ndarray | pandas.Series): Segment label
                vector.
            ref_segment (int): Label of the reference segment.

        Returns:
            numpy.ndarray: refined Segment label vector.
        """
        # segments1 = refine_reference_segment(
        #     vlist=vlist,
        #     segments=copy.copy(segments),
        #     rl=ref_segment,
        #     sorted_index=sorted_index,
        #     ip_func=partial(
        #         self.in_percentiles_multi[self.in_percentiles_ind],
        #         check_var=True, switch=False),
        #     small_no_per_seg=self.small_no_per_seg
        # )
        segments1 = refine_reference_segment_inbetween(
            segments=copy.copy(segments),
            rl=ref_segment,
            small_no_per_seg=self.small_no_per_seg
        )

        # identify large segments
        chpts0M = segments_to_chpts0M(segments1)
        segment_len = np.diff(chpts0M) / len(segments1)
        large_seg = segment_len >= self.min_ref_percent_to_keep
        # identify original reference segment
        ref_chpt = np.squeeze(np.argwhere(segments1 == mode(segments1)))[0]
        large_seg[chpts0M[:-1]==ref_chpt] = True
        # combine the above to create ne reference segment
        large_seg = np.squeeze(np.argwhere(large_seg))
        rl = np.max(segments1) + 1
        if len(large_seg.shape) > 0:
            for i in large_seg:
                segments1[chpts0M[i]:chpts0M[i+1]] = rl

        # remove beginning and ends if applicable
        segments2 = remove_reference_ends(
            values=values,
            segments=copy.copy(segments1),
            rl=mode(segments1),
            ip_func=partial(
                self.in_percentiles_multi[self.in_percentiles_ind],
                check_mean=True, check_var=False, switch=False),
            small_no_per_seg=self.small_no_per_seg * 2
        )

        return segments2

class MetaCleanFCS():
    """
    Given a compensated (flow cytometry) / unmixed (spectral cytometry) and
    transformed cytometry data, prepares the data for cleaning via 0.
    """
    def __init__(
        self,
        # identify best channels for cleaning
        clean_chans: list | np.ndarray | pd.Series | None = None,
        clean_chans_no: int = 4,
        candidate_chans_type: str = 'fluo',
        # corr_method: str = 'spearman',
        corr_type: str = 'max',
        # remove duplicates by row (slow) or by row sums (fast)
        strict_remove_duplicates: bool = False,
        # calculate density
        n_cores: int = -1,
        dens_agg_type: str = 'max', # 'max', 'median'
        dens_k_dtm: int = 15,
        p: int = 2,
        eps: float = 0.1,
        # identify outliers
        outlier_thresh: float = 0.01,
        outlier_trees: int = 500,
        outlier_drop_and: bool = True, # else OR
        rm_outliers: str = 'all',
        # others
        random_seed: int = 623,
        verbose: bool = True,
        png_dir: str = '',
        # metaclean
        metaclean: Type[MetaClean] | None = None,
        **kwargs
    ):
        """
        Args:
            clean_chans (list | numpy.ndarray | pandas.Series | None,
                optional): Users can choose to provide the fluorescent channels
                they want to use. If set to `None`, 0 will select
                channels automatically. Defaults to None.
            clean_chans_no (int, optional): The maximum number of channels
                to use for cleaning. Less channels make feature calculation
                faster and reduces potential noise. Defaults to 4.
            candidate_chans_type (str, optional): The type of channels to use
                for cleaning i.e. 'fluo' (fluorescent),
                'phys' (physical morphology), or 'all'. Defaults to 'fluo'.
            corr_type (str, optional): Type of summarization to use when
                calculating correlation for finding best channels to use for
                cleaning (i.e. 'max', 'mean', or 'min'). Defaults to 'max'.
            strict_remove_duplicates (bool, optional): Whether to remove
                duplicates strictly (by row) (slow) or not (by row sum) (fast).
                Defaults to False.
            n_cores (int, optional): The number of cores to use while
                calculating the density feature. Set to -1 to use all cores.
                Defaults to -1.
            dens_agg_type (str, optional): Type of summarization to use when
                calculating density features (i.e. 'max', 'mean', or 'min').
                Defaults to 'max'.
            p (int, optional): For calculating density features.
                See `kdtree.query`. Defaults to 2.
            eps (float, optional): For calculating density features.
                See `kdtree.query`. Defaults to 0.1.
            outlier_thresh (float, optional): For calculating density features.
                See `contamination` argument in
                `sklearn.ensemble.IsolationForest`. Defaults to 0.01.
            rm_outliers (str, optional): Wether to remove all, some, or none of
                the outliers (i.e. 'all', 'some', or 'none'). Defaults to 'all'.
            random_seed (int, optional): Random seed. Defaults to 623.
            verbose (bool, optional): Whether to display progress messages.
                Defaults to True.
            png_dir (str, optional): .png path of where users want to save
                plots; no plot will be made if set to ''. Defaults to ''.
            metaclean (MetaClean | None, optional): Initialized `MetaClean`
                class. Defaults to None.
            \**kwargs param: See `MetaClean` class attributes.
        """
        self.clean_chans = clean_chans
        self.clean_chans_no = arg_lim(clean_chans_no, xmin=1)
        if corr_type not in ['max', 'mean', 'min']:
            raise ValueError('{}{}'.format(
                'corr_type chosen is not available, ',
                'please choose from \'max\', \'mean\', or \'min\''))
        self.corr_type = corr_type
        if candidate_chans_type not in ['fluo', 'phys', 'all']:
            raise ValueError('{}{}'.format(
                'candidate_chans_type chosen is not available, ',
                'please choose from \'fluo\', \'phys\', or \'all\''))
        self.candidate_chans_type = candidate_chans_type
        self.strict_remove_duplicates = strict_remove_duplicates

        # calculate density
        self.n_cores = 1 if n_cores == 0 else arg_lim(n_cores, xmin=-1)
        if dens_agg_type not in ['max', 'mean', 'min']:
            raise ValueError('{}{}'.format(
                'dens_agg_type chosen is not available, ',
                'please choose from \'max\', \'mean\', or \'min\''))
        self.dens_agg_type = dens_agg_type
        self.p = p # see kdtree.query
        self.eps = eps # see kdtree.query
        self.dens_k_dtm = dens_k_dtm
        self.outlier_func = IsoForestDetector(
            contamination=outlier_thresh,
            n_estimators=outlier_trees,
            random_state=random_seed,
        )
        self.outlier_drop_and = outlier_drop_and
        if rm_outliers not in ['none', 'some', 'all']:
            raise ValueError('{}{}'.format(
                'rm_outliers chosen is not available, ',
                'please choose from \'none\', \'some\', or \'all\''))
        self.rm_outliers = rm_outliers

        self.random_seed = random_seed
        self.verbose = verbose
        self.png_dir = png_dir
        self.fcs = None
        self.metaclean = None

        # fit: metaclean
        if not (metaclean is None):
            self.metaclean = metaclean
        else:
            if 'verbose' not in kwargs:
                kwargs['verbose'] = self.verbose
            if 'png_dir' not in kwargs:
                kwargs['png_dir'] = self.png_dir
            self.metaclean = MetaClean(random_seed=random_seed, **kwargs)

    # apply split in two for testing
    def apply(
        self,
        fcs: Type[FCSfile] | None = None,
        return_binned_data: bool = False,
        **kwargs
    ):# -> pd.DataFrame:
        """Prepare given `fcs` and apply MetaClean.

        Args:
            fcs (FCSfile | None, optional): Initialized `FCSfile` data class.
                Defaults to None.
            return_binned_data (bool, optional): Whether or not to return binned
                data. Defaults to False.
            \**kwargs param: See `FCSfile` class attributes, namely `data`
                (event x featire pandas.DataFrame).

        Returns:
            pandas.DataFrame: A data frame with the same number of rows as
                `fcs.data`. Columns with prefix `val_` contains
                feature values, `outlier_keep` contain boolean values for bins
                considered (`True`) as outliers, `bin` contain bin labels,
                `segments_raw` contain the raw segment labels, `segments`
                contain the merged and refined segment labels, and `clean_keep`
                contain boolean values for bins to keep (`True`) ---
                `clean_keep` is the final recommentation given by 0.
        """
        start0 = time.time()
        data0 = self.apply_features(
            fcs=fcs, calculate_outlier2=self.rm_outliers == 'some', **kwargs)
        data1 = self.apply_clean(data=data0)
        if self.verbose:
            print(' + Total time for MetaCleanFCS: {}\'s'.format(
                np.round(time.time() - start0, 3)))

        # keep outliers if specified
        if self.rm_outliers != 'none':
            dt = self.remove_outliers(copy.copy(data1))
            if not (dt is None):
                data1 = dt

        data_final = pd.merge(self.fcs.data, data1, on=self.fcs.bin_chan)
        data_final.sort_values(by='index_original')

        if return_binned_data:
            return data_final, data1
        return data_final

    def apply_features(
        self,
        fcs: FCSfile | None = None,
        calculate_outlier2: bool = False,
        **kwargs
    ):# -> pd.DataFrame:
        """Calcuate bins and features using raw values.

        Args:
            fcs (FCSfile | None, optional): _description_. Defaults to None.
            calculate_outlier2 (bool, optional): Whether or not to identify
                outliers again more leniently if user wishes to remove only
                some outliers. Defaults to False.

        Returns:
            pandas.DataFrame: A data matrix containing bins (`bin` column),
                feature values (prefix `val_` column(s)), and
                outliers (`outlier_keep` column).
        """
        self.fcs = FCSfile(**kwargs) if fcs is None else fcs
        # find best channels for cleaning
        start = time.time()
        candidate_chans = self.clean_chans
        if candidate_chans is None:
            candidate_chans = np.concatenate(
                (self.fcs.phys_chans, self.fcs.fluo_chans))
            if self.candidate_chans_type == 'fluo':
                if len(self.fcs.fluo_chans) > 0:
                    candidate_chans = self.fcs.fluo_chans
                else:
                    warnings.warn('{}{}'.format(
                        'No fluorescent channels found, ',
                        'defaulting to all channels.'))
            elif self.candidate_chans_type == 'phys':
                if len(self.fcs.phys_chans) > 0:
                    candidate_chans = self.fcs.phys_chans
                else:
                    warnings.warn('{}{}'.format(
                        'No physical morphology channels found, ',
                        'defaulting to all channels.'))
        self.clean_chans = most_corr_channels(
            data=self.fcs.data[candidate_chans],
            bins=self.fcs.data[self.fcs.bin_chan],
            chosen_chans=self.clean_chans,
            candidate_no=self.clean_chans_no,
            corr_type=self.corr_type
        )
        # scale data
        # 210 seconds
        ds_input = randomize_duplicates(
            self.fcs.data[self.clean_chans],
            strict=self.strict_remove_duplicates,
            seed=self.random_seed
        )
        if self.verbose:
            print(' + Time to choose channels and check duplicates: {}\'s ({})'.format(
                np.round(time.time() - start, 3), self.clean_chans))
        start = time.time()
        ds = RobustScaler().fit_transform(ds_input)
        bin_values = self.fcs.data[self.fcs.bin_chan]
        d = get_bin_density( # 29 vs 180 with dups
            data=ds,
            bins=bin_values,
            dens_agg_type=self.dens_agg_type,
            n_cores=self.n_cores,
            p=self.p, # see kdtree.query
            eps=self.eps, # see kdtree.query
            dens_k_dtm=self.dens_k_dtm)
        v = get_bin_moments(data=ds, bins=bin_values, mmt=2)
        s = get_bin_moments(data=ds, bins=bin_values, mmt=3)
        # _, fr = np.unique(bin_values, return_counts=True) #***
        vs = np.sum(np.stack((v, s)), axis=0)
        if self.verbose:
            print(' + Time to calculate features: {}\'s'.format(
                np.round(time.time() - start, 3)))
        # remove outliers
        start = time.time()
        data = pd.DataFrame({'val_dens': d, 'val_varskew': vs})
        out_keep = drop_outliers(
            data=data,
            outlier_func=self.outlier_func,
            drop_and=self.outlier_drop_and
        )
        if calculate_outlier2:
            self.out_keep_final = drop_outliers(
                data=pd.DataFrame({'val': d+vs}),
                outlier_func=self.outlier_func,
                drop_and=self.outlier_drop_and
            )
        # scale values
        ds = scale_values(data[out_keep])
        for v1 in ds.columns:
            data['{}_scaled'.format(v1)] = 0.0
            data['{}_scaled'.format(v1)][out_keep] = ds[v1]
        data['val_scaled'] = data[
            ['val_dens_scaled', 'val_varskew_scaled']].sum(axis=1)
        data['outlier_keep'] = out_keep
        # data['flowrate'] = fr
        if calculate_outlier2:
            data['outlier_keep_final'] = self.out_keep_final

        if self.verbose:
            print(' + Time to remove outliers + scale features: {}\'s'.format(
                np.round(time.time() - start, 3)))

        return data

    def apply_clean(self, data: pd.DataFrame):# -> pd.DataFrame:
        """Apply cleaning.

        Args:
            data (pandas.DataFrame): Object x feature matrix.

        Returns:
            pandas.DataFrame: See `MetaClean`.
        """
        out_keep = data['outlier_keep']

        # metaclean!
        data_ = data[out_keep]
        data_.index = range(len(data_))
        clean_keep, segments2, segments0 = self.metaclean.clean(
            data=data_,
            val_cols=['val_dens_scaled', 'val_varskew_scaled'],
            val_col_final='val_scaled'
        )
        data[self.fcs.bin_chan] = np.unique(self.fcs.data[self.fcs.bin_chan])
        data = data.assign(
            **{'clean_keep': False, 'segments_raw': -1, 'segments': -1})
        data['clean_keep'][out_keep] = clean_keep
        data['segments'][out_keep] = segments2
        data['segments_raw'][out_keep] = segments0

        return data

    def remove_outliers(
        self,
        data_merged: pd.DataFrame
    ):# -> pd.DataFrame:
        """
        After cleaning is done, revisits outliers to either keep all, some,
        or none of the outliers.

        Args:
            data_merged (pandas.DataFrame): object/row x feature matrix.

        Returns:
            pandas.DataFrame: `data_merged` where the `clean_keep` column is
                editted such that all, some, or no outliers are kept.
        """
        outs = data_merged['outlier_keep']
        if self.rm_outliers == 'some':
            outs = (outs & ~self.out_keep_final)
        if not any(outs) or self.rm_outliers == 'none':
            return None

        out_chpts = np.where( # not the case, but just in case
            np.diff(np.concatenate(([True], outs, [True]))) != 0)[0]

        if out_chpts[0] == 0:
            if len(out_chpts) == 2:
                return None

        out_chpts = out_chpts[2:]
        if out_chpts[-1] >= (len(outs) - 1):
            if len(out_chpts) == 2:
                return None

        out_chpts = out_chpts[:-2]
        clean_keepers = np.array(data_merged['clean_keep'])
        data_new = copy.copy(data_merged)
        for i in range(0, len(out_chpts), 2):
            keeper = all(clean_keepers[[out_chpts[i] - 1, out_chpts[i+1] + 1]])
            data_new['clean_keep'][out_chpts[i]:out_chpts[i+1]] = keeper

        return data_new
