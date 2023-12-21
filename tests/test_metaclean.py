import unittest
from pathlib import Path
import os
import numpy as np
import copy
import fcsparser
from ddt import ddt, data, unpack

from metaclean3.utils import (
    get_timestep,
    get_spillover_raw,
    apply_compensation_matrix
)
from metaclean3.fcs import FCSfile
from metaclean3.metaclean import MetaCleanFCS

DATA_DIR = Path(__file__).parent / 'data'

@ddt
class TestMetaCleanFCS(unittest.TestCase):
    def load_fcs(self):
        fcs_paths = [p for p in os.listdir(DATA_DIR) if p.endswith('.fcs')]
        for fp in fcs_paths:
            print('loading test file: {}'.format(fp))
            meta, data = fcsparser.parse(DATA_DIR / fp, reformat_meta=False)
            time_step = get_timestep(meta)
            sm = get_spillover_raw(meta=meta, dat_columns=list(data.columns))
            if not (sm is None):
                data[sm.columns] = apply_compensation_matrix(
                    data[sm.columns], sm)
            yield FCSfile(data=data, time_step=time_step)

    @unpack
    @data({'seg_method': 'pelt', 'merge_method': 'sd',
           'rm_outliers': 'all', 'cost_model_seg': 'rbf',
           'cost_model_gain': 'rank'},
          {'seg_method': 'pelt', 'merge_method': 'sequential',
           'rm_outliers': 'some', 'cost_model_seg': 'rbf',
           'cost_model_gain': 'rank'},
          {'seg_method': 'pelt', 'merge_method': 'sd',
           'rm_outliers': 'some', 'cost_model_seg': 'rbf',
           'cost_model_gain': 'rank'},
          {'seg_method': 'pelt', 'merge_method': 'sequential',
           'rm_outliers': 'none', 'cost_model_seg': 'rbf',
           'cost_model_gain': 'rank'},
          {'seg_method': 'pelt', 'merge_method': 'sd',
           'rm_outliers': 'none', 'cost_model_seg': 'rbf',
           'cost_model_gain': 'rank'},
          {'seg_method': 'pelt', 'merge_method': 'sequential',
           'rm_outliers': 'all', 'cost_model_seg': 'rbf',
           'cost_model_gain': 'rank'})
    def test_apply(self, **kwargs):
        print('testing apply.')
        for f in self.load_fcs():
            png_paths = [p for p in os.listdir(DATA_DIR) if p.endswith('.png')]
            if len(png_paths) > 0:
                for pp in png_paths:
                    os.remove(DATA_DIR / pp)

            # initialize metaclean
            mc = MetaCleanFCS(png_dir=DATA_DIR, **kwargs)
            d = mc.apply(fcs=copy.copy(f))

            # check row/col
            self.assertEqual( len(d), len(f.data) )
            self.assertEqual( len(d), len(mc.fcs.data) )
            data_cols = ['val_dens', 'val_varskew', 'val_dens_scaled',
                         'val_varskew_scaled', 'val_scaled',
                         'outlier_keep', #'outlier_keep_final',
                         'bin', 'clean_keep']#, 'segments_raw', 'segments']
            self.assertTrue( all(np.isin(data_cols, d.columns)) )

            # check plots
            png_paths = [p for p in os.listdir(DATA_DIR) if p.endswith('.png')]
            self.assertGreater( len(png_paths), 0 )
            for pp in png_paths:
                os.remove(DATA_DIR / pp)

if __name__ == '__main__':
    unittest.main()