import unittest
import numpy as np
from ddt import ddt, data, unpack

from metaclean3.utils import seq_array
from metaclean3.binning import (
    get_time_bin_unit,
    adjust_time_bin,
    data_to_flow,
    get_time_binned
)

@ddt
class TestBinning(unittest.TestCase):
    def test_get_time_bin_unit(self):
        print('testing time bin unit retrieval')
        self.assertEqual( get_time_bin_unit([1]), 'S' )

    @data(('100S', '50.0S', 'S', True),
          ('50S', '100.0S', 'S', False),
          ('1S', '0.5S', 'S', True),
          ('0.5S', '1.0S', 'S', False),
          ('0.001S', '0.5ms', 'ms', True),
          ('50ms', '100.0ms', 'ms', False),
          ('0.001ms', '0.5us', 'us', True),
          ('500us', '1.0ms', 'ms', False))
    @unpack
    def test_adjust_time_bin(
        self,
        in_tb: str,
        out_tb: str,
        out_time: str,
        rd: bool
    ):
        print('testing time bin adjustment')
        tb, tu = adjust_time_bin(in_tb, reduce=rd)
        self.assertEqual( tb, out_tb )
        self.assertEqual( tu, out_time )

    @data((10000, 2000))
    @unpack
    def test_data_to_flow(self, size: int, max_val: int):
        print('testing purturbation of time channel for more granular bins')
        x = seq_array(size=size, max_val=max_val)
        y = data_to_flow(x)
        self.assertLessEqual( len(np.unique(x)), len(np.unique(y)) )

    @data((10000, 1999, 2000, 10000, 50, 200),
          (10000, 1999, 2000, 10000, 5, 2000),
          (100000, 100, 2000, 10000, 5, 2000),
          (4253555, 6513, 3000, 10000, 1, 3000))
    @unpack
    def test_get_time_binned(
        self,
        size: int,
        max_val: int,
        min_bin_size: int,
        max_bin_size: int,
        min_events_per_bin: int,
        out_len: int
    ):
        print('testing binning')
        x = seq_array(size=size, max_val=max_val)
        y = get_time_binned(
            x, min_bin_size=min_bin_size,
            max_bin_size=max_bin_size,
            min_events_per_bin=min_events_per_bin
        )
        self.assertEqual( len(np.unique(y)), out_len )

if __name__ == '__main__':
    unittest.main()