import unittest
import math
import numpy as np
import pandas as pd
from ddt import ddt, data, unpack

from metaclean3.utils import is_monotonic, seq_array
from metaclean3.channels import (
    get_clean_time_chan,
    verify_channels,
    get_clean_fp_channels,
    most_corr_channels
)

@ddt
class TestChannels(unittest.TestCase):
    @data((10000, 2000, 'time'))
    @unpack
    def test_get_clean_time_chan(self, size: int, max_val: int, tc0: str):
        print('testing time channel detection.')
        x_temp = {'x': [i for i in range(size)], 'y': [1 for _ in range(size)]}
        x = pd.DataFrame(x_temp)

        # data with no time column + make time column
        tc, x = get_clean_time_chan(data=x, time_chan=tc0, min_bin_size=max_val)
        self.assertEqual( tc, tc0 )
        self.assertIn( tc, x.columns )
        self.assertTrue( is_monotonic(x[tc], strict=False) )

        # data with unordered time column
        x = pd.concat([x, x])
        self.assertEqual( tc, tc0 )
        self.assertIn( tc, x.columns )
        self.assertFalse( is_monotonic(x[tc], strict=False) )

        # data with time column
        tc, _ = get_clean_time_chan(data=x)
        self.assertEqual( tc, 'time' )

    def test_verify_channels(self):
        print('testing channel subsetting.')
        x = ['Fluo1', 'Fluo2']
        y = ['fluo2', 'fluo3', 'fluo4']
        self.assertEqual( len(verify_channels(x, y)), 1 )
        self.assertEqual( len(verify_channels(x[:1], y)), 0 )
        self.assertEqual( len(verify_channels([], y)), 0 )
        self.assertEqual( len(verify_channels(x, [])), 0 )

    @data((20000, 25))
    @unpack
    def test_get_clean_fp_channels(self, size: int, K: int):
        print('testing clean fluorescent channel selection.')
        x_temp = {'x': [i for i in range(size)], 'y': [1 for _ in range(size)],
                'z': seq_array(size=size, max_val=10)}
        x = pd.DataFrame(x_temp)
        with self.assertRaises(ValueError):
            get_clean_fp_channels(data=x, fluo_chans='x')

        # verification of manually given channels
        x['f'] = np.random.normal(size=size)
        fc, pc = get_clean_fp_channels(data=x, fluo_chans=['x', 'f'])
        self.assertEqual( len(fc), 1 )
        self.assertEqual( fc[0], 'f' )
        self.assertEqual( len(pc), 0 )
        fc, pc = get_clean_fp_channels(data=x, fluo_chans=['f'])
        self.assertEqual( len(fc), 1 )
        self.assertEqual( fc[0], 'f' )
        self.assertEqual( len(pc), 0 )
        fc, pc = get_clean_fp_channels(data=x)
        self.assertEqual( len(fc), 1 )
        self.assertEqual( fc[0], 'f' )
        self.assertEqual( len(pc), 0 )

        # data with one fluo channel
        x = pd.DataFrame({ 'a': np.random.normal(size=size) })
        _, x = get_clean_time_chan(data=x)
        fc, pc = get_clean_fp_channels(data=x)
        self.assertIn( 'a', fc )
        self.assertEqual( len(fc), 1 )
        self.assertEqual( len(pc), 0 )

        # data with one -a fluo channel
        x['a-a'] = x['a']
        fc, pc = get_clean_fp_channels(data=x)
        self.assertIn( 'a-a', fc )
        self.assertEqual( len(fc), 1 )
        self.assertEqual( len(pc), 0 )

        # data with one phys channel
        x['fs'] = x['a']
        fc, pc = get_clean_fp_channels(data=x)
        self.assertIn( 'a-a', fc )
        self.assertIn( 'fs', pc )
        self.assertEqual( len(fc), 1 )
        self.assertEqual( len(pc), 1 )

        # data with one -a phys channel
        x['fs-a'] = x['a']
        fc, pc = get_clean_fp_channels(data=x)
        self.assertIn( 'a-a', fc )
        self.assertIn( 'fs-a', pc )
        self.assertEqual( len(fc), 1 )
        self.assertEqual( len(pc), 1 )

        # data with unqualified -a fluo and phys channels
        k = int(math.sqrt(K))
        x['ss-a'] = np.random.choice(k, size=len(x))
        ba = [i for i in range(1, k) for _ in range(math.ceil(len(x)/(k - 1)))]
        x['b-a'] = ba[:len(x)]
        fc, pc = get_clean_fp_channels(data=x, channel_unique_no=K)
        self.assertIn( 'a-a', fc )
        self.assertIn( 'fs-a', pc )
        self.assertEqual( len(fc), 1 )
        self.assertEqual( len(pc), 1 )

    @data((20000, 2000, 'time'))
    @unpack
    def test_most_corr_channels(self, size: int, max_val: int, tc0: str):
        print('testing most correlated channel selection')
        # test one column
        x = pd.DataFrame({ 'a': np.random.normal(size=size) })
        fc = most_corr_channels(data=x)
        self.assertIn( 'a', fc )
        self.assertEqual( len(fc), 1 )

        x_temp = {'x': [i for i in range(size)], 'y': [1 for _ in range(size)],
                  'z': np.random.normal(size=size)}
        x = pd.DataFrame(x_temp)
        x = pd.concat([x, x])
        _, x = get_clean_time_chan(data=x, time_chan=tc0, min_bin_size=max_val)
        bins = seq_array(size=len(x), max_val=max_val)

        # test with/out bins
        xl = len(x.columns)
        fc = most_corr_channels(data=x, candidate_no=xl)
        self.assertEqual( len(fc), xl )
        fc = most_corr_channels(data=x, candidate_no=xl-1)
        self.assertEqual( len(fc), xl - 1 )
        self.assertEqual( fc[0], tc0 )
        fc = most_corr_channels(data=x, bins=bins, candidate_no=xl)
        self.assertEqual( len(fc), xl )
        fc = most_corr_channels(data=x, bins=bins, candidate_no=xl-1)
        self.assertEqual( len(fc), xl - 1 )
        self.assertEqual( fc[0], tc0 )

        # test with all/few candidate numbers
        fc = most_corr_channels(data=x, candidate_no=10)
        self.assertEqual( len(fc), xl )
        fc = most_corr_channels(data=x, candidate_no=1)
        self.assertEqual( len(fc), 1 )
        self.assertEqual( fc[0], tc0 )

if __name__ == '__main__':
    unittest.main()