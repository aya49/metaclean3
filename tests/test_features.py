import unittest
import numpy as np
import pandas as pd
from ddt import ddt, data, unpack

from metaclean3.utils import seq_array
from metaclean3.features import scale_values, get_bin_density, get_bin_moments

@ddt
class TestFeatures(unittest.TestCase):
    @data(20000)
    def test_scale_values(self, size: int):
        print('testing scale values')
        x_temp = {'x': [i for i in range(size)], 'y': [1 for _ in range(size)],
                  'z': np.random.normal(size=size)}
        x = pd.DataFrame(x_temp)
        xs = scale_values(x)
        self.assertLessEqual( np.max(xs['x']), 1.0 )
        self.assertAlmostEqual( np.min(xs['x']), 0.0 )
        self.assertAlmostEqual( np.max(xs['z']), 1.0 )
        self.assertAlmostEqual( np.min(xs['z']), 0.0 )
        self.assertAlmostEqual( np.max(xs['y']), 0.0 )
        self.assertAlmostEqual( np.min(xs['y']), 0.0 )

    @data((20000, 2000))
    @unpack
    def test_get_bin_density(self, size: int, max_val: int):
        print('testing calculate bin density')
        x = pd.DataFrame({'x': [i for i in range(size)]})
        y = pd.DataFrame({'x': [i for i in range(size)],
                          'y': [1 for _ in range(size)]})
        bins = seq_array(size=len(x), max_val=max_val)
        xd = get_bin_density(data=x, bins=bins)
        self.assertEqual( len(xd.shape), 1 )
        self.assertEqual( len(xd), max_val )
        yd = get_bin_density(data=y, bins=bins)
        self.assertEqual( len(yd.shape), 1 )
        self.assertEqual( len(yd), max_val )

    @data((20000, 2000))
    @unpack
    def test_get_bin_moments(self, size: int, max_val: int):
        print('testing calculate bin moments')
        x = pd.DataFrame({'x': [i for i in range(size)]})
        y = pd.DataFrame({'y': [1 for _ in range(size)]})
        bins = seq_array(size=len(x), max_val=max_val)
        xd = get_bin_moments(data=x, bins=bins, mmt=2)
        self.assertEqual( len(xd.shape), 1 )
        self.assertEqual( len(xd), max(bins) )
        yd = get_bin_moments(data=y, bins=bins, mmt=2)
        self.assertEqual( len(yd.shape), 1 )
        self.assertEqual( len(yd), max(bins) )
        xd = get_bin_moments(data=x, bins=bins, mmt=3)
        self.assertEqual( len(xd.shape), 1 )
        self.assertEqual( len(xd), max(bins) )
        yd = get_bin_moments(data=y, bins=bins, mmt=3)
        self.assertEqual( len(yd.shape), 1 )
        self.assertEqual( len(yd), max(bins) )

        # check the case when not all bins have the same # of events
        x = pd.DataFrame({'x': [i for i in range(size-1)]})
        y = pd.DataFrame({'y': [1 for _ in range(size-1)]})
        bins = seq_array(size=len(x), max_val=max_val)
        xd = get_bin_moments(data=x, bins=bins, mmt=2)
        self.assertEqual( len(xd.shape), 1 )
        self.assertEqual( len(xd), max(bins) )
        yd = get_bin_moments(data=y, bins=bins, mmt=2)
        self.assertEqual( len(yd.shape), 1 )
        self.assertEqual( len(yd), max(bins) )
        xd = get_bin_moments(data=x, bins=bins, mmt=3)
        self.assertEqual( len(xd.shape), 1 )
        self.assertEqual( len(xd), max(bins) )
        yd = get_bin_moments(data=y, bins=bins, mmt=3)
        self.assertEqual( len(yd.shape), 1 )
        self.assertEqual( len(yd), max(bins) )
