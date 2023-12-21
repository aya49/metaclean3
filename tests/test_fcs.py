import unittest
import numpy as np
import pandas as pd
from ddt import ddt, data, unpack

from metaclean3.channels import get_clean_time_chan
from metaclean3 import FCSfile

@ddt
class TestFCSfile(unittest.TestCase):
    @data((10000, 2000, 'time'))
    @unpack
    def test_FCSfile(self, size: int, max_val: int, tc0: str):
        print('testing data class `FCSfile`')
        x_temp = {'x': [i for i in range(size)], 'y': [1 for _ in range(size)],
                  'z': np.random.normal(size=size)}
        x = pd.DataFrame(x_temp)
        x = pd.concat([x, x])
        _, x = get_clean_time_chan(data=x, time_chan=tc0, min_bin_size=max_val)
        ff = FCSfile(data=x)
        self.assertTrue( all(
            np.isin(['time', 'index_original', 'bin'], ff.data.columns)) )
        self.assertEqual( len(ff.fluo_chans), 2 )
        self.assertTrue( all(np.isin(['x', 'z'], ff.fluo_chans)) )
        self.assertEqual( len(ff.phys_chans), 0 )

if __name__ == '__main__':
    unittest.main()