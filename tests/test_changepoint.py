import numpy as np
import ruptures as rpt
import unittest
from ddt import ddt, data

from metaclean3.changepoint import binseg_gain, list_gains

@ddt
class TestChangepoint(unittest.TestCase):
    @data(2000)
    def test_binseg_gain(self, size: int):
        print('Testing gains (single)')
        x = np.random.normal(size=size)
        y = np.random.normal(size=size, scale=10)
        rpt_class1 = rpt.Binseg(model='rbf', jump=2, min_size=10)
        rpt_class1 = rpt_class1.fit(signal=x)
        rpt_class2 = rpt.Binseg(model='rbf', jump=2, min_size=10)
        rpt_class2 = rpt_class2.fit(signal=np.concatenate((x, y)))

        xg1 = binseg_gain(
            start=0, bkp=int(size/2), end=size, rpt_class=rpt_class1, dif=True)
        xg2 = binseg_gain(
            start=0, bkp=int(size/2), end=size, rpt_class=rpt_class1, dif=False)
        yg1 = binseg_gain(
            start=0, bkp=size, end=size*2, rpt_class=rpt_class2, dif=True)
        yg2 = binseg_gain(
            start=0, bkp=size, end=size*2, rpt_class=rpt_class2, dif=False)

        self.assertLess( xg1, yg1 )
        self.assertLess( xg2, yg2 )

    @data(2000)
    def test_list_gains(self, size: int):
        print('Testing gains')
        x = np.random.normal(size=size)
        chpts0M = [0, 10, 200, 1000, 1300, size]
        rpt_class = rpt.Binseg(model='rbf', jump=2, min_size=10)
        rpt_class = rpt_class.fit(signal=x)
        xg = list_gains(chpts0M=chpts0M, rpt_class=rpt_class, dif=True)
        self.assertEqual( len(xg), 6 )
        self.assertEqual( np.sum(xg == -1), 2 )
        self.assertEqual( np.sum(xg >= 0), 4 )
        xg = list_gains(chpts0M=chpts0M, rpt_class=rpt_class, dif=False)
        self.assertEqual( len(xg), 6 )
        self.assertEqual( np.sum(xg == -1), 2 )
        self.assertEqual( np.sum(xg >= 0), 4 )
        xg = list_gains(chpts0M=chpts0M, rpt_class=rpt_class, chpts_inds=[1, 3])
        self.assertEqual( np.sum(xg >= 0), 2 )

if __name__ == '__main__':
    unittest.main()
