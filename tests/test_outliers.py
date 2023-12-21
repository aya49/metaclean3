import unittest
import numpy as np
import pandas as pd
from ddt import ddt, data

from metaclean3.outliers import drop_outliers, IsoForestDetector

@ddt
class TestOutliers(unittest.TestCase):
    @data(20000)
    def test_drop_outliers(self, size: int):
        print('testing outlier detection')
        x_temp = {'x': [1 for _ in range(size)], 'y': [5 for _ in range(size)]}
        x = pd.DataFrame(x_temp)
        outlier_func = IsoForestDetector(
            contamination=0.01,
            n_estimators=500,
            random_state=10
        )

        # test no outliers
        xo = drop_outliers(x, outlier_func=outlier_func)
        self.assertTrue( all(xo) )

        # test and/or outlier detection
        x.iloc[[10, 20, 30], 0] = 3
        x.iloc[[10, 20, 30, 40, 50], 1] = 0
        xo = drop_outliers(x, outlier_func=outlier_func, drop_and=True)
        self.assertEqual( np.sum(xo), (size - 5) )
        xo = drop_outliers(x, outlier_func=outlier_func, drop_and=False)
        self.assertEqual( np.sum(xo), (size - 3) )

if __name__ == '__main__':
    unittest.main()