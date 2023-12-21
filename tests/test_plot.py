import unittest
import os
from pathlib import Path
import numpy as np
from ddt import ddt, data

from metaclean3.plot import plot_scat

@ddt
class TestPlot(unittest.TestCase):
    @data(10000)
    def test_plot_scat(self, size: int):
        print('testing scatterplot')
        x = np.random.normal(size=size)
        y = [1 for _ in range(size)]
        L = [np.random.randint(0, 10) for _ in range(size)]
        fp = Path(__file__).parent / 'temp.png'
        plot_scat(x=x, y=y, L=L, out_path=fp)
        self.assertTrue( os.path.isfile(fp) )
        os.remove(fp)

if __name__ == '__main__':
    unittest.main()