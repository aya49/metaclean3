import unittest
import numpy as np
from ddt import ddt, data, unpack

from metaclean3.utils import (
    arg_lim,
    is_monotonic,
    mode,
    str_to_time,
    duplicated_rows,
    randomize_duplicates,
    round_to_1,
    seq_array,
    order_value_bins
)

@ddt
class TestUtils(unittest.TestCase):
    @data((10, -np.Inf, np.Inf, 10),
          (10, 10, np.Inf, 10),
          (10, 15, np.Inf, 15),
          (10, 15, np.Inf, 15),
          (10, 5, 0, 10),
          (10, 0, 5, 5))
    @unpack
    def test_arg_lim(
        self,
        x: int | float,
        xmin: int | float,
        xmax: int | float,
        out_x: int | float
    ):
        print("testing argument limits.")
        self.assertEqual( arg_lim(x=x, xmin=xmin, xmax=xmax), out_x )

    def test_is_monotonic(self):
        print("testing monotonicity.")
        x = np.array([i for i in range(10)])
        y = np.array([11 for _ in range(10)])
        z = np.append(x, y)
        self.assertFalse( is_monotonic(np.array([])) )
        self.assertFalse( is_monotonic(np.array([1])) )
        self.assertTrue( is_monotonic(x, strict=True) )
        self.assertTrue( is_monotonic(x, strict=False) )
        self.assertTrue( is_monotonic(-x, strict=True) )
        self.assertTrue( is_monotonic(-x, strict=False) )
        self.assertFalse( is_monotonic(y, strict=True) )
        self.assertTrue( is_monotonic(y, strict=False) )
        self.assertFalse( is_monotonic(z, strict=True) )
        self.assertTrue( is_monotonic(z, strict=False) )
        self.assertFalse( is_monotonic(-z, strict=True) )
        self.assertTrue( is_monotonic(-z, strict=False) )

    @data(([0], 0), ([0,0], 0), ([1,1,2], 1), ([0,0,1,1,2], 0))
    @unpack
    def test_mode(self, x: list, out_mode: int):
        print("testing significance test.")
        self.assertEqual( mode(x), out_mode )

    @data(('2.5S', 2.5, 'S'), ('10ms', 10.0, 'ms'))
    @unpack
    def test_str_to_time(self, in_time: str, out_tb: float, out_time: str):
        print("testing string to time converstion")
        tb, tu = str_to_time(in_time)
        self.assertEqual( tb, out_tb )
        self.assertEqual( tu, out_time )

    @data((np.array([[1,2], [3,1], [1,3], [2,1]]), False, 2, 2),
          (np.array([[1,2], [3,1], [1,3], [2,1]]), True, 0, 0),
          (np.array([[1,2], [3,1], [1,3], [2,1], [2,1]]), True, 1, 1))
    @unpack
    def test_duplicated_rows(
        self,
        x: np.ndarray,
        strict: bool,
        out_sum: int,
        out_count: int
    ):
        print("testing duplicate row detection.")
        xtf, xc = duplicated_rows(x, strict=strict)
        self.assertEqual( np.sum(xtf), out_sum )
        self.assertEqual( xc, out_count )

    def test_randomize_duplicates(self):
        print("testing duplicate row removal.")
        x = np.array([[1,2], [3,1], [1,3], [2,1], [2,1]])
        _, xc = duplicated_rows(x, strict=True)
        x1 = randomize_duplicates(x, strict=False)
        x2 = randomize_duplicates(x, strict=True)
        self.assertLess( duplicated_rows(x1, strict=True)[1], xc )
        self.assertLess( duplicated_rows(x2, strict=True)[1], xc )

    @data((0.0029, 0.003), (0.0142, 0.01), (65000.0, 60000.0), (930.0, 900.0))
    @unpack
    def test_round_to_1(self, in_float: float, out_float: float):
        print('testing round to nearest power of 10')
        self.assertEqual( round_to_1(in_float), out_float)

    @data((10000, 2000))
    @unpack
    def test_seq_array(self, size: int, max_val: int):
        print('testing time channel creation.')
        x = seq_array(size=size, max_val=max_val)
        self.assertEqual( len(x), size )
        self.assertEqual( min(x), 1 )
        self.assertEqual( max(x), max_val )

    @data(2000)
    def test_order_value_bins(self, size: int):
        print('testing bin/segment sorting.')
        # test with one segment
        x = np.array([i for i in range(size)])
        y = [1 for _ in range(size)]
        xo = order_value_bins(vlist=x, segments=y)
        for i in range(size):
            self.assertEqual(xo[i], i)
        # test with malformed segment
        y = [1 for _ in range(size-1)]
        with self.assertRaises(ValueError):
            xo = order_value_bins(vlist=x, segments=y)
        # test with more than one segment
        y = [np.random.randint(1, size/50) for _ in range(size)]
        xo = order_value_bins(vlist=x, segments=y)
        for i in range(size):
            self.assertIn( i, xo )
        xo = order_value_bins(vlist=x, segments=np.sort(y))
        for i in range(size):
            self.assertEqual( xo[i], i )

if __name__ == '__main__':
    unittest.main()


