import unittest
import numpy as np
import copy
from ddt import ddt, data, unpack

from metaclean3.segments import (
    chpts0M_to_segments,
    segments_to_chpts0M,
    get_ref_ranges,
    get_test_ranges,
    apply_ranges,
    in_percentiles,
    get_ref_updown,
    get_ref_func2,
    merge_segments_test,
    merge_segments_gain,
    merge_segments_quant,
    refine_reference_segment,
    refine_reference_segment_inbetween,
    remove_reference_ends
)

# note: scipy.stats.gmean sometimes ->
# RuntimeWarning: invalid value encountered in log log_a = np.log(a)

@ddt
class TestSegments(unittest.TestCase):
    @data(2000)
    def test_get_ref_ranges(self, size: int):
        print('testing getting reference segment ranges.')
        x = [np.random.normal(size=size) for _ in range(2)]
        res, res_all = get_ref_ranges(ref_values=x)
        self.assertEqual( len(res), 4 )
        self.assertLess( res[0], res[1] )
        self.assertLessEqual( res[1], 0 )
        self.assertGreaterEqual( res[2], 0 )
        self.assertLess( res[2], res[3] )
        self.assertEqual( len(res_all), len(x) )
        # test the length of res_all
        x = [np.random.normal(size=size) for _ in range(4)]
        res, res_all = get_ref_ranges(ref_values=x)
        self.assertEqual( len(res_all), len(x) )

    @data(2000)
    def test_get_test_ranges(self, size: int):
        print('testing getting test segment ranges.')
        x = [np.random.normal(size=size) for _ in range(2)]
        y = np.random.normal(size=size)
        res = get_test_ranges(ref_values=x, test_value=y)
        self.assertEqual( len(res), 4 )
        self.assertLess( res[0], res[1] )
        self.assertLessEqual( res[1], 0 )
        self.assertGreaterEqual( res[2], 0 )
        self.assertLess( res[2], res[3] )

    @data(2000)
    def test_apply_ranges(self, size: int):
        print('testing range comparisons.')
        x = [np.random.normal(size=size) for _ in range(2)]
        y = np.random.normal(size=size)
        res1, _ = get_ref_ranges(ref_values=x)
        res2 = get_test_ranges(ref_values=x, test_value=y)
        self.assertTrue( apply_ranges(res1, res2) )
        # test when two segments are different
        res2 = np.array(res2) + 1
        self.assertFalse( apply_ranges(ref_range=res1, test_range=res2) )

    @data(2000)
    def test_in_percentiles(self, size: int):
        print('testing in-percentiles.')
        x = [np.random.normal(size=size) for _ in range(2)]
        y = np.random.normal(size=size)
        self.assertTrue( in_percentiles(ref_values=x, test_value=y) )
        # test when two segments are different
        y = y + 1
        res = in_percentiles(ref_values=x, test_value=y)
        self.assertFalse( res )

    @data(2000)
    def test_get_ref_updown(self, size: int):
        print('testing getting <=2 adjacent reference segments.')
        r = 0
        x = [np.random.normal(size=size) for _ in range(1)]
        xr = get_ref_updown(r=r, chpts0M=[0, size], vlist=x)
        xr_len = [[[[len(r) for r in xr[i][j]] for j in
                    range(len(xr[i]))]] for i in range(len(xr))]
        self.assertEqual( np.max(xr_len), size )

    # def test_go_updown():
    def test_get_ref_func2(self):
        print('testing gerge test.')
        # when test segment is >= 50
        size = 2000
        x = [np.random.normal(size=size) for _ in range(1)]
        i1 = 100
        i2 = 200
        ref_inds1 = np.array([i for i in range(20)]) + 80
        ref_inds2 = np.array([i for i in range(200)]) + 200
        ref_inds3 = np.array([i for i in range(100)]) + 400
        ris = np.concatenate((ref_inds1, ref_inds2, ref_inds3))
        xr = get_ref_func2(vlist=x, ref_inds=ris, test_i1=i1, test_i2=i2)
        self.assertEqual( len(xr[0][0]), len(ris) )
        self.assertEqual( len(xr[0][1]),i2 - i1 )
        i2 = 150
        ris = np.concatenate((ref_inds2, ref_inds3))
        xr = get_ref_func2(vlist=x, ref_inds=ris, test_i1=i1, test_i2=i2)
        self.assertEqual( len(xr[0][0]), len(ref_inds2) + len(ref_inds3) )
        # when test segment is < 50
        i2 = 120
        ris = np.concatenate((ref_inds1, ref_inds2, ref_inds3))
        xr = get_ref_func2(vlist=x, ref_inds=ris, test_i1=i1, test_i2=i2)
        self.assertEqual( len(xr[0][0]), len(ris) )
        i2 = 100
        ris = np.concatenate((ref_inds1, ref_inds2, ref_inds3))
        with self.assertRaises(ValueError):
            xr = get_ref_func2(vlist=x, ref_inds=ris, test_i1=i1, test_i2=i2)

    # segment merge tests have duplicated examples for easier future
    # editing/customizing.
    @data(2000)
    def test_merge_segments_test(self, size: int):
        print('testing segment merge (significance test).')
        x = np.random.normal(size=size)
        # test no segments
        chpts0M = [0, size]
        chpts0M = segments_to_chpts0M(merge_segments_test(
            values=x, segments=chpts0M_to_segments(chpts0M)))
        self.assertEqual( len(chpts0M), 2 )
        self.assertEqual( chpts0M[0], 0 )
        self.assertEqual( chpts0M[-1], size )
        # test many segments with one small segment
        # significance tests don't work for small segments i.e. length 1.
        chpts0M = [0, 10, 200, 400, 450, 1000, 1300, 1301, size]
        chpts0M_ = segments_to_chpts0M(merge_segments_test(
            values=x, segments=chpts0M_to_segments(chpts0M)))
        self.assertLessEqual( len(chpts0M_), len(chpts0M) )
        self.assertEqual( chpts0M[0], 0 )
        self.assertEqual( chpts0M[-1], size )

    @data((2000, 0.4))
    @unpack
    def test_merge_segments_gain(self, size: int, mrp: float):
        print('testing segment merge (gain).')
        x = np.random.normal(size=size)
        # test no segments
        chpts0M = [0, size]
        segments = chpts0M_to_segments(chpts0M)
        segments_ = merge_segments_gain(
            values=x, segments=segments, min_ref_percent=mrp)
        self.assertEqual( len(np.unique(segments_)), 1 )
        # test many segments with one small segment
        # significance tests don't work for small segments i.e. length 1.
        chpts0M = [0, 10, 200, 400, 450, 1000, 1300, 1301, size]
        segments = chpts0M_to_segments(chpts0M)
        segments_ = merge_segments_gain(
            values=x, segments=segments, min_ref_percent=mrp)
        u, n = np.unique(segments_, return_counts=True)
        self.assertLessEqual( len(u), len(np.unique(segments)) )
        self.assertTrue( any(n >= size * mrp) )

    @data(2000)
    def test_merge_segments_quant(self, size: int):
        print('testing segment merge (quantiles).')
        x = [np.random.normal(size=size) for _ in range(4)]
        # test no segments
        chpts0M = [0, size]
        segments = chpts0M_to_segments(chpts0M)
        segments_ = merge_segments_quant(vlist=x, segments=segments)
        self.assertEqual( len(np.unique(segments_)), 1 )
        # test many segments with one small segment
        # significance tests don't work for small segments i.e. length 1.
        chpts0M = [0, 10, 200, 400, 450, 1000, 1300, 1301, size]
        segments = chpts0M_to_segments(chpts0M)
        segments_ = merge_segments_quant(vlist=x, segments=segments)
        self.assertLessEqual( len(np.unique(segments_)), len(np.unique(segments)) )

    @data(2000)
    def test_refine_reference_segment(self, size: int):
        print('testing segment refinement.')
        x = [np.random.normal(size=size) for _ in range(4)]
        # test no segments
        segments = np.zeros((size))
        segments_ = refine_reference_segment(
            vlist=x, segments=segments, rl=0)
        self.assertEqual( len(np.unique(segments_)), 1 )
        # test many segments with one small segment
        # significance tests don't work for small segments i.e. length 1.
        # segments must all be > 5 in size.
        chpts0M = [0, 10, 35, 75, 140, 400, 450, 1000, 1300, 1305, size]
        segments = chpts0M_to_segments(chpts0M)
        segments[140:400] = 0
        segments[450:1000] = 0
        segments[1305:] = 0
        segments_ = refine_reference_segment(
            vlist=x, segments=copy.copy(segments), rl=0)
        u, n = np.unique(segments_, return_counts=True)
        self.assertLessEqual( len(u), len(np.unique(segments)) )
        self.assertEqual( np.max(n), n[0] )

    @data(2000)
    def test_refine_reference_segment_inbetween(self, size: int):
        print('testing segment refinement (inbetween).')
        # test no segments
        segments = np.zeros((size))
        segments_ = refine_reference_segment_inbetween(segments=segments, rl=0)
        self.assertEqual( len(np.unique(segments_)), 1 )
        # test many segments with one small segment
        # significance tests don't work for small segments i.e. length 1.
        chpts0M = [0, 10, 35, 75, 140, 400, 450, 1000, 1300, 1301, size]
        segments = chpts0M_to_segments(chpts0M)
        segments[140:400] = 0
        segments[450:1000] = 0
        segments[1301:] = 0
        segments_ = refine_reference_segment_inbetween(
            segments=copy.copy(segments), rl=0)
        u_, n = np.unique(segments_, return_counts=True)
        u = np.unique(segments)
        self.assertTrue( len(u_) == len(u) or len(u_) - 1 == len(u) )
        self.assertEqual( np.max(n), n[u_==0] )

    @data(2000)
    def test_remove_reference_ends(self, size: int):
        print('testing removal of segment ends.')
        x = np.random.normal(size=size)
        # test no segments
        segments = np.zeros((size))
        segments_ = remove_reference_ends(
            values=x, segments=segments, rl=0)
        self.assertEqual( len(np.unique(segments_)), 1 )
        # test many segments with one small segment
        # significance tests don't work for small segments i.e. length 1.
        chpts0M = [0, 10, 35, 100, 140, 400, 450, 1000, 1300, 1305, size]
        segments = chpts0M_to_segments(chpts0M)
        segments[35:100] = 0
        segments[1000:1300] = 0
        segments[1305:] = 0
        segments_ = remove_reference_ends(
            values=x, segments=segments, rl=0)
        u_, n = np.unique(segments_, return_counts=True)
        u = np.unique(segments)
        self.assertTrue( len(u_) == len(u) or len(u_) - 1 == len(u) )
        self.assertEqual( np.max(n), n[u_==0] )

if __name__ == '__main__':
    unittest.main()







