import unittest
import numpy as np
import pandas as pd
from disarm_gears.frames import Timeframe


# Inputs

start_date = '2000-06-15'
end_date = '2010-06-20'

class TimeframeTests(unittest.TestCase):

    def test_inputs(self):
        # Check bad inputs
        self.assertRaises(AssertionError, Timeframe, start=start_date, length=0)
        self.assertRaises(AssertionError, Timeframe, start=start_date, length=3, step=.5)
        self.assertRaises(ValueError, Timeframe, start=start_date, length=3, step=1, by='x')

        self.assertRaises(AssertionError, Timeframe, start=None, end=end_date, length=0)
        self.assertRaises(AssertionError, Timeframe, start=None, end=end_date, length=3, step=.5)
        self.assertRaises(ValueError, Timeframe, start=None, end=end_date, length=3, step=1, by='x')


    def test_outputs(self):

        # By day, length = 1
        tf_00 = Timeframe(start=start_date, length=1, by='day')
        self.assertIsInstance(tf_00.knots_info, pd.DataFrame)
        self.assertEqual(tf_00.knots_info.shape[0], 1)
        #self.assertEqual(tf_00.knots_info.shape[1], 2)#TODO three columns if we add tag
        self.assertEqual(tf_00.start, tf_00.end)

        tf_01 = Timeframe(start=None, end=end_date, length=1, by='day')
        self.assertIsInstance(tf_01.knots_info, pd.DataFrame)
        self.assertEqual(tf_01.knots_info.shape[0], 1)
        #self.assertEqual(tf_01.knots_info.shape[1], 2)
        self.assertEqual(tf_01.start, tf_01.end)

        # By day, length = 2
        tf_1 = Timeframe(start=start_date, length=2, by='day')
        self.assertIsInstance(tf_1.knots_info, pd.DataFrame)
        self.assertEqual(tf_1.knots_info.shape[0], 2)
        #self.assertEqual(tf_1.knots_info.shape[1], 2)
        self.assertEqual((tf_1.end - tf_1.start).days, 1)

        # By day, length = 1, step = 2
        tf_2 = Timeframe(start=start_date, length=1, step=2, by='day')
        self.assertIsInstance(tf_2.knots_info, pd.DataFrame)
        self.assertEqual(tf_2.knots_info.shape[0], 1)
        #self.assertEqual(tf_2.knots_info.shape[1], 2)
        self.assertEqual((tf_2.end - tf_2.start).days, 1)

        # By month
        tf_30 = Timeframe(start=start_date, length=3, step=1, by='month')
        self.assertEqual(tf_30.knots_info.knot[0], 0)
        self.assertEqual(tf_30.knots_info.knot[2], 2)
        self.assertEqual(tf_30.knots_info.shape[0], 3)
        self.assertEqual(tf_30.knots_info.init_date[1], pd.to_datetime('2000-07-15'))
        self.assertEqual(tf_30.end, pd.to_datetime('2000-09-14'))

        tf_31 = Timeframe(start=None, end=end_date, length=3, step=1, by='month')
        self.assertEqual(tf_31.knots_info.knot[0], 0)
        self.assertEqual(tf_31.knots_info.knot[2], 2)
        self.assertEqual(tf_31.knots_info.shape[0], 3)


        # By year, step = 2
        tf_4 = Timeframe(start=start_date, length=5, step=2, by='year')
        self.assertIsInstance(tf_4.knots_info, pd.DataFrame)
        self.assertEqual(tf_4.knots_info.shape[0], 5)
        self.assertEqual(tf_4.knots_info.init_date[3], pd.to_datetime('2006-06-15'))
        self.assertEqual(tf_4.end, pd.to_datetime('2010-06-14'))

        # By year, step = 2, start and end non None
        tf_5 = Timeframe(start=start_date, length=5, step=2, by='year', end=end_date)
        self.assertIsInstance(tf_5.knots_info, pd.DataFrame)
        self.assertEqual(tf_5.knots_info.shape[0], 5)
        self.assertEqual(tf_5.knots_info.init_date[3], pd.to_datetime('2006-06-15'))
        self.assertEqual(tf_5.end, pd.to_datetime('2010-06-14'))

        # By year, step = 2, start = None
        tf_6 = Timeframe(start=None, length=5, step=2, by='year', end=end_date)
        self.assertIsInstance(tf_6.knots_info, pd.DataFrame)
        self.assertEqual(tf_6.knots_info.shape[0], 5)
        self.assertEqual(tf_6.knots_info.init_date[3], pd.to_datetime('2006-06-21'))
        self.assertEqual(tf_6.start, pd.to_datetime('2000-06-21'))
        self.assertEqual(tf_6.end, pd.to_datetime(end_date))


    def test_which_knots(self):
        tf_6 = Timeframe(start=start_date, length=10, step=7, by='day')
        dates = np.array(['2000-06-22', '2000-07-12', '2000-01-01', '2002-12-31', '2000-08-23', '2000-08-24'])
        ix = tf_6.which_knot(dates)
        self.assertEqual(ix.size, dates.size)
        self.assertIsInstance(ix, np.ndarray)
        self.assertEqual(ix[0], 1)
        self.assertEqual(ix[1], 3)
        self.assertEqual(ix[2], -1)
        self.assertEqual(ix[3], -1)
        self.assertEqual(ix[4], 9)
        self.assertEqual(ix[5], -1)

