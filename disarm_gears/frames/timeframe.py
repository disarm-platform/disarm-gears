from datetime import timedelta, datetime
import calendar
import pandas as pd
import numpy as np
import warnings
from disarm_gears.validators import validate_1d_array


class Timeframe:

    def __init__(self, start, length, by='day', step=1, end=None, directive='%Y-%m-%d'):
        '''
        Define a timeframe for a model.

        :param start: Initial date of the timeframe.
                      Date encoded as: integer, float, string or datetime.
                      Can be a None value if end is specified.
        :param length: Length of the timeframe.
                       Positive integer.
        :param by: Units of time in which the frame is measured.
                   String, one of 'day', 'month', 'year'.
        :param step: Number of units per time point.
                     Positive integer.
        :param end: Last date of the timeframe (optional and only used if start is None).
                    Date encoded as: integer, float, string or datetime.
        :param directive: Date format representation (default '%Y-%m-%d').
                          String.
        '''
        assert isinstance(length, int) and length > 0, 'length must be a positive integer.'
        assert isinstance(step, int) and step > 0, 'step must be a positive integer.'
        assert isinstance(by, str), 'by has to be a str object.'
        if start is not None:
            #if end is not None:
            #    warnings.warn('When start and end are both not None,'
            #                          'end is overwritten according to start and length.')
            start = datetime.strptime(start, directive)
            if by == 'day':
                diff_ = length * step
                init_date = [start + timedelta(days=i) for i in range(0, diff_, step)]
                end = init_date[-1] + timedelta(days=step - 1)
            elif by == 'month':
                diff_ = (1 + length) * step
                year_ = np.array([start.year + int((start.month + i)/12) for i in range(0, diff_, step)])
                month_ = np.array([(start.month + i) % 12 for i in range(0, diff_, step)])
                month_[month_ == 0] = 12
                day_ = np.array([min(start.day, calendar.monthrange(i, j)[1]) for i,j in zip(year_, month_)])
                init_date = [start.replace(year=i, month=j, day=k) for i,j,k in zip(year_, month_, day_)]
                end = init_date[-1] - timedelta(days=1)
                init_date = init_date[:-1]
            elif by == 'year':
                diff_ = (1 + length) * step
                init_date = [start.replace(year=start.year + i) for i in range(0, diff_, step)]
                end = init_date[-1] - timedelta(days=1)
                init_date = init_date[:-1]
            else:
                raise ValueError('by parameter was not understood.')
        else:
            assert end is not None, 'end date has to be specified if start date is None'
            end = datetime.strptime(end, directive)
            if by == 'day':
                diff_ = length * step
                e_ = end - timedelta(days=step - 1)
                init_date = [e_ - timedelta(days=i) for i in range(0, diff_, step)][::-1]
            elif by == 'month':
                diff_ = (1 + length) * step
                e_ = end + timedelta(days=1)
                year_ = np.array([e_.year - int((e_.month + i)/12) for i in range(0, diff_, step)[::-1]])
                month_ = np.array([(e_.month - i) % 12 for i in range(0, diff_, step)[::-1]])
                month_[month_ == 0] = 12
                day_ = np.array([min(e_.day, calendar.monthrange(i, j)[1]) for i,j in zip(year_, month_)])
                init_date = [e_.replace(year=i, month=j, day=k) for i,j,k in zip(year_, month_, day_)]
                init_date = init_date[:-1]
            elif by == 'year':
                diff_ = (1 + length) * step
                e_ = end + timedelta(days=1)
                init_date = [e_.replace(year=e_.year - i) for i in range(0, diff_, step)[::-1]]
                init_date = init_date[:-1]
            else:
                raise ValueError('by parameter was not understood.')
        # Store values
        self.start = init_date[0]
        self.end = end
        self.by = by
        self.length = len(init_date)
        #self.knots_info = pd.DataFrame({'knot': np.arange(self.length), 'init_date': init_date})
        self.knots_info = pd.DataFrame({'knot': np.arange(self.length),
                                        'init_date': init_date,
                                        'tag': [datetime.strftime(i, directive) for i in init_date]})


    def which_knot(self, x):
        '''
        Identify the knot (point within the timeframe) to which a date x belongs.

        :param x: Array of dates.
                  Numpy array, shape [n, ]. Elements may be of type: integer, float, string, datetime.
        :return: Array with knots per date (-1 for dates outside the timeframe).
                 Numpy array of integers.
        '''
        validate_1d_array(x)
        _x = pd.to_datetime(x)
        #assert isinstance(x, pd.Series) or all(isinstance(xi, pd.Timestamp) for xi in x), 'Not a pandas date format.'
        knots = np.repeat(None, _x.size)
        for a,b,k in zip(self.knots_info.init_date[:-1], self.knots_info.init_date[1:],
                         self.knots_info.knot[:-1]):
            knots[np.logical_and(a <= _x, _x < b)] = int(k)
        knots[np.logical_and(b <= _x, _x <= self.end)] = int(k + 1)

        return np.array([-1 if ki is None else ki for ki in knots])

