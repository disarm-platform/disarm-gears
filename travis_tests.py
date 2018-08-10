#!/usr/bin/env python

import nose, warnings
with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nose.main('disarm_gears', defaultTest='disarm_gears/testing/', argv=[''])

