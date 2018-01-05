#!/usr/bin/env python

import nose
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
nose.main('regain', defaultTest='regain/tests/', argv=[''])
