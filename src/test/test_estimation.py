#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimate

class EstimationObjectTestCase(unittest.TestCase):

    def test_basic_creation(self):
        """ """
        expected = str((1, 2, 3))
        assert expected == str(estimate.Estimation(numbers = [1, 2, 3]))
        assert expected == str(estimate.Estimation(numbers = (1, 2, 3)))
        assert expected == str(estimate.Estimation(numbers = [1, 2, 3, 4]))

    def test_basic_getitem(self):
        """ """
        e = estimate.Estimation(numbers = [1, 2, 3])
        assert (1, 2, 3) == (e[0], e[1], e[2])

    def test_add(self):
        """ """
        e1 = estimate.Estimation(numbers = [1, 2, 3])
        e2 = estimate.Estimation(numbers = [4, 5, 6])
        e3 = e1 + e2
        assert (5, 7, 9) == (e3[0], e3[1], e3[2])

