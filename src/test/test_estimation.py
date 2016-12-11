#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimate

class EstimationObjectTestCase(unittest.TestCase):
    """Test estimate.Estimation class"""

    def test_str(self):
        """Test string representation"""

        expected = str((1, 2, 3))
        assert expected == str(estimate.Estimation(numbers = [1, 2, 3]))

    def test_creation(self):
        """Test __init__ method"""

        expected = str((1, 2, 3))
        assert expected == str(estimate.Estimation(numbers = [1, 2, 3]))
        assert expected == str(estimate.Estimation(numbers = (1, 2, 3)))
        assert expected == str(estimate.Estimation(numbers = [1, 2, 3, 4]))

    def test_getitem(self):
        """Test an access of component"""

        e = estimate.Estimation(numbers = [1, 2, 3])
        assert (1, 2, 3) == (e[0], e[1], e[2])

    def test_add(self):
        """Test aggregation"""

        def _test(init, addon, expected):
            result = init + addon
            assert tuple(expected) == (result[0], result[1], result[2])

        _test(
            init = estimate.Estimation(numbers = [1, 2, 3]),
            addon = (4, 5, 6),
            expected=(5, 7, 9)
        )
        _test(
            init = estimate.Estimation(numbers = [1, 2, 3]),
            addon = [4, 5, 6],
            expected=(5, 7, 9)
        )
        _test(
            init = estimate.Estimation(numbers = [1, 2, 3]),
            addon = estimate.Estimation(numbers = [4, 5, 6]),
            expected=(5, 7, 9)
        )

