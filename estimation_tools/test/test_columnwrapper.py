#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import string
import estimation_tools.estimate as estimate

class ColumnWrapperTestCase(unittest.TestCase):
    """Test estimate.ColumnWrapper class"""

    def test_name_and_index_alphabet_0(self):
        """Test name and index works well"""

        for i, a in enumerate(string.uppercase):
            self.assertEqual( estimate.ColumnWrapper._index(a), i )
            self.assertEqual( estimate.ColumnWrapper._name(i), a )

    def test_name_and_index_complex(self):
        """Test name and index works well"""

        self.assertEqual( estimate.ColumnWrapper._index('A'), 0 )
        self.assertEqual( estimate.ColumnWrapper._name(0), 'A' )

        self.assertEqual( estimate.ColumnWrapper._index('Z'), 25 )
        self.assertEqual( estimate.ColumnWrapper._name(25), 'Z' )

        self.assertEqual( estimate.ColumnWrapper._index('AA'), 26 )
        self.assertEqual( estimate.ColumnWrapper._name(26), 'AA' )


    def test_name_in_index_alphabet(self):
        """Test name and index works well"""

        for a in string.uppercase:
            idx = estimate.ColumnWrapper._index(a)
            b = estimate.ColumnWrapper._name(idx)
            self.assertEqual( b, a )

    def test_name_in_index_complex(self):
        """Test name and index works well"""

        for a in ['Z', 'AA', 'AZ', 'BA']:
            idx = estimate.ColumnWrapper._index(a)
            b = estimate.ColumnWrapper._name(idx)
            self.assertEqual( b, a )

