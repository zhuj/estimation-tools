#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimation_tools.estimate as estimate

class ThemeObjectTestCase(unittest.TestCase):
    """Test estimate.Theme class"""

    def test_getattr_empty_theme(self):
        """Test getattr with empty object"""
        t = estimate.Theme(obj = object())
        self.assertIs( t.F_DEFAULT, estimate.Theme.DEFAULT_F_DEFAULT )

    def test_getattr_nonempty_theme(self):
        """Test getattr with non-empty object"""
        class T: F_DEFAULT = { 'font_name': 'Verdana' }
        t = estimate.Theme(obj = T)
        self.assertIsNot( t.F_DEFAULT, estimate.Theme.DEFAULT_F_DEFAULT )
        self.assertIs( t.F_DEFAULT, T.F_DEFAULT )

    def test_f_default(self):
        """Test format merging with default"""
        class T: F_DEFAULT = { 'font_name': 'Verdana' }
        t = estimate.Theme(obj = T)
        self.assertEqual( t.format(t.F_DEFAULT)['font_name'], 'Verdana' )
        self.assertEqual( t.format(t.F_CAPTION)['font_name'], 'Verdana' )
        self.assertEqual( t.format(t.F_FINAL)['font_name'], 'Verdana' )

    def test_sections_colors(self):
        """Test section colors"""
        class T:
            SECTION_1 = '#010101'
            SECTION_2 = '#020202'
            SECTION_3 = '#030303'
            SECTION_4 = '#040404'
            SECTION_5 = '#050505'

        t = estimate.Theme(obj = T)
        self.assertEqual( t.format(t.F_SECTION_1)['fill_color'], 'FF010101' )
        self.assertEqual( t.format(t.F_SECTION_2)['fill_color'], 'FF020202' )
        self.assertEqual( t.format(t.F_SECTION_3)['fill_color'], 'FF030303' )
        self.assertEqual( t.format(t.F_SECTION_4)['fill_color'], 'FF040404' )
        self.assertEqual( t.format(t.F_SECTION_5)['fill_color'], 'FF050505' )

    def test_sections_clear(self):
        """Test section color clears"""
        class T:
            SECTION_1 = None
            SECTION_2 = None
            SECTION_3 = None
            SECTION_4 = None
            SECTION_5 = None

        t = estimate.Theme(obj = T)
        x = object()
        self.assertIs( t.format(t.F_SECTION_1).get('fill_color', x), x )
        self.assertIs( t.format(t.F_SECTION_2).get('fill_color', x), x )
        self.assertIs( t.format(t.F_SECTION_3).get('fill_color', x), x )
        self.assertIs( t.format(t.F_SECTION_4).get('fill_color', x), x )
        self.assertIs( t.format(t.F_SECTION_5).get('fill_color', x), x )

