#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimate

class ThemeObjectTestCase(unittest.TestCase):
    """Test estimate.Theme class"""

    def test_getattr_empty_theme(self):
        """Test getattr with empty object"""
        t = estimate.Theme(obj = object())
        assert t.F_DEFAULT is estimate.Theme.DEFAULT_F_DEFAULT

    def test_getattr_nonempty_theme(self):
        """Test getattr with non-empty object"""
        class T: F_DEFAULT = { 'font_name': 'Verdana' }
        t = estimate.Theme(obj = T)
        assert t.F_DEFAULT is not estimate.Theme.DEFAULT_F_DEFAULT
        assert t.F_DEFAULT is T.F_DEFAULT

    def test_f_default(self):
        """Test format merging with default"""
        class T: F_DEFAULT = { 'font_name': 'Verdana' }
        t = estimate.Theme(obj = T)
        assert t.format(t.F_DEFAULT)['font_name'] == 'Verdana'
        assert t.format(t.F_CAPTION)['font_name'] == 'Verdana'
        assert t.format(t.F_FINAL)['font_name'] == 'Verdana'

    def test_sections_colors(self):
        """Test section colors"""
        class T:
            SECTION_1 = '#010101'
            SECTION_2 = '#020202'
            SECTION_3 = '#030303'
            SECTION_4 = '#040404'
            SECTION_5 = '#050505'

        t = estimate.Theme(obj = T)
        assert t.format(t.F_SECTION_1)['bg_color'] == '#010101'
        assert t.format(t.F_SECTION_2)['bg_color'] == '#020202'
        assert t.format(t.F_SECTION_3)['bg_color'] == '#030303'
        assert t.format(t.F_SECTION_4)['bg_color'] == '#040404'
        assert t.format(t.F_SECTION_5)['bg_color'] == '#050505'

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
        assert t.format(t.F_SECTION_1).get('bg_color', x) is x
        assert t.format(t.F_SECTION_2).get('bg_color', x) is x
        assert t.format(t.F_SECTION_3).get('bg_color', x) is x
        assert t.format(t.F_SECTION_4).get('bg_color', x) is x
        assert t.format(t.F_SECTION_5).get('bg_color', x) is x

