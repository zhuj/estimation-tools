#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimate

class FormatCacheObjectTestCase(unittest.TestCase):
    """Test estimate.FormatCache class"""

    def test_register_call(self):
        """Test register calls"""

        class C: _count = 0
        def _register(format):
            C._count += 1
            return format

        t = estimate.Theme(obj = object())
        c = estimate.FormatCache(theme = t, register = _register)

        # initially we have no calls
        assert C._count == 0

        # first call after empty options
        assert c.get() == t.F_DEFAULT
        assert C._count == 1

        # no call (the value has already been cached)
        assert c.get({}) == t.F_DEFAULT
        assert C._count == 1

        # 2nd call - we have new key
        assert c.get({'a':1}) == t.format({'a':1})
        assert C._count == 2

        # the same key - no extra call
        assert c.get({'a':1}) == t.format({'a':1})
        assert C._count == 2

        # the same key - no extra call
        assert c.get({'a':1}, {'b':2}) == t.format({'a':1, 'b':2})
        assert C._count == 3

