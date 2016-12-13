#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimation_tools.estimate as estimate

class FormatCacheObjectTestCase(unittest.TestCase):
    """Test estimate.FormatCache class"""

    def test_register_call(self):
        """Test register calls"""

        class C: _count = 0
        def _register(format):
            C._count += 1
            return format

        t = estimate.Theme(obj = object())
        c = estimate.FormatCache(format = t.format, register = _register)

        # initially we have no calls
        self.assertEqual( C._count, 0 )

        # first call after empty options
        self.assertEqual( c.get(), t.F_DEFAULT )
        self.assertEqual( C._count, 1 )

        # no call (the value has already been cached)
        self.assertEqual( c.get({}), t.F_DEFAULT )
        self.assertEqual( C._count, 1 )

        # 2nd call - we have new key
        self.assertEqual( c.get({'a':1}), t.format({'a':1}) )
        self.assertEqual( C._count, 2 )

        # the same key - no extra call
        self.assertEqual( c.get({'a':1}), t.format({'a':1}) )
        self.assertEqual( C._count, 2 )

        # the same key - no extra call
        self.assertEqual( c.get({'a':1}, {'b':2}), t.format({'a':1, 'b':2}) )
        self.assertEqual( C._count, 3 )

