#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimate

class NodeObjectTestCase(unittest.TestCase):
    """Test estimate.Node class"""

    def test_role(self):
        """Test role detection by the pattern"""

        n = estimate.Node(parent=None, title="(role)")
        assert n.is_role()

        n = estimate.Node(parent=None, title="role")
        assert not n.is_role()


    def test_parent_and_level(self):
        """Test parent and level"""

        root = estimate.Node(parent=None, title=None)
        assert root.level() is -1
        assert root.parent() is None

        n0 = estimate.Node(parent=root, title="")
        assert n0.level() is 0
        assert n0.parent() is root

        n1 = estimate.Node(parent=n0,   title="")
        assert n1.level() is 1
        assert n1.parent() is n0
        assert n1.parent(0) is n0
        assert n1.parent(1) is n1

        n2 = estimate.Node(parent=n1,   title="")
        assert n2.level() is 2
        assert n2.parent() is n1
        assert n2.parent(0) is n0
        assert n2.parent(1) is n1
        assert n2.parent(2) is n2

        n3 = estimate.Node(parent=n2,   title="")
        assert n3.level() is 3
        assert n3.parent() is n2
        assert n3.parent(0) is n0
        assert n3.parent(1) is n1
        assert n3.parent(2) is n2
        assert n3.parent(3) is n3

    def test_parent_and_childs(self):
        """Test relations between parent and childs"""

        root = estimate.Node(parent=None, title=None)
        assert root.parent() is None
        assert len(root.childs()) is 0

        n0 = estimate.Node(parent=root, title="")
        assert n0.parent() is root
        assert len(root.childs()) is 0  # still empty (no autoappend)

        root.append(n0)
        assert len(root.childs()) is 1
        assert root.childs()[0] is n0

    def test_estimations_no_role(self):
        """Test estimation (no role)"""

        root = estimate.Node(parent=None, title=None)
        n0 = estimate.Node(parent=root, title="")

        n0.estimate(role=None, numbers=(1, 2, 3))
        n0.estimate(role=None, numbers=(4, 5, 6))

        assert n0.parent().estimates() is None # parent node shouldn't be affected

        e = n0.estimates()
        assert e is not None
        assert (5, 7, 9) == (e[0], e[1], e[2])

    def test_estimations_no_role(self):
        """Test estimation (different roles)"""

        root = estimate.Node(parent=None, title=None)
        n0 = estimate.Node(parent=root, title="")

        n0.estimate(role="role1", numbers=(1, 2, 3))
        n0.estimate(role="role2", numbers=(4, 5, 6))

        assert n0.parent().estimates() is None # parent node shouldn't be affected

        e = n0.estimates()
        assert e is not None
        assert (5, 7, 9) == (e[0], e[1], e[2])

    def test_annotation(self):
        """Test annotation (comments)"""

        root = estimate.Node(parent=None, title=None)
        n0 = estimate.Node(parent=root, title="")

        n0.annotation(name="todo", value="v1")
        assert n0.annotation(name="todo") == [ "v1" ]

        n0.annotation(name="impl", value="v2")
        assert n0.annotation(name="todo") == [ "v1" ]
        assert n0.annotation(name="impl") == [ "v2" ]

        n0.annotation(name="impl", value="v3")
        assert n0.annotation(name="impl") == [ "v2", "v3" ]

        assert len(root.annotation(name="todo")) == 0
        assert len(root.annotation(name="impl")) == 0

