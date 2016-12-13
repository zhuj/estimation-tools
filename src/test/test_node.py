#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimate

class NodeObjectTestCase(unittest.TestCase):
    """Test estimate.Node class"""

    def test_role(self):
        """Test role detection by the pattern"""

        n = estimate.Node(parent=None, title="(role)")
        self.assertEqual( n.is_role(), True )

        n = estimate.Node(parent=None, title="role")
        self.assertEqual( n.is_role(), False )


    def test_parent_and_level(self):
        """Test parent and level"""

        root = estimate.Node(parent=None, title=None)
        self.assertEqual( root.level(), -1 )
        self.assertIs( root.parent(), None )

        n0 = estimate.Node(parent=root, title="")
        self.assertEqual( n0.level(), 0 )
        self.assertIs( n0.parent(), root )

        n1 = estimate.Node(parent=n0,   title="")
        self.assertEqual( n1.level(), 1 )
        self.assertIs( n1.parent(), n0 )
        self.assertIs( n1.parent(0), n0 )
        self.assertIs( n1.parent(1), n1 )

        n2 = estimate.Node(parent=n1,   title="")
        self.assertEqual( n2.level(), 2 )
        self.assertIs( n2.parent(), n1 )
        self.assertIs( n2.parent(0), n0 )
        self.assertIs( n2.parent(1), n1 )
        self.assertIs( n2.parent(2), n2 )

        n3 = estimate.Node(parent=n2,   title="")
        self.assertEqual( n3.level(), 3 )
        self.assertIs( n3.parent(), n2 )
        self.assertIs( n3.parent(0), n0 )
        self.assertIs( n3.parent(1), n1 )
        self.assertIs( n3.parent(2), n2 )
        self.assertIs( n3.parent(3), n3 )

    def test_parent_and_childs(self):
        """Test relations between parent and childs"""

        root = estimate.Node(parent=None, title=None)
        self.assertIs( root.parent(), None )
        self.assertEqual( len(root.childs()), 0 )

        n0 = estimate.Node(parent=root, title="")
        self.assertIs( n0.parent(), root )
        self.assertEqual( len(root.childs()), 0 )  # still empty (no autoappend)

        root.append(n0)
        self.assertEqual( len(root.childs()), 1 )
        self.assertIs( root.childs()[0], n0 )

    def test_estimations_no_role(self):
        """Test estimation (no role)"""

        root = estimate.Node(parent=None, title=None)
        n0 = estimate.Node(parent=root, title="")

        n0.estimate(role=None, numbers=(1, 2, 3))
        n0.estimate(role=None, numbers=(4, 5, 6))

        self.assertEqual( n0.parent().estimates(), None ) # parent node shouldn't be affected

        e = n0.estimates()
        self.assertIsNot( e, None )
        self.assertEqual( (e[0], e[1], e[2]), (5, 7, 9) )

    def test_estimations_roles(self):
        """Test estimation (different roles)"""

        root = estimate.Node(parent=None, title=None)
        n0 = estimate.Node(parent=root, title="")

        n0.estimate(role="role1", numbers=(1, 2, 3))
        n0.estimate(role="role2", numbers=(4, 5, 6))

        self.assertIs( n0.parent().estimates(), None ) # parent node shouldn't be affected

        e = n0.estimates()
        self.assertIsNot( e, None )
        self.assertEqual( (e[0], e[1], e[2]), (5, 7, 9) )

    def test_annotation(self):
        """Test annotation (comments)"""

        root = estimate.Node(parent=None, title=None)
        n0 = estimate.Node(parent=root, title="")

        n0.annotation(name="todo", value="v1")
        self.assertEqual( n0.annotation(name="todo"), [ "v1" ] )

        n0.annotation(name="impl", value="v2")
        self.assertEqual( n0.annotation(name="todo"), [ "v1" ] )
        self.assertEqual( n0.annotation(name="impl"), [ "v2" ] )

        n0.annotation(name="impl", value="v3")
        self.assertEqual( n0.annotation(name="impl"), [ "v2", "v3" ] )

        self.assertEqual( len(root.annotation(name="todo")), 0 )
        self.assertEqual( len(root.annotation(name="impl")), 0 )

