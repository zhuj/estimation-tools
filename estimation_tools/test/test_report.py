#!/usr/bin/env python2.7
# -*- coding: utf-8 -*-

import unittest
import estimation_tools.estimate as estimate

def _val(ws, cell):
    v = ws[cell].value
    return (v is not None) and str(v) or ""

class ReportTestCase(unittest.TestCase):
    """Test reports itselves"""

    @staticmethod
    def _report(root, options={}):
        wb = estimate.Processor(options).report(root, filename=None)
        names = wb.get_sheet_names()
        return wb[names[0]]

    def _check_header(self, ws):
        self.assertEqual( (ws.min_column, ws.min_row), (1, 1) )
        self.assertEqual( _val(ws, 'A1'), 'Task / Subtask' )
        self.assertEqual( _val(ws, 'B1'), 'Filter' )
        if ( _val(ws, 'C1') != 'MVP' ): self.assertEqual( _val(ws, 'C1'), '' )
        self.assertEqual( _val(ws, 'D1'), 'Comment' )
        self.assertEqual( _val(ws, 'E1'), 'Min' )
        self.assertEqual( _val(ws, 'F1'), 'Real' )
        self.assertEqual( _val(ws, 'G1'), 'Max' )
        self.assertEqual( _val(ws, 'H1'), '' )
        self.assertEqual( _val(ws, 'I1'), 'Avg' )
        self.assertEqual( _val(ws, 'J1'), 'SD' )
        self.assertEqual( _val(ws, 'K1'), 'Sq' )
        return 2

    def _check_footer(self, ws):
        row = ws.max_row

        # max
        self.assertTrue( row > 1 )
        self.assertTrue( _val(ws, 'A%s' % row).startswith('Max') )
        row -= 1

        # min
        self.assertTrue( row > 1 )
        self.assertTrue( _val(ws, 'A%s' % row).startswith('Min') )
        row -= 1

        # kappa
        self.assertTrue( row > 1 )
        self.assertTrue( _val(ws, 'A%s' % row).startswith('K') )
        row -= 1

        # standard deviation
        self.assertTrue( row > 1 )
        self.assertTrue( _val(ws, 'A%s' % row).startswith('Standard deviation') )
        row -= 1

        # empty
        self.assertTrue( row > 1 )
        self.assertTrue( _val(ws, 'A%s' % row) == '' )
        row -= 1

        # total (mvp), optional
        self.assertTrue( row > 1 )
        if ( _val(ws, 'A%s' % row).startswith('Total (MVP)') ):
            row -= 1

            # separator
            self.assertTrue( row > 1 )
            self.assertTrue( _val(ws, 'A%s' % row) == '' )
            row -= 1

        # roles (if exists)
        self.assertTrue( row > 1 )
        while _val(ws, 'A%s' % row).strip().startswith('- '):
            row -= 1
            self.assertTrue( row > 1 )

        # total
        self.assertTrue( row > 1 )
        self.assertTrue( _val(ws, 'A%s' % row).startswith('Total') )
        row -= 1

        # empty
        self.assertTrue( row > 1 )
        self.assertTrue( _val(ws, 'A%s' % row) == '' )
        row -= 1

        # return it
        return row

    def _check_header_and_footer(self, ws):
        data_min = self._check_header(ws)
        data_max = self._check_footer(ws)
        self.assertTrue( data_max > data_min )
        return (data_min, data_max)


    def test_empty_mindmap(self):
        """just check it doesn't throw an exception"""

        root = estimate.Node(parent=None, title=None)
        ReportTestCase._report(root)

    def test_no_estimates(self):
        """test for simple case without estimates"""

        root = estimate.Node(parent=None, title=None)
        root.append(estimate.Node(parent=root, title="1st node with no estimate"))
        root.append(estimate.Node(parent=root, title="2nd node with no estimate"))
        root.append(estimate.Node(parent=root, title="3rd node with no estimate"))

        ws = ReportTestCase._report(root)

        data_min, data_max = self._check_header_and_footer(ws)
        self.assertEqual( data_min, 2 )
        self.assertEqual( data_max, 4 )

        self.assertEqual( _val(ws, 'A2').strip(), "1st node with no estimate" )
        self.assertEqual( _val(ws, 'A3').strip(), "2nd node with no estimate" )
        self.assertEqual( _val(ws, 'A4').strip(), "3rd node with no estimate" )

    def test_estimates_sorting(self):
        """estimation roles should be in the end of the child list for the node"""

        root = estimate.Node(parent=None, title=None)

        n1 = estimate.Node(parent=root, title="1st node")
        n1.estimate(role=None, numbers=(0,0,0))
        root.append(n1)

        n2 = estimate.Node(parent=root, title="2nd node")
        root.append(n2)

        n3 = estimate.Node(parent=root, title="3rd node")
        root.append(n3)

        ws = ReportTestCase._report(root)

        data_min, data_max = self._check_header_and_footer(ws)
        self.assertEqual( data_min, 2 )
        self.assertEqual( data_max, 4 )

        self.assertEqual( _val(ws, 'A2').strip(), "2nd node" )
        self.assertEqual( _val(ws, 'A3').strip(), "3rd node" )
        self.assertEqual( _val(ws, 'A4').strip(), "1st node" ) # estimation rows should be in the end

    def _mk_root(self):
        """ it creates a complex node tree for the following testing """

        class _node:
            def __init__(self, parent=None, title=""):
                if (isinstance(parent, _node)):
                    parent = parent._n

                self._n = estimate.Node(
                    parent = parent,
                    title = title
                )

                if (parent is not None):
                    parent.append(self._n)

            def estimate(self, numbers):
                assert self._n.is_role()
                assert self._n.parent() is not None
                self._n.estimate(None, numbers)
                self._n.parent().estimate(self._n.title(), numbers)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        with _node(parent=None, title=None) as root:

            with _node(parent=root, title="C. 1st node") as n1:

                with _node(parent=n1, title="ZA. 1st subnode") as n1_1:

                    with _node(parent=n1_1, title="subnode A") as n1_1a:
                        with _node(parent=n1_1a, title="(role1)") as role: role.estimate((1,2,3))
                        with _node(parent=n1_1a, title="(role2)") as role: role.estimate((2,3,4))
                        with _node(parent=n1_1a, title="(role3)") as role: role.estimate((3,4,5))

                    with _node(parent=n1_1, title="subnode B") as n1_1b:
                        with _node(parent=n1_1b, title="(role2)") as role: role.estimate((1,2,3))
                        with _node(parent=n1_1b, title="(role3)") as role: role.estimate((2,3,4))
                        with _node(parent=n1_1b, title="(role4)") as role: role.estimate((3,4,5))

                with _node(parent=n1, title="YA. 2nd subnode") as n1_2:

                    with _node(parent=n1_2, title="subnode A") as n1_2a:
                        with _node(parent=n1_2a, title="(role3)") as role: role.estimate((1,2,3))
                        with _node(parent=n1_2a, title="(role4)") as role: role.estimate((2,3,4))
                        with _node(parent=n1_2a, title="(role5)") as role: role.estimate((3,4,5))

                    with _node(parent=n1_2, title="subnode B") as n1_2b:
                        with _node(parent=n1_2b, title="(role4)") as role: role.estimate((1,2,3))
                        with _node(parent=n1_2b, title="(role5)") as role: role.estimate((2,3,4))
                        with _node(parent=n1_2b, title="(role1)") as role: role.estimate((3,4,5))

                with _node(parent=n1, title="XA. 3rd subnode") as n1_3:

                    with _node(parent=n1_3, title="subnode A") as n1_3a:
                        with _node(parent=n1_3a, title="(role5)") as role: role.estimate((1,2,3))
                        with _node(parent=n1_3a, title="(role1)") as role: role.estimate((2,3,4))
                        with _node(parent=n1_3a, title="(role2)") as role: role.estimate((3,4,5))

                    with _node(parent=n1_3, title="subnode B") as n1_3b:
                        with _node(parent=n1_3b, title="(role1)") as role: role.estimate((1,2,3))
                        with _node(parent=n1_3b, title="(role2)") as role: role.estimate((2,3,4))
                        with _node(parent=n1_3b, title="(role3)") as role: role.estimate((3,4,5))

            with _node(parent=root, title="B. 2nd node") as n2:

                with _node(parent=n2, title="BZ. 1st subnode") as n2_1:
                    with _node(parent=n2_1, title="(role21)") as role: role.estimate((1,2,3))

                with _node(parent=n2, title="BY. 2nd subnode") as n2_2:
                    with _node(parent=n2_2, title="(role22)") as role: role.estimate((1,2,3))

                with _node(parent=n2, title="BX. 3rd subnode") as n2_3:
                    with _node(parent=n2_3, title="(role23)") as role: role.estimate((1,2,3))

            with _node(parent=root, title="A. 3rd node") as n3:

                with _node(parent=n3, title="3) 1st subnode") as n3_1:
                    with _node(parent=n3_1, title="subnode A") as n3_1a:
                        with _node(parent=n3_1a, title="(role32)") as role: role.estimate((1,2,3))
                    with _node(parent=n3_1, title="subnode B") as n3_1b:
                        with _node(parent=n3_1b, title="(role33)") as role: role.estimate((1,2,3))

                with _node(parent=n3, title="2) 2nd subnode") as n3_2:
                    with _node(parent=n3_2, title="subnode A") as n3_2a:
                        with _node(parent=n3_2a, title="(role31)") as role: role.estimate((1,2,3))
                    with _node(parent=n3_2, title="subnode B") as n3_2b:
                        with _node(parent=n3_2b, title="(role33)") as role: role.estimate((1,2,3))

                with _node(parent=n3, title="1) 3rd subnode") as n3_3:
                    with _node(parent=n3_3, title="subnode A") as n3_3a:
                        with _node(parent=n3_3a, title="(role31)") as role: role.estimate((1,2,3))
                    with _node(parent=n3_3, title="subnode B") as n3_3b:
                        with _node(parent=n3_3b, title="(role32)") as role: role.estimate((1,2,3))

            # just a strange thing with deep hierarchy too
            with _node(parent=root, title="...") as n:
                with _node(parent=n, title="...") as n:
                    with _node(parent=n, title="...") as n:
                        with _node(parent=n, title="...") as n:
                            with _node(parent=n, title="...") as n:
                                with _node(parent=n, title="...") as n:
                                    with _node(parent=n, title="...") as n:
                                        with _node(parent=n, title="...") as n:
                                            with _node(parent=n, title="...") as n:
                                                with _node(parent=n, title="...") as n:
                                                    with _node(parent=n, title="(role)") as role: role.estimate((0,0,0))
                                    with _node(parent=n, title="...") as n:
                                        with _node(parent=n, title="(role)") as role: role.estimate((0,0,0))
                            with _node(parent=n, title="...") as n:
                                with _node(parent=n, title="...") as n:
                                    with _node(parent=n, title="...") as n:
                                        with _node(parent=n, title="...") as n:
                                            with _node(parent=n, title="(role)") as role: role.estimate((0,0,0))
                                    

            return root._n

    def _test_options(self, options):
        options = dict({
            estimate.Processor.OPT_ROLES: True,
            estimate.Processor.OPT_VALIDATION: True,
        }, **options)
        ws = ReportTestCase._report(root=self._mk_root(), options=options)
        d0, d1 = self._check_header_and_footer(ws) # basic document structure test (nothing has been lost)
        # todo: implement additional checks
        return ws, d0, d1

    def test_options__p99(self):
        ws, d0, d1 = self._test_options({
            estimate.Processor.OPT_P_99: True
        })
        # todo: implement specific tests

    def test_options__p99nr(self):
        ws, d0, d1 = self._test_options({
            estimate.Processor.OPT_ROLES: False,
            estimate.Processor.OPT_P_99: True
        })
        # todo: implement specific tests

    def test_options__formulas(self):
        ws, d0, d1 = self._test_options({
            estimate.Processor.OPT_FORMULAS: True
        })
        self._test_formulas(ws, d0, d1)


    def test_options__filtering(self):
        ws, d0, d1 = self._test_options({
            estimate.Processor.OPT_FORMULAS: True,
            estimate.Processor.OPT_FILTER_VISIBILITY: True
        })
        # todo: implement specific tests

    def test_options__mvp_full(self):
        ws, d0, d1 = self._test_options({
            estimate.Processor.OPT_MVP: True,
            estimate.Processor.OPT_FORMULAS: True,
            estimate.Processor.OPT_FILTER_VISIBILITY: True
        })
        # todo: implement specific tests

    def test_options__mvp_no_validation(self):
        ws, d0, d1 = self._test_options({
            estimate.Processor.OPT_MVP: True,
            estimate.Processor.OPT_FORMULAS: True,
            estimate.Processor.OPT_FILTER_VISIBILITY: True,
            estimate.Processor.OPT_VALIDATION: False
        })
        # todo: implement specific tests

    def _test_formulas(self, ws, d0, d1):

        titles = [ (r, _val(ws, 'A%s'%r).strip()) for r in xrange(d0, 1+d1) ] # obtain titles
        titles = [ (r, t) for (r, t) in titles if not (t[0] == '(' and t[-1] == ')') ] # remove roles
        titles = [ (r, t) for (r, t) in titles if t != '...' ] # remove temporary hierarchy elements

        _assert_is_formula = lambda c, r, chk: self.assertTrue(
            chk(ws['%s%s'%(c, r)].value),
            '%s%s: Should start with "="' % (c, r)
        )

        for r, t in titles:
            for c in ('E', 'F', 'G'): _assert_is_formula(c, r, chk=lambda val: (val != '') and val.startswith('='))
            for c in ('I', 'J', 'K'): _assert_is_formula(c, r, chk=lambda val: (val == '') or val.startswith('='))
