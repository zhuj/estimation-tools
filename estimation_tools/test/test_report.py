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
    def _report(name, root, options):
        wb = estimate.Processor(options).report(root, filename=None)
        names = wb.get_sheet_names()
        return wb[names[0]]

    def _check_header(self, ws):
        self.assertEqual( (ws.min_column, ws.min_row), (1, 1) )
        self.assertEqual( _val(ws, 'A1'), 'Task / Subtask' )
        self.assertEqual( _val(ws, 'B1'), 'Filter' )
        self.assertEqual( _val(ws, 'C1'), '' )
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

        # roles (if exists)
        self.assertTrue( row > 1 )
        while _val(ws, 'A%s' % row).startswith(' - '):
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
        ReportTestCase._report("simple", root, object())

    def test_no_estimates(self):
        """test for simple case without estimates"""

        root = estimate.Node(parent=None, title=None)
        root.append(estimate.Node(parent=root, title="1st node with no estimate"))
        root.append(estimate.Node(parent=root, title="2nd node with no estimate"))
        root.append(estimate.Node(parent=root, title="3rd node with no estimate"))

        ws = ReportTestCase._report("no.estimates", root, object())

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

        ws = ReportTestCase._report("estimate.sort", root, object())

        data_min, data_max = self._check_header_and_footer(ws)
        self.assertEqual( data_min, 2 )
        self.assertEqual( data_max, 4 )

        self.assertEqual( _val(ws, 'A2').strip(), "2nd node" )
        self.assertEqual( _val(ws, 'A3').strip(), "3rd node" )
        self.assertEqual( _val(ws, 'A4').strip(), "1st node" ) # estimation rows should be in the end
