#!/usr/bin/env python2.7

#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Copyright (c) 2016 Viktor A. Danilov
# MIT License
# https://opensource.org/licenses/mit-license.html
#

#
# estimate.py
# pip install xlsxwriter
# don't forget to recalculate fromulas (ctrl+shitf+f9 for libreoffice)

"""
"""

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

# estimatuion
class Estimation:
    """Estimation is a set of numbers (min <= real <= max) which describes a range for time requirement."""

    @staticmethod
    def _checktype(numbers):
        from types import ListType, TupleType
        if (not isinstance(numbers, (ListType, TupleType))):
            raise TypeError("Invalid operand type: %s", type(numbers))


    def __init__(self, numbers):
        Estimation._checktype(numbers)
        self._numbers = tuple(numbers[0:3])

    def __str__(self):
        return str(self._numbers)

    def __repr__(self):
        return str(self._numbers)

    def __getitem__(self, idx):
        return self._numbers[idx]

    def __add__(self, numbers):
        if (isinstance(numbers, Estimation)):
            numbers = numbers._numbers

        Estimation._checktype(numbers)
        return Estimation(tuple(
            self._numbers[i] + numbers[i]
            for i in xrange(3)
        ))

# node
class Node:
    """It represents a node in modified mindmap tree (all estimates, roles and comments are collected as attributes)"""

    def __init__(self, parent, title):
        self._parent = parent
        if (self._parent is None): self._level = -1
        else: self._level = 1 + self._parent.level()
        self._title = title.strip()
        self._annotations = {}
        self._childs = []
        self._estimations = {}

    def title(self): return self._title

    def is_role(self): return ((self._title[0] == '(') and (self._title[-1] == ')'))

    def level(self): return self._level

    def parent(self, level=None):
        if (level is None): return self._parent
        if (self._level <= level): return self
        if (self._parent is None): return self
        return self._parent.parent(level)

    def append(self, node):
        self._childs.append(node)

    def childs(self):
        return self._childs

    def annotation(self, name, value=None):
        l = self._annotations.get(name, [])
        if (value is None): return l
        value = value.strip()
        if value:
            l.append(value)
            self._annotations[name] = l

    def estimate(self, role, numbers):
        prev = self._estimations.get(role, None)
        if prev is None:
            self._estimations[role] = Estimation(numbers)
        else:
            self._estimations[role] = prev + numbers

    def estimates(self):
        if (len(self._estimations) > 0):
            estimations = self._estimations.values()
            if (estimations):
                return reduce(
                    lambda x,y: x+y,
                    estimations
                )
        return None



# processor helper
class Processor:
    """Helper class which does all transformation work"""

    import re
    RE_ESTIMATE = re.compile("estimate\\s*=(\\s*\\d+\\s*)[/](\\s*\\d+\\s*)[/](\\s*\\d+\\s*)")
    RE_ANNOTATION = re.compile("^[[](\\w+)[]]\\s*(.*)")

    ANNOTATIONS = {
        'warn': ('comment', '(!) '),
        'idea': ('comment', ''),
        'todo': ('comment', ''),
        'impl': ('comment', '')
    }

    @staticmethod
    def _loadTheme(path):
        if (path is not None):
            import importlib
            module = importlib.import_module(path)
            theme = module.Theme
            if (theme is not None): return theme

        class Theme:
            SECTION_0 = '#DEEAF2'
            TOTAL     = '#F2EFED'
            FINAL     = '#E9DFDB'

        return Theme

    #
    def __init__(self, options):
        self._sorting = options.sorting and True or False
        self._theme = Processor._loadTheme(options.theme)
        self._p99 = options.p99 and True or False
        self._roles = options.roles and True or False

    @staticmethod
    def _text(xmlNode):
        ### if (xmlNode.hasAttribute('TEXT')): 
        ###     return xmlNode.getAttribute('TEXT') # xmind style for attributes
        text = ( v for k, v in xmlNode._get_attributes().items() if k.upper() == 'TEXT' )
        return text.next()


    #
    def _process(self, parent, xmlNodes):
        required = 0

        xmlNodes = [ n for n in xmlNodes if n.nodeType == n.ELEMENT_NODE ]
        xmlNodes = [ n for n in xmlNodes if n.tagName == 'node' ]

        if (self._sorting):
            xmlNodes = [ (Processor._text(x), x) for x in xmlNodes ]
            xmlNodes.sort(lambda x, y: cmp(x[0], y[0]))
            xmlNodes = [ x for title, x in xmlNodes ]

        for xmlNode in xmlNodes:
            title = Processor._text(xmlNode)

            # first look at the estimation pattern
            match = Processor.RE_ESTIMATE.match(title)
            if (match):
                estimates = [ float(match.group(x).strip()) for x in (1,2,3) ]
                if (parent.is_role()):
                    role = parent.title()
                    parent.parent().estimate(role, estimates)

                parent.estimate(None, estimates) # (always) set the estimation for node itself
                required = 1
                continue

            # then, try to parse the node as a comment
            for title_line in title.split('\n'):
                match = Processor.RE_ANNOTATION.match(title_line)
                if (match):
                    prefix, text = [ match.group(x).strip() for x in (1, 2) ]
                    prefix = Processor.ANNOTATIONS.get(prefix, None)
                    if (prefix):
                        k, p = prefix
                        parent.annotation(k, p + text)
                        required = 1

            # else handle it as a regular node
            node = Node(parent, title)
            node = self._process(node, xmlNode.childNodes)
            if (node is not None):
                parent.append(node)
                required = 1

        if (required): return parent
        return None

    #
    def parse(self, path):

        # parse mm document
        from xml.dom import minidom
        xmldoc = minidom.parse(path)
        xmlmap = xmldoc.getElementsByTagName('map')[0]
        xmlmap = xmlmap.childNodes[0]

        # transform document to nodelist
        root = Node(None, "root")
        root = self._process(root, xmlmap.childNodes)
        return root

    #
    def _report(self, root, wb):
        # see https://en.wikipedia.org/wiki/Three-point_estimation

        # first transform tree to lines
        def _collect(root):
            """ it collects the given hierarchy to list of lines """
            lines = []
            def _collectChilds(node):
                for n in node.childs():
                    lines.append(n)
                    _collectChilds(n)
            _collectChilds(root)
            return lines

        lines = _collect(root)

        # helper functions
        cell = lambda a, b: "%s%s" % (a, (1+b))
        cells = lambda a, b: "%s:%s" % (cell(a, b[0]), cell(a, b[1]))
        format = lambda opts: wb.add_format(dict(opts, **{
            'font_name': 'Arial',
            'font_size': 10,
            'valign': 'top'
        }))

        # init format (style) table
        f_default = format({})

        f_header  = format({'bold': True, 'bottom': 1, 'text_wrap': True })

        f_bold    = format({'bold': True })
        f_numbers = format({'bold': True, 'num_format': '0.00'})
        f_comment = format({'text_wrap': True})

        f_section_0 = format({'bold': True, 'bg_color': self._theme.SECTION_0})
        f_section_1 = format({'italic': True, 'text_wrap': True})

        f_total           = format({'bold': True, 'bg_color': self._theme.TOTAL})
        f_total_numbers   = format({'bold': True, 'bg_color': self._theme.TOTAL, 'num_format': '0.00'})
        f_final_numbers   = format({'bold': True, 'bg_color': self._theme.FINAL, 'num_format': '0.00'})

        # create & init sheet
        wb_sheet = wb.add_worksheet()

        _string = lambda c, r, string, f=None: wb_sheet.write_string(cell(c, r), string, cell_format = f)
        _number = lambda c, r, number, f=None: wb_sheet.write_number(cell(c, r), number, cell_format = f)
        _formula = lambda c, r, formula, f=None: wb_sheet.write_formula(cell(c, r), formula, cell_format = f)


        # columns
        wb_sheet.set_column('A:A',  width=40, cell_format=f_default)
        wb_sheet.set_column('B:B',  width=3,  cell_format=f_default, options={'hidden': True})
        wb_sheet.set_column('C:C',  width=6,  cell_format=f_comment)
        wb_sheet.set_column('D:D',  width=50, cell_format=f_default)
        wb_sheet.set_column('E:K',  width=7,  cell_format=f_numbers)

        # start rows
        row = 0

        # header (row = 1)
        _string('A', row, 'Task / Subtask', f_header)  # A: caption
        _string('B', row, '', f_header)                # B: hidden
        _string('C', row, '', f_header)                # C: empty
        _string('D', row, 'Comment', f_header)         # D: comment
        _string('E', row, 'Min', f_header)             # E: estimate
        _string('F', row, 'Real', f_header)            # F: estimate
        _string('G', row, 'Max', f_header)             # G: estimate
        _string('H', row, '', f_header)                # H: empty
        _string('I', row, 'Avg', f_header)             # I: weighted mean
        _string('J', row, 'SD', f_header)              # J: standard deviation
        _string('K', row, 'Sq', f_header)              # K: squared deviation

        # lines
        roles_rows = {}
        row_lines = []
        for l in lines:

            # first, remove all empty lines
            if (not l): continue

            # then remove role-rows, if it's required
            role = l.is_role()
            if (role and (not self._roles)): continue

            # let's start
            row += 1
            row_lines.append(row)

            # obtain estimations for the row (total, aggregated from the node and its roles)
            estimates = l.estimates()

            # collect roles rows, if roles are required
            if (role and self._roles):
                role_rows = roles_rows.get(l.title(), None)
                if role_rows is None:
                    roles_rows[l.title()] = role_rows = []
                role_rows.append(row)

            # calculate and apply a style for row
            row_options = {}
            if (role):
                row_options['hidden'] = True
                row_options['collapsed'] = True
            wb_sheet.set_row(row, options=row_options)

            # do so with cell format for the title
            cell_format = None
            if (0 == l.level()):
                cell_format=f_section_0
            elif (1 == l.level()):
                if (estimates is None):
                    cell_format=f_section_1

            # multiplier will be used in SUMPRODUCT formulas:
            # true means it's a raw data for formulas - we have to use it
            multiplier = (not role) and (estimates is not None)

            # let's fill the row with data
            _string('A', row, '%s' % (('  ' * l.level()) + l.title()), cell_format)  # A (title)
            _number('B', row, (multiplier and 1 or 0), cell_format)                  # B (visibility/multiplier)
            _string('C', row, '', cell_format)                                       # C (empty)
            _string('D', row, ';\n'.join(l.annotation('comment')), f_comment)        # D (comment)

            if (estimates is not None):
                _number('E', row, estimates[0], f_numbers)               # E (estimate)
                _number('F', row, estimates[1], f_numbers)               # F (estimate)
                _number('G', row, estimates[2], f_numbers)               # G (estimate)

                if (not role):
                    _formula('I', row, '=(%s+4*%s+%s)/6' % (cell('E', row), cell('F', row), cell('G', row)), f_numbers) # I (weighted mean)
                    _formula('J', row, '=(%s-%s)/6' % (cell('G', row), cell('E', row)), f_numbers)                      # J (standard deviation)
                    _formula('K', row, '=%s*%s' % (cell('J', row), cell('J', row)), f_numbers)                          # K (squared deviation)

        # calculate ranges
        row_lines = (row_lines[0], row_lines[-1])
        row_multiplier = cells('B', row_lines)

        # set up autofilters (in headers)
        wb_sheet.autofilter('%s:%s' % (cell('A',0), cell('K', row_lines[-1])))

        # footer
        row_footer = row+1

        # total values (it uses row_multiplier to avoid duuble calculations for roles)
        row_footer += 1
        row_total = row_footer
        _string('A', row_total, 'Total', f_total)                                                                   # A (caption)
        _string('B', row_total, '', f_total)                                                                        # B (hidden)
        _string('C', row_total, '', f_total)                                                                        # C
        _string('D', row_total, '', f_total)                                                                        # D
        _formula('E', row_total, '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), row_multiplier), f_total_numbers)   # E (sum)
        _formula('F', row_total, '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), row_multiplier), f_total_numbers)   # F (sum)
        _formula('G', row_total, '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), row_multiplier), f_total_numbers)   # G (sum)
        _string('H', row_total, '', f_total)                                                                        # H
        _formula('I', row_total, '=SUMPRODUCT(%s,%s)' % (cells('I', row_lines), row_multiplier), f_total_numbers)   # I (sum)
        _formula('J', row_total, '=SUMPRODUCT(%s,%s)' % (cells('J', row_lines), row_multiplier), f_total_numbers)   # J (sum)
        _formula('K', row_total, '=SUMPRODUCT(%s,%s)' % (cells('K', row_lines), row_multiplier), f_total_numbers)   # K (sum)

        # total values for each role, if it's required
        if (self._roles):
            role_num = 0
            role_rows = roles_rows.items()
            role_rows.sort()
            for role, role_rows in role_rows:
                row_footer += 1
                role_num += 1
                role_column = chr(ord('L') + role_num) # new column for each role, started from 'L'

                wb_sheet.set_column('%s:%s' % (role_column, role_column), width=10, cell_format=f_default, options={'hidden': True})
                _string(role_column, 0, role, f_header)
                for role_row in role_rows:
                    _number(role_column, role_row, 1)

                role_multiplier = cells(role_column, row_lines)
                _string('A', row_footer, '  - %s' % role.strip('()'), f_total)                                                 # A (caption)
                _string('B', row_footer, '', f_total)                                                                          # B (hidden)
                _string('C', row_footer, '', f_total)                                                                          # C
                _string('D', row_footer, '', f_total)                                                                          # D
                _formula('E', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), role_multiplier), f_total_numbers)    # E (sum)
                _formula('F', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), role_multiplier), f_total_numbers)    # F (sum)
                _formula('G', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), role_multiplier), f_total_numbers)    # G (sum)


        # one extra line
        row_footer += 1

        # sigma: standard deviation
        row_footer += 1
        row_sigma = row_footer
        _string('A', row_sigma, 'Standard deviation', f_bold)                      # A (caption)
        _formula('C', row_sigma, '=SQRT(%s)' % (cell('K', row_total)), f_numbers)  # C (sigma)

        # kappa: correction factor
        row_footer += 1
        row_kappa = row_footer
        _string('A', row_kappa, 'K', f_bold)     # A (caption)
        _number('C', row_kappa, 1.5, f_numbers)  # C (kappa)

        if (self._p99):
            # P=99%, super precision
            p_title = "P=99%"
            p_multiplier = 3
        else:
            # P=95%, regular precision
            p_title = "P=95%"
            p_multiplier = 2

        # Min (P=95/99%)
        row_footer += 1
        _string('A', row_footer, 'Min (%s)' % p_title, f_total)  # A (caption)
        _string('B', row_footer, '', f_total)                    # B
        _formula('C', row_footer, '=%s-%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total_numbers)  # C (min)
        _formula('D', row_footer, '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final_numbers)                 # D (modified)

        # Max (P=95/99%)
        row_footer += 1
        _string('A', row_footer, 'Min (%s)' % p_title, f_total)  # A (caption)
        _string('B', row_footer, '', f_total)                    # B
        _formula('C', row_footer, '=%s+%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total_numbers)  # C (min)
        _formula('D', row_footer, '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final_numbers)                 # D (modified)

    # create a report
    def report(self, root, path):
        import xlsxwriter
        with xlsxwriter.Workbook(path) as wb:
            self._report(root, wb)

# let's dance
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='''Converts freemind estimation to xlsx report.'''
    )

    parser.add_argument(
        '--sort', '-s',
        action='store_true',
        dest='sorting',
        help='''sort children nodes by title'''
    )

    parser.add_argument(
        '--theme',
        action='store',
        dest='theme',
        help='''use a given .py file as a theme'''
    )

    parser.add_argument(
        '--p99',
        action='store_true',
        dest='p99',
        help='''use P=99%% instead of P=95%%'''
    )

    parser.add_argument(
        '--no-roles',
        action='store_false',
        dest='roles',
        help='''don't provide estimation details for each role'''
    )

    parser.add_argument(
        '-o',
        action='store',
        dest='output',
        help='''out file name'''
    )

    parser.add_argument(
        'filename',
        help='''a freemind (mindmap) file to be converted'''
    )

    options = parser.parse_args()
    filename = options.filename

    processor = Processor(options)
    root = processor.parse(filename)
    processor.report(root, options.output or (filename + ".xlsx"))

