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
            module = importlib.load_module(path)
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

    @staticmethod
    def _text(xmlNode):
        # TODO: case insensitive attributes
        return xmlNode.getAttribute('TEXT') # xmind style for attributes

    #
    def _process(self, parent, xmlNodes):
        required = 0

        xmlNodes = [ n for n in xmlNodes if n.nodeType == n.ELEMENT_NODE ]
        xmlNodes = [ n for n in xmlNodes if n.tagName == 'node' ]

        if (self._sorting):
            xmlNodes.sort(lambda x, y: cmp(Processor._text(x), Processor._text(y)))

        for xmlNode in xmlNodes:
            title = Processor._text(xmlNode)

            # estimate
            match = Processor.RE_ESTIMATE.match(title)
            if (match):
                estimates = [ float(match.group(x).strip()) for x in (1,2,3) ]
                if (parent.is_role()):
                    role = parent.title()
                    parent.parent().estimate(role, estimates)

                parent.estimate(None, estimates) # (always) set the estimation for node itself
                required = 1
                continue

            # annotation (comment)
            for title_line in title.split('\n'):
                match = Processor.RE_ANNOTATION.match(title_line)
                if (match):
                    k, v = [ match.group(x).strip() for x in (1, 2) ]
                    k = Processor.ANNOTATIONS.get(k, None)
                    if (k):
                        k, p = k
                        parent.annotation(k, p + v)
                        required = 1

            # else
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

        # columns
        wb_sheet.set_column('A:A',  width=40, cell_format=f_default)
        wb_sheet.set_column('B:B',  width=3,  cell_format=f_default, options={'hidden': True})
        wb_sheet.set_column('C:C',  width=6,  cell_format=f_comment)
        wb_sheet.set_column('D:D',  width=50, cell_format=f_default)
        wb_sheet.set_column('E:K',  width=7,  cell_format=f_numbers)

        # start rows
        row = 0

        # header (row = 1)
        wb_sheet.write_string(cell('A', row), 'Task / Subtask', f_header)  # A: caption
        wb_sheet.write_string(cell('B', row), '', f_header)                # B: hidden
        wb_sheet.write_string(cell('C', row), '', f_header)                # C: empty
        wb_sheet.write_string(cell('D', row), 'Comment', f_header)         # D: comment
        wb_sheet.write_string(cell('E', row), 'Min', f_header)             # E: estimate
        wb_sheet.write_string(cell('F', row), 'Real', f_header)            # F: estimate
        wb_sheet.write_string(cell('G', row), 'Max', f_header)             # G: estimate
        wb_sheet.write_string(cell('H', row), '', f_header)                # H: empty
        wb_sheet.write_string(cell('I', row), 'Avg', f_header)             # I: weighted mean
        wb_sheet.write_string(cell('J', row), 'SD', f_header)              # J: standard deviation
        wb_sheet.write_string(cell('K', row), 'Sq', f_header)              # K: squared deviation

        # lines
        roles_rows = {}
        row_lines = []
        for l in lines:
            if not l: continue
            row += 1
            row_lines.append(row)

            role = l.is_role()
            estimates = l.estimates()

            if role:
                role_rows = roles_rows.get(l.title(), None)
                if role_rows is None:
                    roles_rows[l.title()] = role_rows = []
                role_rows.append(row)

            row_options = {}
            if (role):
                row_options['hidden'] = True
                row_options['collapsed'] = True
            wb_sheet.set_row(row, options=row_options)

            cell_format = None
            if (0 == l.level()):
                cell_format=f_section_0
            elif (1 == l.level()):
                if (estimates is None):
                    cell_format=f_section_1

            multiplier = (not role) and (estimates is not None)

            wb_sheet.write_string(cell('A', row), '%s' % (('  ' * l.level()) + l.title()), cell_format)  # A (title)
            wb_sheet.write_number(cell('B', row), (multiplier and 1 or 0), cell_format)                  # B (visibility/multiplier)
            wb_sheet.write_string(cell('C', row), '', cell_format)                                       # C (empty)
            wb_sheet.write_string(cell('D', row), ';\n'.join(l.annotation('comment')), f_comment)        # D (comment)

            if (estimates is not None):
                wb_sheet.write_number(cell('E', row), estimates[0], f_numbers)               # E (estimate)
                wb_sheet.write_number(cell('F', row), estimates[1], f_numbers)               # F (estimate)
                wb_sheet.write_number(cell('G', row), estimates[2], f_numbers)               # G (estimate)

                if (not role):
                    wb_sheet.write_formula(cell('I', row), '=(%s+4*%s+%s)/6' % (cell('E', row), cell('F', row), cell('G', row)), f_numbers) # I (weighted mean)
                    wb_sheet.write_formula(cell('J', row), '=(%s-%s)/6' % (cell('G', row), cell('E', row)), f_numbers)                      # J (standard deviation)
                    wb_sheet.write_formula(cell('K', row), '=%s*%s' % (cell('J', row), cell('J', row)), f_numbers)                          # K (squared deviation)

        # total
        row_lines = (row_lines[0], row_lines[-1])
        row_multiplier = cells('B', row_lines)

        # autofilters
        wb_sheet.autofilter('%s:%s' % (cell('A',0), cell('K', row_lines[-1])))

        # footer
        row_footer = row+1

        # total
        row_footer += 1
        row_total = row_footer
        wb_sheet.write_string(cell('A', row_total), 'Total', f_total)                                                                   # A (caption)
        wb_sheet.write_string(cell('B', row_total), '', f_total)                                                                        # B (hidden)
        wb_sheet.write_string(cell('C', row_total), '', f_total)                                                                        # C
        wb_sheet.write_string(cell('D', row_total), '', f_total)                                                                        # D
        wb_sheet.write_formula(cell('E', row_total), '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), row_multiplier), f_total_numbers)   # E (sum)
        wb_sheet.write_formula(cell('F', row_total), '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), row_multiplier), f_total_numbers)   # F (sum)
        wb_sheet.write_formula(cell('G', row_total), '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), row_multiplier), f_total_numbers)   # G (sum)
        wb_sheet.write_string(cell('H', row_total), '', f_total)                                                                        # H
        wb_sheet.write_formula(cell('I', row_total), '=SUMPRODUCT(%s,%s)' % (cells('I', row_lines), row_multiplier), f_total_numbers)   # I (sum)
        wb_sheet.write_formula(cell('J', row_total), '=SUMPRODUCT(%s,%s)' % (cells('J', row_lines), row_multiplier), f_total_numbers)   # J (sum)
        wb_sheet.write_formula(cell('K', row_total), '=SUMPRODUCT(%s,%s)' % (cells('K', row_lines), row_multiplier), f_total_numbers)   # K (sum)


        # total (by roles)
        role_num = 0
        role_rows = roles_rows.items()
        role_rows.sort()
        for role, role_rows in role_rows:
            row_footer += 1
            role_num += 1
            role_column = chr(ord('L') + role_num)

            wb_sheet.set_column('%s:%s' % (role_column, role_column), width=10, cell_format=f_default, options={'hidden': True})
            wb_sheet.write_string(cell(role_column, 0), role, f_header)
            for role_row in role_rows:
                wb_sheet.write_number(cell(role_column, role_row), 1)

            role_multiplier = cells(role_column, row_lines)
            wb_sheet.write_string(cell('A', row_footer), '  - %s' % role.strip('()'), f_total)                                                 # A (caption)
            wb_sheet.write_string(cell('B', row_footer), '', f_total)                                                                          # B (hidden)
            wb_sheet.write_string(cell('C', row_footer), '', f_total)                                                                          # C
            wb_sheet.write_string(cell('D', row_footer), '', f_total)                                                                          # D
            wb_sheet.write_formula(cell('E', row_footer), '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), role_multiplier), f_total_numbers)    # E (sum)
            wb_sheet.write_formula(cell('F', row_footer), '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), role_multiplier), f_total_numbers)    # F (sum)
            wb_sheet.write_formula(cell('G', row_footer), '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), role_multiplier), f_total_numbers)    # G (sum)


        # one extra line
        row_footer += 1

        # sigma: standard deviation
        row_footer += 1
        row_sigma = row_footer
        wb_sheet.write_string(cell('A', row_sigma), 'Standard deviation', f_bold)                      # A (caption)
        wb_sheet.write_formula(cell('C', row_sigma), '=SQRT(%s)' % (cell('K', row_total)), f_numbers)  # C (sigma)

        # kappa: correction factor
        row_footer += 1
        row_kappa = row_footer
        wb_sheet.write_string(cell('A', row_kappa), 'K', f_bold)     # A (caption)
        wb_sheet.write_number(cell('C', row_kappa), 1.5, f_numbers)  # C (kappa)

        if self._p99:
            # P=99%, super precision
            p_title = "P=99%"
            p_multiplier = 3
        else:
            # P=95%, regular precision
            p_title = "P=95%"
            p_multiplier = 2

        # Min (P=95/99%)
        row_footer += 1
        wb_sheet.write_string(cell('A', row_footer), 'Min (%s)' % p_title, f_total)  # A (caption)
        wb_sheet.write_string(cell('B', row_footer), '', f_total)                    # B
        wb_sheet.write_formula(cell('C', row_footer), '=%s-%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total_numbers)  # C (min)
        wb_sheet.write_formula(cell('D', row_footer), '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final_numbers)                 # D (modified)

        # Max (P=95/99%)
        row_footer += 1
        wb_sheet.write_string(cell('A', row_footer), 'Min (%s)' % p_title, f_total)  # A (caption)
        wb_sheet.write_string(cell('B', row_footer), '', f_total)                    # B
        wb_sheet.write_formula(cell('C', row_footer), '=%s+%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total_numbers)  # C (min)
        wb_sheet.write_formula(cell('D', row_footer), '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final_numbers)                 # D (modified)

    # create a report
    def report(self, root, path):
        import xlsxwriter
        with xlsxwriter.Workbook(path + '.xlsx') as wb:
            self._report(root, wb)

# let's dance
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description='Converts freemind estimation to xlsx report.')

    parser.add_argument(
        '--sort', '-s',
        action='store_true',
        dest='sorting',
        help='sort children nodes by title'
    )

    parser.add_argument(
        '--theme',
        action='store',
        dest='theme',
        help='use a given .py file as a theme'
    )

    parser.add_argument(
        '--p99',
        action='store_true',
        dest='p99',
        help='Use P=99%% instead of P=95%%'
    )

    parser.add_argument(
        'filename',
        help='a freemind (mindmap) file to be converted'
    )

    options = parser.parse_args()
    filename = options.filename

    processor = Processor(options)
    root = processor.parse(filename)
    processor.report(root, filename)

