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
#
# don't forget to recalculate fromulas (ctrl+shitf+f9 for libreoffice)
# don't forget to refresh filtering (column B filter, empty values, a bug in xlsxwriter and libreoffice)

"""
"""

# magic (hack), also it's possible to use export PYTHONIOENCODING=utf8
try:
    import sys
    reload(sys) # reload makes sys.setdefaultencoding method accessible
    sys.setdefaultencoding('utf-8')
except:
    pass

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
        self._role = (self._title) and ((self._title[0] == '(') and (self._title[-1] == ')')) or False
        self._annotations = {}
        self._childs = []
        self._estimations = {}
        self._estimates_cache = None

    def title(self):
        return self._title

    def is_role(self):
        return self._role

    def level(self):
        return self._level

    def title_with_level(self):
        return (('  ' * self.level()) + self.title())

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

        # update source
        val = self._estimations.get(role, None)
        if val is None: val = Estimation(numbers)
        else: val = val + numbers
        self._estimations[role] = val

        # and merge it
        self._estimates_cache = Node._estimates(self._estimations)

    def estimates(self):
        return self._estimates_cache

    @staticmethod
    def _estimates(estimations):
        if (len(estimations) <= 0): return None

        estimations = estimations.values()
        if (not estimations): return None

        return reduce(lambda x,y: x+y, estimations)


# color management
import colorsys
def _hls(h, l, s):
    r, g, b = ( int(0xff * i) for i in colorsys.hls_to_rgb(h, l, s) )
    return "#%02X%02X%02X" % (r, g, b)

# default theme (theme wrapper)
class Theme:
    """It wraps given object and looks into it for values: it will use default value it nothing is found.""" 

    _NO_VALUE = object()

    _SERIES_L = [ 0x84, 0xc2, 0xe0, 0xe8, 0xf0, 0xf8 ]
    _SERIES_L = [ float(l)/0xff for l in _SERIES_L ]
    _SERIES_S = 0.318

    _SERIES_H = 0.567
    DEFAULT_SECTION_H,\
    DEFAULT_SECTION_1,\
    DEFAULT_SECTION_2,\
    DEFAULT_SECTION_3,\
    DEFAULT_SECTION_4,\
    DEFAULT_SECTION_5 = [
        _hls(_SERIES_H, l, _SERIES_S) for l in _SERIES_L
    ]

    _SERIES_H = 0.067
    DEFAULT_FINAL,\
    DEFAULT_TOTAL = [
        _hls(_SERIES_H, l, _SERIES_S) for l in (_SERIES_L[2], _SERIES_L[4])
    ]

    DEFAULT_ROLE = '#7f7f7f'

    # default base style
    DEFAULT_F_DEFAULT = {
        'font_name': 'Arial',
        'font_size': 10,
        'valign': 'top'
    }

    # default header format
    DEFAULT_F_HEADER = {
        'bold': True,
        'bottom': 1,
        'text_wrap': True
    }

    # default cation format
    DEFAULT_F_CAPTION = {
        'bold': True
    }

    # default multiplier format
    DEFAULT_F_MULTIPLIER = {
        'num_format': '0'
    }

    # default number format
    DEFAULT_F_NUMBERS = {
        'num_format': '0.00'
    }

    # default number format
    DEFAULT_F_ESTIMATES = lambda self: self._merge_format(self.F_NUMBERS, {
        'bold': True
    })

    # comment row
    DEFAULT_F_COMMENT = {
        'text_wrap': True,
        'italic': False,
        'bold': False
    }

    # roles (temporary) row caption
    DEFAULT_F_ROLE_ROW = lambda self: {
        'font_color': self.ROLE,
        'italic': False,
        'bold': False
    }

    # 1st level (root) sections
    DEFAULT_F_SECTION_1 = lambda self: {
        'bold': True,
        'text_wrap': True,
        'bg_color': self.SECTION_1
    }

    # 2nd level sections
    DEFAULT_F_SECTION_2 = lambda self: {
        'italic': True,
        'text_wrap': True,
        'bg_color': self.SECTION_2
    }

    # 3rd level sections
    DEFAULT_F_SECTION_3 = lambda self: {
        'text_wrap': True,
        'bg_color': self.SECTION_3
    }

    # 4th level sections
    DEFAULT_F_SECTION_4 = lambda self: {
        'text_wrap': True,
        'bg_color': self.SECTION_4
    }

    # 5th level sections
    DEFAULT_F_SECTION_5 = lambda self: {
        'text_wrap': True,
        'bg_color': self.SECTION_5
    }

    # total row
    DEFAULT_F_TOTAL = lambda self: {
        'bold': True,
        'bg_color': self.TOTAL
    }

    # final values
    DEFAULT_F_FINAL = lambda self: {
        'bold': True,
        'bg_color': self.FINAL
    }

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):
        v = getattr(self._obj, attr, Theme._NO_VALUE)
        if (v is Theme._NO_VALUE):
            v = getattr(Theme, 'DEFAULT_' + attr)

        if (callable(v)): v = v(self)
        return v

    def _merge_format(self, *formats):
        result = {}
        for f in formats:
            if (callable(f)): f = f(self)
            if (not f): continue
            result.update(f)
        return { k:v for k,v in result.items() if v is not None }

    def format(self, opts = {}):
        return self._merge_format(self.F_DEFAULT, opts)

# format cache
class FormatCache:
    """It wraps current theme and call format registration only for completely new format combination"""

    def __init__(self, theme, register):
        self._cache = {}
        self._theme = theme
        self._register = register

    @staticmethod
    def _key(format):
        items = format.items()
        items.sort()
        return tuple(items)

    @staticmethod
    def _merge(*formats):
        result = {}
        for f in formats:
            if (not f): continue
            result.update(f)
        return result

    def get(self, *formats):
        format = FormatCache._merge(*formats)
        key = FormatCache._key(format)
        val = self._cache.get(key, None)
        if (val is None):
            format = self._theme.format(format)
            self._cache[key] = val = self._register(format)

        return val


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
        theme = None
        if (path is not None):
            import importlib
            module = importlib.import_module(path)
            theme = module.Theme

        if (theme is None): theme = object() # just an empty object
        return Theme(theme)

    #
    def __init__(self, options):
        self._sorting = options.sorting and True or False
        self._theme = Processor._loadTheme(options.theme)
        self._p99 = options.p99 and True or False
        self._roles = options.roles and True or False
        self._formulas = self._roles and (options.formulas and True or False)
        self._filter_visibility = (options.filter_visibility and True or False) # self._roles

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

        # sort by title
        if (self._sorting):
            xmlNodes = [ (Processor._text(x), x) for x in xmlNodes ]
            xmlNodes.sort(lambda x, y: cmp(x[0], y[0]))
            xmlNodes = [ x for title, x in xmlNodes ]

        # start working with nodes
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
                childs = node.childs()
                childs.sort(lambda x, y: cmp(
                    x.estimates() is not None,
                    y.estimates() is not None
                ))
                for n in childs:
                    lines.append(n)
                    _collectChilds(n)
            _collectChilds(root)
            return lines

        lines = _collect(root)

        # cell format cache
        _cf_cache = FormatCache(self._theme, lambda f: wb.add_format(f))

        # helper functions
        cell  = lambda a, b: "%s%s" % (a, (1+b))
        cells = lambda a, b: "%s:%s" % (cell(a, b[0]), cell(a, b[1]))

        _column  = lambda c, width, f=None, options={}: wb_sheet.set_column('%s:%s' % (c, c), width=width, cell_format=_cf_cache.get(f), options=options)

        _string  = lambda c, r, string, *f: wb_sheet.write_string(cell(c, r), string, cell_format = _cf_cache.get(*f))
        _number  = lambda c, r, number, *f: wb_sheet.write_number(cell(c, r), number, cell_format = _cf_cache.get(self._theme.F_NUMBERS, *f))
        _formula = lambda c, r, formula, *f: wb_sheet.write_formula(cell(c, r), formula, cell_format = _cf_cache.get(self._theme.F_NUMBERS, *f))
        #_blank   = lambda c, r, *f: wb_sheet.write_blank(cell(c, r), blank = '', cell_format = _cf_cache.get(*f))
        _blank   = lambda c, r, *f: _string(c,r,'',*f)

        # init format (style) table
        f_caption = self._theme.F_CAPTION
        f_comment = self._theme.F_COMMENT
        f_estimates = self._theme.F_ESTIMATES
        f_role = self._theme.F_ROLE_ROW
        f_total = self._theme.F_TOTAL
        f_final = self._theme.F_FINAL
        f_multiplier = self._theme.F_MULTIPLIER

        # create & init sheet
        wb_sheet = wb.add_worksheet()

        # setup columns
        _column('A', width=40, f=self._theme.F_DEFAULT)
        _column('B', width=3,  f=self._theme.F_MULTIPLIER)
        _column('C', width=6,  f=self._theme.F_DEFAULT)
        _column('D', width=50, f=self._theme.F_COMMENT)
        _column('E', width=7,  f=self._theme.F_NUMBERS)
        _column('F', width=7,  f=self._theme.F_NUMBERS)
        _column('G', width=7,  f=self._theme.F_NUMBERS)
        _column('H', width=7,  f=self._theme.F_DEFAULT)
        _column('I', width=7,  f=self._theme.F_NUMBERS)
        _column('J', width=7,  f=self._theme.F_NUMBERS)
        _column('K', width=7,  f=self._theme.F_NUMBERS)

        if (not self._filter_visibility):
            _column('B', width=3,  f=self._theme.F_MULTIPLIER, options = { 'hidden': True })

        # start rows
        row = 0

        # header (row = 1)
        f_header = self._theme.F_HEADER
        _string('A', row, 'Task / Subtask', f_header)  # A: caption
        _string('B', row, 'Filter', f_header)          # B: visibility
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
        row_lines = {}
        for l in lines:

            # first, remove all empty lines
            if (not l): continue

            # then remove role-rows, if it's required
            role = l.is_role()
            if (role and (not self._roles)): continue

            # let's start
            row += 1
            row_lines[id(l)] = row

            # obtain estimations for the row (total, aggregated from the node and its roles)
            estimates = l.estimates()

            # ----------------------
            # special case for roles (they are just info rows)
            if (role):

                # collect roles rows, if roles are required
                role_rows = roles_rows.get(l.title(), None)
                if role_rows is None:
                    roles_rows[l.title()] = role_rows = []
                role_rows.append(row)

                # set row options (hide it)
                wb_sheet.set_row(row, options={ 'hidden': True })

                # fill with values
                _string('A', row, l.title_with_level(), f_role)  # A (title)
                _number('B', row, 0, f_role, f_multiplier)       # B (visibility/multiplier)
                _blank('C', row, f_role)                         # C (empty)
                _string('D', row, '', f_role)                    # D (comment)
                if (estimates is not None):
                    _number('E', row, estimates[0], f_role)      # E (estimate)
                    _number('F', row, estimates[1], f_role)      # F (estimate)
                    _number('G', row, estimates[2], f_role)      # G (estimate)

                continue

            # ---------------
            # not a role case (visible rows, main data)

            # determine row format (for non-estimation lines only)
            f_row = None
            if (estimates is None):
                try: f_row = getattr(self._theme, 'F_SECTION_%s' % (1 + l.level()))
                except: pass

            # and store them to current line
            l._f_row = f_row

            # multiplier will be used in SUMPRODUCT formulas:
            # true means it's a raw data for formulas - we have to use it
            multiplier = (not role) and (estimates is not None)

            # let's fill the row with data
            _string('A', row, l.title_with_level(), f_row)                            # A (title)
            _blank('B', row, f_row, f_multiplier)                                     # B (visibility/multiplier)
            _blank('C', row, f_row)                                                   # C (empty)
            _string('D', row, ';\n'.join(l.annotation('comment')), f_row, f_comment)  # D (comment)
            _string('E', row, '', f_row)                                              # E (estimate)
            _string('F', row, '', f_row)                                              # F (estimate)
            _string('G', row, '', f_row)                                              # G (estimate)
            _blank('H', row, f_row)                                                   # H (empty)
            _string('I', row, '', f_row)                                              # I (estimate)
            _string('J', row, '', f_row)                                              # J (estimate)
            _string('K', row, '', f_row)                                              # K (estimate)

            # setup visibility/multiplier
            if (multiplier):
                _number('B', row, 1, f_row, f_multiplier)    # B (visibility/multiplier)

            if (estimates is not None):
                _number('E', row, estimates[0], f_row, f_estimates)               # E (estimate)
                _number('F', row, estimates[1], f_row, f_estimates)               # F (estimate)
                _number('G', row, estimates[2], f_row, f_estimates)               # G (estimate)
                _formula('I', row, '=(%s+4*%s+%s)/6' % (cell('E', row), cell('F', row), cell('G', row)), f_row, f_estimates) # I (weighted mean)
                _formula('J', row, '=(%s-%s)/6' % (cell('G', row), cell('E', row)), f_row, f_estimates)                      # J (standard deviation)
                _formula('K', row, '=%s*%s' % (cell('J', row), cell('J', row)), f_row, f_estimates)                          # K (squared deviation)


        # -------------------------------------
        # change estimation numbers to formulas
        if (self._formulas):
            for l in lines:
                if (not l): continue
                if (l.is_role()): continue
                if (not l.childs()): continue

                l_row = row_lines[id(l)]

                # build a formula
                template = "=0"
                for c in l.childs():
                    c_row = row_lines[id(c)]
                    template += "+"+cell("#", c_row)

                # trim the formula
                template = template.replace("=0+", "=")

                # write write to document
                f_row = l._f_row
                if (l.estimates()):
                    _formula('E', l_row, template.replace('#', 'E'), f_row, f_estimates)  # E (estimate)
                    _formula('F', l_row, template.replace('#', 'F'), f_row, f_estimates)  # F (estimate)
                    _formula('G', l_row, template.replace('#', 'G'), f_row, f_estimates)  # G (estimate)
                else:
                    _formula('E', l_row, template.replace('#', 'E'), f_row, f_role) # E (estimate)
                    _formula('F', l_row, template.replace('#', 'F'), f_row, f_role) # F (estimate)
                    _formula('G', l_row, template.replace('#', 'G'), f_row, f_role) # G (estimate)


        # footer
        row_footer = row+1

        # ----------------------------------
        # calculate ranges & apply filtering
        row_lines = row_lines.values()
        row_lines = (min(row_lines), max(max(row_lines), row_footer))
        row_multiplier = cells('B', row_lines)

        # set up autofilters (in headers)
        wb_sheet.autofilter('%s:%s' % (cell('A', 0), cell('K', row_footer)))
        if (self._filter_visibility):
            wb_sheet.filter_column('B', 'x == Blanks or x == 1') # B (visibility/multiplier)

        # ------
        # footer

        # total values (it uses row_multiplier to avoid duuble calculations for roles)
        row_footer += 1
        row_total = row_footer
        _string('A', row_total, 'Total', f_total)                                                                        # A (caption)
        _string('B', row_total, '', f_total)                                                                             # B (hidden)
        _string('C', row_total, '', f_total)                                                                             # C
        _string('D', row_total, '', f_total)                                                                             # D
        _formula('E', row_total, '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), row_multiplier), f_total, f_estimates)   # E (sum)
        _formula('F', row_total, '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), row_multiplier), f_total, f_estimates)   # F (sum)
        _formula('G', row_total, '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), row_multiplier), f_total, f_estimates)   # G (sum)
        _string('H', row_total, '', f_total)                                                                             # H
        _formula('I', row_total, '=SUMPRODUCT(%s,%s)' % (cells('I', row_lines), row_multiplier), f_total, f_estimates)   # I (sum)
        _formula('J', row_total, '=SUMPRODUCT(%s,%s)' % (cells('J', row_lines), row_multiplier), f_total, f_estimates)   # J (sum)
        _formula('K', row_total, '=SUMPRODUCT(%s,%s)' % (cells('K', row_lines), row_multiplier), f_total, f_estimates)   # K (sum)

        # total values for each role, if it's required
        if (self._roles):
            role_num = 0
            role_rows = roles_rows.items()
            role_rows.sort()
            for role, role_rows in role_rows:
                row_footer += 1
                role_num += 1
                role_column = chr(ord('L') + role_num) # new column for each role, started from 'L'

                _column(role_column, width=10, f=self._theme.F_DEFAULT, options={'hidden': True})

                _string(role_column, 0, role, f_header)
                for role_row in role_rows:
                    _number(role_column, role_row, 1)

                role_multiplier = cells(role_column, row_lines)
                _string('A', row_footer, '  - %s' % role.strip('()'), f_total)                                                   # A (caption)
                _string('B', row_footer, '', f_total)                                                                            # B (hidden)
                _string('C', row_footer, '', f_total)                                                                            # C
                _string('D', row_footer, '', f_total)                                                                            # D
                _formula('E', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), role_multiplier), f_total, f_estimates) # E (sum)
                _formula('F', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), role_multiplier), f_total, f_estimates) # F (sum)
                _formula('G', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), role_multiplier), f_total, f_estimates) # G (sum)


        # one extra line
        row_footer += 1

        # sigma: standard deviation
        row_footer += 1
        row_sigma = row_footer
        _string('A', row_sigma, 'Standard deviation', f_caption)                     # A (caption)
        _formula('C', row_sigma, '=SQRT(%s)' % (cell('K', row_total)), f_estimates)  # C (sigma)

        # kappa: correction factor
        row_footer += 1
        row_kappa = row_footer
        _string('A', row_kappa, 'K', f_caption)    # A (caption)
        _number('C', row_kappa, 1.5, f_estimates)  # C (kappa)

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
        _formula('C', row_footer, '=%s-%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total, f_estimates)  # C (min)
        _formula('D', row_footer, '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final, f_estimates)                 # D (modified)

        # Max (P=95/99%)
        row_footer += 1
        _string('A', row_footer, 'Min (%s)' % p_title, f_total)  # A (caption)
        _string('B', row_footer, '', f_total)                    # B
        _formula('C', row_footer, '=%s+%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total, f_estimates)  # C (min)
        _formula('D', row_footer, '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final, f_estimates)                 # D (modified)

    # create a report
    def report(self, root, path):
        import xlsxwriter

        ## -----------------------------------------------------
        ## this is a XlsxWriter bug
        ## the following code should be removed after the bugfix

        def _write_filters(self, filters):
            blanks = next((f for f in filters if f == 'blanks'), None)
            if blanks is not None:
                self._xml_start_tag('filters', [('blank', 1)])
            else:
                self._xml_start_tag('filters')

            for autofilter in filters:
                self._write_filter(autofilter)

            self._xml_end_tag('filters')

        xlsxwriter.worksheet.Worksheet._write_filters = _write_filters

        ## -----------------------------------------------------

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
        '--formulas',
        action='store_true',
        dest='formulas',
        help='''use formulas for estimation numbers'''
    )

    # due to the combination of the bugs in libreoffice and xlsxwriter this options are not ready
    parser.add_argument(
        '--filter-visibility',
        action='store_true',
        dest='filter_visibility',
        help='''don't hide multiplier/visibility column, use it as a filter instead (EXPERIMENTAL)'''
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

