#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
# usage: estimate.py [-h] [options] [-o OUTPUT] filename
# see https://github.com/zhuj/estimation-tools/wiki
#
# pre-requirements:
# * pip install openpyxl (http://openpyxl.readthedocs.io/en/default/, https://bitbucket.org/openpyxl/openpyxl/src/)
#

"""
"""

# magic (hack), also it's possible to use export PYTHONIOENCODING=utf8
try:
    import sys
    reload(sys) # reload makes sys.setdefaultencoding method accessible
    sys.setdefaultencoding('utf-8')
except:
    pass

# estimatuion (contains set of numbers)
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
        return self.__repr__()

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

# node (representation of mindmap xml node with additional properties)
class Node:
    """It represents a node in modified mindmap tree (all estimates, roles and comments are collected as attributes)"""

    def __init__(self, parent, title):
        self._parent = parent

        if (self._parent is None): self._level = -1
        else: self._level = 1 + self._parent.level()

        self._title = (title and title.strip() or "")
        self._role = (self._title) and ((self._title[0] == '(') and (self._title[-1] == ')')) or False
        self._annotations = {}
        self._childs = []
        self._estimations = {}
        self._estimates_cache = None

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return '<%s level=%s estimates=%s title="%s">' % (
            self.__class__.__name__,
            self.level(),
            self.estimates(),
            self.title()
        )

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


# color management (just to create color series)
import colorsys
def _hls(h, l, s):
    r, g, b = ( int(0xff * i) for i in colorsys.hls_to_rgb(h, l, s) )
    return "#%02X%02X%02X" % (r, g, b)

# default theme (theme wrapper)
class Theme:
    """It wraps given object and looks into it for values: it will use default value it nothing is found.""" 

    _NO_VALUE = object() # placeholder to distinguish true-None and no-value

    # series of lightness
    _SERIES_L = [ 0x84, 0xc2, 0xe0, 0xe8, 0xf0, 0xf8 ]
    _SERIES_L = [ float(l)/0xff for l in _SERIES_L ]

    # saturation (common for all coloring)
    _SERIES_S = 0.318

    # tone for sections (blue sector)
    _SERIES_H = 0.567
    DEFAULT_SECTION_H,\
    DEFAULT_SECTION_1,\
    DEFAULT_SECTION_2,\
    DEFAULT_SECTION_3,\
    DEFAULT_SECTION_4,\
    DEFAULT_SECTION_5 = [
        _hls(_SERIES_H, l, _SERIES_S) for l in _SERIES_L
    ]

    # tone for total values (red sector)
    _SERIES_H = 0.067
    DEFAULT_FINAL,\
    DEFAULT_TOTAL = [
        _hls(_SERIES_H, l, _SERIES_S) for l in (_SERIES_L[2], _SERIES_L[4])
    ]

    # role font color
    DEFAULT_ROLE = '#7f7f7f'

    # default base style
    DEFAULT_F_DEFAULT = {
        'font_name': 'Arial',
        'font_size': 10,
        'align_vertical': 'top'
    }

    # default header format
    DEFAULT_F_HEADER = lambda self: {
        'font_bold': True,
        'border_bottom_color': '#000000',
        'border_bottom_style': 'thin',
        'align_wrap_text': True
    }

    # default cation format
    DEFAULT_F_CAPTION = {
        'font_bold': True
    }

    # default multiplier format
    DEFAULT_F_MULTIPLIER = {
        'num_format': '0'
    }

    # default number format
    DEFAULT_F_NUMBERS = {
        'num_format': '0.00'
    }

    # default boolean format
    DEFAULT_F_BOOLEAN = {
    }

    # default number format
    DEFAULT_F_ESTIMATES = lambda self: self._merge_format(self.F_NUMBERS, {
        'font_bold': True
    })

    # comment row
    DEFAULT_F_COMMENT = {
        'align_wrap_text': True,
        'font_italic': False,
        'font_bold': False
    }

    # roles (temporary) row caption
    DEFAULT_F_ROLE_ROW = lambda self: {
        'font_color': self.ROLE,
        'font_italic': False,
        'font_bold': False
    }

    # 1st level (root) sections
    DEFAULT_F_SECTION_1 = lambda self: {
        'font_bold': True,
        'align_wrap_text': True,
        'fill_color': self.SECTION_1
    }

    # 2nd level sections
    DEFAULT_F_SECTION_2 = lambda self: {
        'font_italic': True,
        'align_wrap_text': True,
        'fill_color': self.SECTION_2
    }

    # 3rd level sections
    DEFAULT_F_SECTION_3 = lambda self: {
        'align_wrap_text': True,
        'fill_color': self.SECTION_3
    }

    # 4th level sections
    DEFAULT_F_SECTION_4 = lambda self: {
        'align_wrap_text': True,
        'fill_color': self.SECTION_4
    }

    # 5th level sections
    DEFAULT_F_SECTION_5 = lambda self: {
        'align_wrap_text': True,
        'fill_color': self.SECTION_5
    }

    # total row
    DEFAULT_F_TOTAL = lambda self: {
        'font_bold': True,
        'fill_color': self.TOTAL
    }

    # final values
    DEFAULT_F_FINAL = lambda self: {
        'font_bold': True,
        'fill_color': self.FINAL
    }

    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, attr):
        """it looks for the attribute in associated object first, then does the same in the theme object itself"""
        v = getattr(self._obj, attr, Theme._NO_VALUE)
        if (v is Theme._NO_VALUE):
            v = getattr(Theme, 'DEFAULT_' + attr)

        if (callable(v)):
            v = v(self)

        if (type(v) is str):
            if (len(v) == 7 and v[0] == '#'): v = 'FF' + v[1:]

        return v

    def _merge_format(self, *formats):
        """it merges (combines) all formats (dicts) together"""
        result = {}
        for f in formats:
            if (callable(f)): f = f(self)
            if (not f): continue
            result.update(f)
        return { k:v for k,v in result.items() if v is not None }

    def format(self, opts = {}):
        """public method for formats: it extends default format with the given one"""
        return self._merge_format(self.F_DEFAULT, opts)

# format cache
class FormatCache:
    """It wraps current theme and call format registration only for completely new format combination"""

    def __init__(self, format=lambda f:f, register=lambda f:f):
        self._cache = {}
        self._format = format
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
            format = self._format(format)
            self._cache[key] = val = self._register(format)

        return val


# processor helper
class Processor:
    """Helper class which does all transformation work"""

    import openpyxl as pyxl

    # options
    OPT_THEME = 'theme'
    OPT_SORTING = 'sorting'
    OPT_P_99 = 'p99'
    OPT_MVP = 'mvp'
    OPT_ROLES = 'roles'
    OPT_VALIDATION = 'validation'
    OPT_FORMULAS = 'formulas'
    OPT_FILTER_VISIBILITY = 'filter_visibility'

    # regexps patterns
    import re
    RE_ESTIMATE = re.compile("estimate\\s*=(\\s*\\d+\\s*)[/](\\s*\\d+\\s*)[/](\\s*\\d+\\s*)")
    RE_ANNOTATION = re.compile("^[[](\\w+)[]]\\s*(.*)")

    # annotation types
    ANNOTATIONS = {
        'warn': ('comment', '(!) '),
        'idea': ('comment', ''),
        'todo': ('comment', ''),
        'impl': ('comment', '')
    }

    @staticmethod
    def _loadTheme(theme):
        if (isinstance(theme, str)):
            import importlib
            module = importlib.import_module(theme)
            theme = module.Theme

        if (theme is None): theme = object() # just an empty object
        return Theme(theme)

    #
    @staticmethod
    def _wrap_options(options):

        if (isinstance(options, dict)):
            class objectview:
                def __init__(self, d):
                    self.__dict__ = d
            options = objectview(options)

        return options

    #
    def __init__(self, options):
        options = Processor._wrap_options(options)
        self._theme = Processor._loadTheme(getattr(options, Processor.OPT_THEME, None))
        self._sorting = getattr(options, Processor.OPT_SORTING, False) and True
        self._validation = getattr(options, Processor.OPT_VALIDATION, False) and True
        self._mvp = getattr(options, Processor.OPT_MVP, False) and True
        self._p99 = getattr(options, Processor.OPT_P_99, False) and True
        self._roles = getattr(options, Processor.OPT_ROLES, False) and True
        self._formulas = self._roles and (getattr(options, Processor.OPT_FORMULAS, False) and True)
        self._filter_visibility = self._roles and (getattr(options, Processor.OPT_FILTER_VISIBILITY, False) and True)

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
    @staticmethod
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
        return [ l for l in lines if l ]

    #
    @staticmethod
    def _cf_cache(theme):
        """ build cell-format cache """

        _filter_map = lambda map, prefix: {
            k[len(prefix):]: v for (k, v) in map.items() if k.startswith(prefix)
        }

        _cf_font_cache = FormatCache(
            register=lambda f: f and Processor.pyxl.styles.Font(**f) or None
        )

        _cf_fill_cache = FormatCache(
            register=lambda f: f and Processor.pyxl.styles.PatternFill(fill_type='solid', start_color=f.get('color', None)) or None
        )

        _cf_align_cache = FormatCache(
            register=lambda f: f and Processor.pyxl.styles.Alignment(**f) or None
        )

        _cf_border_cache = FormatCache(
            register=lambda f: f and Processor.pyxl.styles.Border(**{
                side: Processor.pyxl.styles.Side(
                    border_style = map.get("style", None),
                    color = map.get("color", None),
                )
                for side, map in (
                    (side, _filter_map(f, side))
                    for side in ('left', 'right', 'top', 'bottom')
                )
            }) or None
        )

        return FormatCache(
            format=lambda f: theme.format(f),
            register=lambda f: {
                'font': _cf_font_cache.get( _filter_map(f, 'font_') ),
                'fill': _cf_fill_cache.get( _filter_map(f, 'fill_') ),
                'alignment': _cf_align_cache.get( _filter_map(f, 'align_') ),
                'border': _cf_border_cache.get( _filter_map(f, 'border_') ),
                'number_format': f.get('num_format', None)
            }
        )


    #
    def _report(self, root, ws):
        # see https://en.wikipedia.org/wiki/Three-point_estimation

        # first transform tree to lines
        lines = Processor._collect(root)
        if (not lines): return # do nothing with empty document

        # cell format cache
        _cf_cache = Processor._cf_cache(self._theme)

        # helper functions
        cell  = lambda c, r: "%s%s" % (c, (1+r))
        cells = lambda c, r: "%s:%s" % (cell(c, r[0]), cell(c, r[1]))

        def _apply_format(c, *f):
            f = _cf_cache.get(*f)
            for x in ('font', 'fill', 'alignment', 'number_format'):
                v = f.get(x, None)
                if v is not None: setattr(c, x, v)
            return c

        def _hide_column(c, hidden=True): ws.column_dimensions[c].hidden = hidden
        def _hide_row(r, hidden=True): ws.row_dimensions[1+r].hidden = hidden

        def _column(c, width, f): _apply_format(ws.column_dimensions[c], f).width = width
        def _string(c, r, string, *f): _apply_format(ws[cell(c, r)], *f).value = string
        def _number(c, r, number, *f): _apply_format(ws[cell(c, r)], self._theme.F_NUMBERS, *f).value = number
        def _formula(c, r, formula, *f): _apply_format(ws[cell(c, r)], self._theme.F_NUMBERS, *f).value = formula
        def _boolean(c, r, bool, *f): _apply_format(ws[cell(c, r)], self._theme.F_BOOLEAN, *f).value = (bool and 1 or 0)
        def _blank(c, r, *f): _string(c, r, '', *f)

        # init format (style) table
        f_caption = self._theme.F_CAPTION
        f_comment = self._theme.F_COMMENT
        f_estimates = self._theme.F_ESTIMATES
        f_role = self._theme.F_ROLE_ROW
        f_total = self._theme.F_TOTAL
        f_final = self._theme.F_FINAL
        f_multiplier = self._theme.F_MULTIPLIER

        # ------------
        # setup header

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
            _hide_column('B', hidden=True)

        # start rows
        row = 0

        # header (row = 1)
        f_header = self._theme.F_HEADER
        _string('A', row, 'Task / Subtask', f_header)  # A: caption
        _string('B', row, 'Filter', f_header)          # B: visibility
        _string('C', row, '', f_header)                # C: empty/MVP
        _string('D', row, 'Comment', f_header)         # D: comment
        _string('E', row, 'Min', f_header)             # E: estimate
        _string('F', row, 'Real', f_header)            # F: estimate
        _string('G', row, 'Max', f_header)             # G: estimate
        _string('H', row, '', f_header)                # H: empty
        _string('I', row, 'Avg', f_header)             # I: weighted mean
        _string('J', row, 'SD', f_header)              # J: standard deviation
        _string('K', row, 'Sq', f_header)              # K: squared deviation

        if (self._mvp):
            _string('C', row, 'MVP', f_header) # C: MVP

        # ------------------------
        # prepare data validation

        # validation values for B (multiplier) and C (mvp)
        _hide_column('Z', hidden=True)     # hide Z
        _blank('Z', 0, f_multiplier)       # Z1 = empty
        _number('Z', 1, 0, f_multiplier)   # Z2 = 0
        _number('Z', 2, 1, f_multiplier)   # Z3 = 1

        # multiplier
        # XXX: note, the validation here is general only (for the whole column)
        # XXX: the idea is that user wants to control which lines are raw data source and which are representation only
        dv_mul_list = Processor.pyxl.worksheet.datavalidation.DataValidation(
            type="list",
            formula1='$Z$1:$Z$3', # '', 0, 1
            allow_blank=True
        )
        dv_mul_list.hide_drop_down = True

        # mvp (selecatable)
        dv_mvp_list = Processor.pyxl.worksheet.datavalidation.DataValidation(
            type="list",
            formula1='$Z$2:$Z$3', # 0, 1
            allow_blank=False
        )
        dv_mvp_list.hide_drop_down = False

        # mvp (empty only)
        dv_mvp_empty = Processor.pyxl.worksheet.datavalidation.DataValidation(
            type="list",
            formula1="", # none
            allow_blank=True
        )
        dv_mvp_empty.hide_drop_down = True

        # ------------------------
        # start the transformation

        # lines
        roles_rows = {}
        row_lines = {}
        for l in lines:

            # then remove role-rows, if it's required
            role = l.is_role()
            if (role and (not self._roles)): continue

            # let's start
            row += 1
            row_lines[id(l)] = row

            # obtain estimations for the row (total, aggregated from the node and its roles)
            estimates = l.estimates()

            # validation over mvp column
            if (self._validation and self._mvp):
                if (estimates is not None):
                    # allow selection for rows with estimations
                    dv_mvp_list.ranges.append(cells('C', (row, row)))
                else:
                    # no mvp selection is allowed for non-estimate rows
                    dv_mvp_empty.ranges.append(cells('C', (row, row)))

            # ----------------------
            # special case for roles (they are just info rows)
            if (role):

                # collect roles rows, if roles are required
                role_rows = roles_rows.get(l.title(), None)
                if role_rows is None:
                    roles_rows[l.title()] = role_rows = []
                role_rows.append(row)

                # set row options (hide it)
                _hide_row(row, hidden=True)

                # fill with values
                _string('A', row, l.title_with_level(), f_role)  # A (title)
                _number('B', row, 0, f_role, f_multiplier)       # B (visibility/multiplier)
                _blank('C', row, f_role)                         # C (empty/MVP)
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
            mvp = multiplier and self._mvp and True

            # let's fill the row with data
            _string('A', row, l.title_with_level(), f_row)                            # A (title)
            _blank('B', row, f_row, f_multiplier)                                     # B (visibility/multiplier)
            _blank('C', row, f_row)                                                   # C (empty/MVP)
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
                _number('B', row, 1, f_row, f_multiplier) # B (visibility/multiplier)

            if (estimates is not None):
                if (self._mvp):
                    _boolean('C', row, mvp, f_row) # C (MVP)

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


        # ----------------
        # start the footer

        # footer
        row_footer = row+1

        # ----------------------------------
        # calculate ranges & apply filtering
        row_lines = row_lines.values()
        row_lines = (min(row_lines), max(max(row_lines), row_footer))

        # data validation (multiplier/filter and mvp, if enabled)
        if (self._validation):
            dv_mul_list.ranges.append(cells('B', row_lines))
            ws.add_data_validation(dv_mul_list)
            if (self._mvp):
                ws.add_data_validation(dv_mvp_empty)
                ws.add_data_validation(dv_mvp_list)

        # set up autofilters (in headers)
        ws.auto_filter.ref = '%s:%s' % (cell('A', 0), cell('K', row_footer))
        if (self._filter_visibility):
            ws.auto_filter.add_filter_column((ord('B')-ord('A')), ["1"], blank=True) # B (visibility/multiplier)

        # ---------
        # total row

        def _total(row_total, caption='Total', row_mul=cells('B', row_lines)):
            # total values (it uses row_mul to avoid duuble calculations for roles)
            _string('A', row_total, caption, f_total)                                                                 # A (caption)
            _string('B', row_total, '', f_total)                                                                      # B (hidden)
            _string('C', row_total, '', f_total)                                                                      # C
            _string('D', row_total, '', f_total)                                                                      # D
            _formula('E', row_total, '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), row_mul), f_total, f_estimates)   # E (sum)
            _formula('F', row_total, '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), row_mul), f_total, f_estimates)   # F (sum)
            _formula('G', row_total, '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), row_mul), f_total, f_estimates)   # G (sum)
            _string('H', row_total, '', f_total)                                                                      # H
            _formula('I', row_total, '=SUMPRODUCT(%s,%s)' % (cells('I', row_lines), row_mul), f_total, f_estimates)   # I (sum)
            _formula('J', row_total, '=SUMPRODUCT(%s,%s)' % (cells('J', row_lines), row_mul), f_total, f_estimates)   # J (sum)
            _formula('K', row_total, '=SUMPRODUCT(%s,%s)' % (cells('K', row_lines), row_mul), f_total, f_estimates)   # K (sum)
            return row_total

        # total values (all)
        row_total = row_footer = _total(
            row_total=row_footer + 1,
            caption=(self._mvp and 'Total (All)' or 'Total')
        )

        # total values for each role, if it's required
        if (self._roles):
            role_num = 0
            role_rows = roles_rows.items()
            role_rows.sort()
            for role, role_rows in role_rows:
                row_footer += 1
                role_num += 1
                role_column = chr(ord('L') + role_num) # new column for each role, started from 'L'

                _column(role_column, width=10, f=self._theme.F_DEFAULT)
                _hide_column(role_column, hidden=True)

                _string(role_column, 0, role, f_header)
                for role_row in role_rows:
                    _number(role_column, role_row, 1)

                role_mul = cells(role_column, row_lines)
                _string('A', row_footer, '  - %s' % role.strip('()'), f_total)                                            # A (caption)
                _string('B', row_footer, '', f_total)                                                                     # B (hidden)
                _string('C', row_footer, '', f_total)                                                                     # C
                _string('D', row_footer, '', f_total)                                                                     # D
                _formula('E', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('E', row_lines), role_mul), f_total, f_estimates) # E (sum)
                _formula('F', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('F', row_lines), role_mul), f_total, f_estimates) # F (sum)
                _formula('G', row_footer, '=SUMPRODUCT(%s,%s)' % (cells('G', row_lines), role_mul), f_total, f_estimates) # G (sum)

        if (self._mvp):
            # one extra line
            row_footer += 1

            # total (mvp) values
            # next calculation will use this row as a basis
            row_total = row_footer = _total(
                row_total=row_footer + 1,
                caption='Total (MVP)',
                row_mul="%s,%s" % (cells('B', row_lines), cells('C', row_lines))
            )

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

        if (self._mvp):
            p_title = "MVP, %s" % p_title

        # Min (P=95/99%)
        row_footer += 1
        _string('A', row_footer, 'Min (%s)' % p_title, f_total)  # A (caption)
        _string('B', row_footer, '', f_total)                    # B
        _formula('C', row_footer, '=%s-%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total, f_estimates)  # C (min)
        _formula('D', row_footer, '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final, f_estimates)                 # D (modified)

        # Max (P=95/99%)
        row_footer += 1
        _string('A', row_footer, 'Max (%s)' % p_title, f_total)  # A (caption)
        _string('B', row_footer, '', f_total)                    # B
        _formula('C', row_footer, '=%s+%s*%s' % (cell('I', row_total), p_multiplier, cell('C', row_sigma)), f_total, f_estimates)  # C (min)
        _formula('D', row_footer, '=%s*%s' % (cell('C', row_footer), cell('C', row_kappa)),  f_final, f_estimates)                 # D (modified)

    # create a report
    def report(self, root, filename):
        wb = Processor.pyxl.Workbook()
        ws = wb.active
        ws.title = 'Estimates'
        self._report(root, ws)
        if (filename):
            wb.save(filename=filename)
        return wb


# let's dance
if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='''Converts freemind estimation to xlsx report.'''
    )

    parser.add_argument(
        '--theme',
        action='store',
        dest=Processor.OPT_THEME,
        help='''use a given .py file as a theme'''
    )

    parser.add_argument(
        '--sort', '-s',
        action='store_true',
        dest=Processor.OPT_SORTING,
        help='''sort children nodes by title'''
    )

    parser.add_argument(
        '--p99',
        action='store_true',
        dest=Processor.OPT_P_99,
        help='''use P=99%% instead of P=95%%'''
    )

    parser.add_argument(
        '--mvp',
        action='store_true',
        dest=Processor.OPT_MVP,
        help='''add Minimum Viable Product (MVP) features'''
    )

    parser.add_argument(
        '--no-data-validation',
        action='store_false',
        dest=Processor.OPT_VALIDATION,
        help='''don't apply data validation for filter and MVP column'''
    )

    parser.add_argument(
        '--no-roles',
        action='store_false',
        dest=Processor.OPT_ROLES,
        help='''don't provide estimation details for each role'''
    )

    parser.add_argument(
        '--formulas',
        action='store_true',
        dest=Processor.OPT_FORMULAS,
        help='''use formulas for estimation numbers'''
    )

    # this option doesn't work for LibreOffice (due to the bug it doesn't recognize initial filter state)
    parser.add_argument(
        '--filter-visibility',
        action='store_true',
        dest=Processor.OPT_FILTER_VISIBILITY,
        help='''don't hide multiplier/visibility column, use it as a filter instead (doesn't work for LibreOffice)'''
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

