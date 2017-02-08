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

    def __set_parent(self, parent, deep=True):
        if (parent is None):
            self._parent = None
            self._level = -1
        else:
            self._parent = parent
            self._level = 1 + self._parent.level()

        # refresh child nodes, if required
        if (deep and len(self._childs) > 0):
            for c in self._childs:
                c.__set_parent(self, deep=True)

    def __init__(self, parent, title):
        self.__set_parent(parent, deep=False)
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
        node.__set_parent(self)
        self._childs.append(node)
        return node

    def detach(self):
        self._parent._childs.remove(self)
        self.__set_parent(None)
        return self

    def childs(self):
        return self._childs

    def annotation(self, name, value=None):
        l = self._annotations.get(name, [])
        if (value is None): return l
        value = value.strip()
        if value:
            l.append(value)
            self._annotations[name] = l

    def mvp_minus(self):
        return len(self.annotation('mvp-')) > 0

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

    def trace(self, stoppers):
        if (self in stoppers): return [ ]
        if (self._parent is None): return [ self ]
        trace = self._parent.trace(stoppers)
        trace.append(self)
        return trace

    def acquire(self, name):
        v = getattr(self, name, None)
        if (v is not None): return v
        if (self._parent is not None): return self._parent.acquire(name)
        return None

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

    # default number format
    DEFAULT_F_PERCENTAGE = lambda self: self._merge_format(self.F_NUMBERS, {
        'num_format': '0%'
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
    OPT_FACTOR = 'factor'
    OPT_THEME = 'theme'
    OPT_SORTING = 'sorting'
    OPT_P_99 = 'p99'
    OPT_MVP = 'mvp'
    OPT_ROLES = 'roles'
    OPT_VALIDATION = 'validation'
    OPT_FORMULAS = 'formulas'
    OPT_FILTER_VISIBILITY = 'filter_visibility'
    OPT_FACTORS = 'factors'
    OPT_ARROWS = 'arrows'
    OPT_STAGES = 'stages'

    # regexps patterns
    import re
    RE_ESTIMATE = re.compile("estimate\\s*=(\\s*\\d+\\s*)[/](\\s*\\d+\\s*)[/](\\s*\\d+\\s*)")
    RE_ANNOTATION = re.compile("^[[]([a-z0-9-]+)[]]\\s*(.*)")

    # annotation types
    ANNOTATIONS = {
        'warn': ('comment', '(!) '),
        'idea': ('comment', ''),
        'todo': ('comment', ''),
        'impl': ('comment', ''),

        'mvp-': ('mvp-', '*'),
        'api+': ('api+', '*'),
        'stage': ('stage', ''),
        'phase': ('stage', ''),
        'module': ('module', '')
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
    @staticmethod
    def _parse_factors(factors):
        if (factors):
            factors = factors.split(',')
            factors = [ x.split(':') for x in factors ]
            factors = { x.strip():float(y.strip()) for x, y in factors }
            return factors
        return None

    #
    def __init__(self, options):
        options = Processor._wrap_options(options)
        self._factor = float(getattr(options, Processor.OPT_FACTOR, None) or "1.0")
        self._theme = Processor._loadTheme(getattr(options, Processor.OPT_THEME, None))
        self._sorting = getattr(options, Processor.OPT_SORTING, False) and True
        self._validation = getattr(options, Processor.OPT_VALIDATION, False) and True
        self._mvp = getattr(options, Processor.OPT_MVP, False) and True
        self._p99 = getattr(options, Processor.OPT_P_99, False) and True
        self._roles = getattr(options, Processor.OPT_ROLES, False) and True
        self._formulas = self._roles and (getattr(options, Processor.OPT_FORMULAS, False) and True)
        self._filter_visibility = self._roles and (getattr(options, Processor.OPT_FILTER_VISIBILITY, False) and True)
        self._factors = Processor._parse_factors(getattr(options, Processor.OPT_FACTORS, None))
        self._stages = getattr(options, Processor.OPT_STAGES, False) and True
        self._arrows = getattr(options, Processor.OPT_ARROWS, False) and True

        if (self._stages): self._mvp = False

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
                estimates = [ self._factor * float(match.group(x).strip()) for x in (1,2,3) ]
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
    def transform_stages(self, tree):
        """ it extracts stages (from annotations) and rearrange nodes according their stages """

        _node_cache = {}
        def _add_node(path, caption=lambda x:x[-1]):
            if (not path): return tree

            v = _node_cache.get(path, None)
            if (v is not None): return v

            r = _add_node(path[:-1], caption)
            e = r.append( Node(r, caption(path)) )
            _node_cache[path] = e
            return e

        lines = Processor._collect(root)
        lines = [ ( l, s.strip() ) for l in lines for s in l.annotation('stage') ]
        lines = [ ( l, tuple([s.strip().lower().capitalize() for s in sl.split('.')]) ) for (l, sl) in lines if (s and not l.is_role()) ]
        # XXX: think about: lines.reverse() # not it's backward hierarchy sorted

        # append stages in right order
        stages = list(set([ sl for (l, sl) in lines ]))
        stages.sort()
        for sl in stages:
            s = _add_node(sl, caption=lambda x:'Stage %s' % '.'.join(x))
            s._stage = sl
        del stages

        # full list of stoppers
        stoppers = [tree] + _node_cache.values()

        # process nodes
        import copy
        for l, sl in lines:
            s = _node_cache.get(sl)
            trace = l.trace(stoppers)[:-1]

            for t in trace:
                sl = sl + (t.title(), )
                s = _add_node(sl)
                s._annotations = copy.copy(t._annotations)

            s.append(l.detach())

        def _cleanup(node):
            if (node.is_role()): return node
            if (node.estimates() is not None): return node

            childs = [ _cleanup(c) for c in node._childs ]
            childs = [ c for c in childs if c is not None ]
            if (len(childs) > 0):
                node._childs = childs
                return node

            return None

        tree = _cleanup(tree)
        return tree

    #
    def transform(self, tree):
        """ it applies transformations to the tree just before its reporting """
        if (self._stages): tree = self.transform_stages(tree)
        return tree

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
        f_percentage = self._theme.F_PERCENTAGE
        f_role = self._theme.F_ROLE_ROW
        f_total = self._theme.F_TOTAL
        f_final = self._theme.F_FINAL
        f_multiplier = self._theme.F_MULTIPLIER

        # --------------
        # define columns

        def _cell_name(number, alphabet='ABCDEFGHIJKLMNOPQRSTUVWXYZ'):
            base = ''
            while number:
                number, i = divmod(number, len(alphabet))
                base = alphabet[i] + base
            return base or alphabet[0]

        def _cell_names(i=0):
            while True:
                yield _cell_name(i)
                i += 1

        _cells = _cell_names()

        B0 = _cells.next()  # A: caption
        B1 = _cells.next()  # B: visibility
        B2 = _cells.next()  # C: empty/MVP/Stage

        S0 = _cells.next()  # D: structure1 (module, submodule, ...)
        S1 = _cells.next()  # E: structure2 (...)
        S2 = _cells.next()  # F: structure3 (...)
        S3 = _cells.next()  # G: structure4 (...)

        C0 = _cells.next()  # H: comment

        E0 = _cells.next()  # I: estimate-0
        E1 = _cells.next()  # J: estimate-1
        E2 = _cells.next()  # K: estimate-2
        E3 = _cells.next()  # L: estimate-separator
        E4 = _cells.next()  # M: estimate-wm
        E5 = _cells.next()  # N: estimate-st
        E6 = _cells.next()  # O: estimate-sq

        R0 = _cells.next()  # P: role name

        del _cells

        # ------------
        # setup header

        # setup columns: base columns
        _column(B0, width=40, f=self._theme.F_DEFAULT)
        _column(B1, width=3,  f=self._theme.F_MULTIPLIER)
        _column(B2, width=8,  f=self._theme.F_DEFAULT)
        _column(S0, width=10, f=self._theme.F_COMMENT)
        _column(S1, width=20, f=self._theme.F_COMMENT)
        _column(S2, width=20, f=self._theme.F_COMMENT)
        _column(S3, width=20, f=self._theme.F_COMMENT)
        _column(C0, width=50, f=self._theme.F_COMMENT)
        _column(E0, width=8,  f=self._theme.F_NUMBERS)
        _column(E1, width=8,  f=self._theme.F_NUMBERS)
        _column(E2, width=8,  f=self._theme.F_NUMBERS)
        _column(E3, width=4,  f=self._theme.F_DEFAULT)
        _column(E4, width=8,  f=self._theme.F_NUMBERS)
        _column(E5, width=8,  f=self._theme.F_NUMBERS)
        _column(E6, width=8,  f=self._theme.F_NUMBERS)
        _column(R0, width=3,  f=self._theme.F_DEFAULT)

        # hide visibility if required
        if (not self._filter_visibility):
            _hide_column(B1, hidden=True)

        # hide structure columns
        _hide_column(S0, hidden=True)
        _hide_column(S1, hidden=True)
        _hide_column(S2, hidden=True)
        _hide_column(S3, hidden=True)

        # hide role column
        _hide_column(R0, hidden=True)

        # start rows
        row = 0

        # header (row = 1)
        f_header = self._theme.F_HEADER
        _string(B0, row, 'Task / Subtask', f_header)  # B0: caption
        _string(B1, row, 'Filter', f_header)          # B1: visibility
        _string(B2, row, '', f_header)                # B2: empty/MVP/Stage
        _string(S0, row, 'Module', f_header)          # S0: structure (module, submodule, ....)
        _string(S1, row, '', f_header)                # S1: structure
        _string(S2, row, '', f_header)                # S2: structure
        _string(S3, row, '', f_header)                # S3: structure
        _string(C0, row, 'Comment', f_header)         # C0: comment
        _string(E0, row, 'Min', f_header)             # E0: estimate
        _string(E1, row, 'Real', f_header)            # E1: estimate
        _string(E2, row, 'Max', f_header)             # E2: estimate
        _string(E3, row, '', f_header)                # E3: empty
        _string(E4, row, 'Avg', f_header)             # E4: weighted mean
        _string(E5, row, 'SD', f_header)              # E5: standard deviation
        _string(E6, row, 'Sq', f_header)              # E6: squared deviation
        _string(R0, row, 'Role', f_header)            # R0: role name

        if (self._mvp):
            _string(B2, row, 'MVP', f_header)         # B2: MVP
        elif (self._stages):
            _string(B2, row, 'Stage', f_header)       # B2: Stage

        # ------------------------
        # prepare data validation

        if (self._validation):

            # validation values for B (multiplier) and C (mvp)
            _hide_column('AZ', hidden=True)     # hide AZ
            _blank('AZ', 0, f_multiplier)       # AZ1 = empty
            _number('AZ', 1, 0, f_multiplier)   # AZ2 = 0
            _number('AZ', 2, 1, f_multiplier)   # AZ3 = 1

            # multiplier
            # XXX: note, the validation here is general only (for the whole column)
            # XXX: the idea is that user wants to control which lines are raw data source and which are representation only
            dv_mul_list = Processor.pyxl.worksheet.datavalidation.DataValidation(
                type="list",
                formula1='$AZ$1:$AZ$3', # '', 0, 1
                allow_blank=True
            )
            dv_mul_list.hide_drop_down = True

            # mvp (selecatable)
            dv_mvp_list = Processor.pyxl.worksheet.datavalidation.DataValidation(
                type="list",
                formula1='$AZ$2:$AZ$3', # 0, 1
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

        # ------
        # stages

        def _stage_to_string(node):
            if (self._stages):
                stage = node.acquire('_stage')
                if (stage):
                    return '%s' % ('.'.join(str(x) for x in stage))
            return ''

        # ------------------------
        # start the transformation

        # lines
        stages = set()
        roles = set()
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
                    dv_mvp_list.ranges.append(cells(B2, (row, row)))
                else:
                    # no mvp selection is allowed for non-estimate rows
                    dv_mvp_empty.ranges.append(cells(B2, (row, row)))

            # ----------------------
            # special case for roles (they are just info rows)
            if (role):

                # collect roles rows, if roles are required
                roles.add(l.title())

                # set row options (hide it)
                _hide_row(row, hidden=True)

                # fill with values
                _string(B0, row, l.title_with_level(), f_role)          # B0 (title)
                _number(B1, row, 0, f_role, f_multiplier)               # B1 (visibility/multiplier)
                _blank(B2, row, f_role)                                 # B2 (empty/MVP/Stage)
                _formula(R0, row, '=TRIM(%s)' % cell(B0, row), f_row)   # R0 (role)
                _string(S0, row, '', f_role)                            # S0 (module, submodule, ...)
                _string(S1, row, '', f_role)                            # S1 (comment)
                _string(S2, row, '', f_role)                            # S2 (comment)
                _string(S3, row, '', f_role)                            # S3 (comment)
                _string(C0, row, '', f_role)                            # C0 (comment)
                if (estimates is not None):
                    _number(E0, row, estimates[0], f_role)              # E0 (estimate)
                    _number(E1, row, estimates[1], f_role)              # E1 (estimate)
                    _number(E2, row, estimates[2], f_role)              # E2 (estimate)

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

            # multiplier will be used in SUMIFS formulas:
            # '1' means that it's a data for formulas - we have to use it,
            # '' means that it's a role (don't use it in total numbers - use aggregated estimations)
            multiplier = (not role) and (estimates is not None)

            # let's fill the row with data
            _string(B0, row, l.title_with_level(), f_row)                            # B0 (title)
            _blank(B1, row, f_row, f_multiplier)                                     # B1 (visibility/multiplier)
            _blank(B2, row, f_row)                                                   # B2 (empty/MVP/Stage)
            _blank(R0, row, f_row)                                                   # R0 (role)
            _string(S0, row, '', f_row)                                              # S0 (module, submodule, ...)
            _string(S1, row, '', f_row)                                              # S1 (comment)
            _string(S2, row, '', f_row)                                              # S2 (comment)
            _string(S3, row, '', f_row)                                              # S3 (comment)
            _string(C0, row, ';\n'.join(l.annotation('comment')), f_row, f_comment)  # C0 (comment)
            _string(E0, row, '', f_row)                                              # E0 (estimate)
            _string(E1, row, '', f_row)                                              # E1 (estimate)
            _string(E2, row, '', f_row)                                              # E2 (estimate)
            _blank(E3, row, f_row)                                                   # E3 (empty)
            _string(E4, row, '', f_row)                                              # E4 (estimate)
            _string(E5, row, '', f_row)                                              # E5 (estimate)
            _string(E6, row, '', f_row)                                              # E6 (estimate)

            # setup visibility/multiplier
            if (multiplier):
                _number(B1, row, 1, f_row, f_multiplier) # B1 (visibility/multiplier)

            # setup MVP/Stage for non-role rows
            if (self._mvp):
                if (estimates is not None):
                    mvp = multiplier and self._mvp and True
                    mvp = mvp and (not l.mvp_minus()) and True
                    _boolean(B2, row, mvp, f_row) # B2 (MVP, estimate rows only)
            elif (self._stages):
                stage = _stage_to_string(l)
                if (not stage): stage = "(?)"
                stages.add(stage)
                if (estimates is not None):
                    _string(B2, row, stage, f_row) # B2 (Stage, all rows)
                else:
                    _string(B2, row, stage, f_role) # B2 (Stage, all rows)

            # fill estimates
            if (estimates is not None):
                _number(E0, row, estimates[0], f_row, f_estimates)               # E0 (estimate)
                _number(E1, row, estimates[1], f_row, f_estimates)               # E1 (estimate)
                _number(E2, row, estimates[2], f_row, f_estimates)               # E2 (estimate)
                _formula(E4, row, '=(%s+4*%s+%s)/6' % (cell(E0, row), cell(E1, row), cell(E2, row)), f_row, f_estimates)  # E1 (weighted mean)
                _formula(E5, row, '=(%s-%s)/6' % (cell(E2, row), cell(E0, row)), f_row, f_estimates)                      # E2 (standard deviation)
                _formula(E6, row, '=%s*%s' % (cell(E5, row), cell(E5, row)), f_row, f_estimates)                          # E3 (squared deviation)


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
                    _formula(E0, l_row, template.replace('#', E0), f_row, f_estimates)  # E0 (estimate)
                    _formula(E1, l_row, template.replace('#', E1), f_row, f_estimates)  # E1 (estimate)
                    _formula(E2, l_row, template.replace('#', E2), f_row, f_estimates)  # E2 (estimate)
                else:
                    _formula(E0, l_row, template.replace('#', E0), f_row, f_role) # E0 (estimate)
                    _formula(E1, l_row, template.replace('#', E1), f_row, f_role) # E1 (estimate)
                    _formula(E2, l_row, template.replace('#', E2), f_row, f_role) # E2 (estimate)


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
            dv_mul_list.ranges.append(cells(B1, row_lines))
            ws.add_data_validation(dv_mul_list)
            if (self._mvp):
                ws.add_data_validation(dv_mvp_empty)
                ws.add_data_validation(dv_mvp_list)

        # set up autofilters (in headers)
        ws.auto_filter.ref = '%s:%s' % (cell(B0, 0), cell(E6, row_footer))
        if (self._filter_visibility):
            ws.auto_filter.add_filter_column((ord(B1)-ord('A')), ["1"], blank=True) # B1 (visibility/multiplier)

        # ---------
        # total row

        def _total(row_total, caption='Total', row_criteria='%s,"=1"' % (cells(B1, row_lines))):
            # total values (it uses row_mul to avoid duuble calculations for roles)
            _string(B0, row_total, caption, f_total)                                                                 # B0 (caption)
            _string(B1, row_total, '', f_total)                                                                      # B1 (hidden)
            _string(B2, row_total, '', f_total)                                                                      # B2
            _string(S0, row_total, '', f_total)                                                                      # S0
            _string(S1, row_total, '', f_total)                                                                      # S1
            _string(S2, row_total, '', f_total)                                                                      # S2
            _string(S3, row_total, '', f_total)                                                                      # S3
            _string(C0, row_total, '', f_total)                                                                      # C0
            _formula(E0, row_total, '=SUMIFS(%s,%s)' % (cells(E0, row_lines), row_criteria), f_total, f_estimates)   # E0 (sum)
            _formula(E1, row_total, '=SUMIFS(%s,%s)' % (cells(E1, row_lines), row_criteria), f_total, f_estimates)   # E1 (sum)
            _formula(E2, row_total, '=SUMIFS(%s,%s)' % (cells(E2, row_lines), row_criteria), f_total, f_estimates)   # E2 (sum)
            _string(E3, row_total, '', f_total)                                                                      # E3
            _formula(E4, row_total, '=SUMIFS(%s,%s)' % (cells(E4, row_lines), row_criteria), f_total, f_estimates)   # E4 (sum)
            _formula(E5, row_total, '=SUMIFS(%s,%s)' % (cells(E5, row_lines), row_criteria), f_total, f_estimates)   # E5 (sum)
            _formula(E6, row_total, '=SUMIFS(%s,%s)' % (cells(E6, row_lines), row_criteria), f_total, f_estimates)   # E6 (sum)
            return row_total

        def _partial(row_total, row_footer, caption, row_criteria='%s,"=1"' % (cells(B1, row_lines))):
            # partial total row
            _string(B0, row_footer, caption, f_total)                                                                # B0 (caption)
            _string(B1, row_footer, '', f_total)                                                                     # B1 (hidden)
            _string(B2, row_footer, '', f_total)                                                                     # B2
            _string(S0, row_footer, '', f_total)                                                                     # S0
            _string(S1, row_footer, '', f_total)                                                                     # S1
            _string(S2, row_footer, '', f_total)                                                                     # S2
            _string(S3, row_footer, '', f_total)                                                                     # S3
            _string(C0, row_footer, '', f_total)                                                                     # C0
            _formula(E0, row_footer, '=SUMIFS(%s,%s)' % (cells(E0, row_lines), row_criteria), f_total, f_estimates)  # E0 (sum)
            _formula(E1, row_footer, '=SUMIFS(%s,%s)' % (cells(E1, row_lines), row_criteria), f_total, f_estimates)  # E1 (sum)
            _formula(E2, row_footer, '=SUMIFS(%s,%s)' % (cells(E2, row_lines), row_criteria), f_total, f_estimates)  # E2 (sum)
            _formula(E4, row_footer, '=(%s+4*%s+%s)/6' % (cell(E0, row_footer), cell(E1, row_footer), cell(E2, row_footer)), f_total, f_estimates) # E4 (total)
            _formula(E5, row_footer, '=(%s/%s)' % (cell(E4, row_footer), cell(E4, row_total)), f_percentage)         # E5 (%)
            return row_footer

        # total values (all)
        row_total = row_footer = _total(
            row_total=row_footer + 1,
            caption=(self._mvp and 'Total (All)' or 'Total')
        )

        # total values for each role, if it's required
        if (self._roles):
            roles = list(roles)
            roles.sort()
            for role in roles:
                row_footer += 1
                row_footer = _partial(
                    row_total=row_total,
                    row_footer=row_footer,
                    caption='  - %s' % role.strip('()'),
                    row_criteria='''%s,"=0",%s,"%s"''' % (cells(B1, row_lines), cells(R0, row_lines), role)
                )
            del roles

        if (self._mvp):
            # one extra line
            row_footer += 1

            # total (mvp) values
            # next calculation will use this row as a basis
            row_total = row_footer = _total(
                row_total=row_footer + 1,
                caption='Total (MVP)',
                row_criteria='%s,"=1",%s,"=1"' % (cells(B1, row_lines), cells(B2, row_lines))
            )

        elif (self._stages):

            # one extra line
            row_footer += 1

            # total (stage) values
            stages = list(stages)
            stages.sort()
            for stage in stages:
                row_footer += 1
                row_footer = _partial(
                    row_total=row_total,
                    row_footer=row_footer,
                    caption=' - Stage %s' % stage,
                    row_criteria='%s,"=1",%s,"=%s"' % (cells(B1, row_lines), cells(B2, row_lines), stage)
                )
            del stages

            # filter (TODO)
            # ws.auto_filter.add_filter_column((ord(B2)-ord('A')), ["1"], blank=True)

        # one extra line
        row_footer += 1

        # sigma: standard deviation
        row_footer += 1
        row_sigma = row_footer
        _string(B0, row_sigma, 'Standard deviation', f_caption)                     # B0 (caption)
        _formula(B2, row_sigma, '=SQRT(%s)' % (cell(E6, row_total)), f_estimates)   # B2 (sigma)

        # factors
        if (self._factors):
            row_footer += 1
            _string(B0, row_footer, 'K:', f_caption)                   # B0 (caption)

            kappa_rows = []

            # base factor
            row_footer += 1
            _string(B0, row_footer, ' = base', f_caption)              # B0 (caption)
            _number(B2, row_footer, 1.0, f_estimates)                  # B2 (kappa)
            kappa_rows.append(row_footer)

            # factors (from options)
            for f, v in self._factors.items():
                row_footer += 1
                _string(B0, row_footer, ' + %s' % f, f_caption)        # B0 (caption)
                _number(B2, row_footer, v, f_estimates, f_percentage)  # B2 (kappa)
                kappa_rows.append(row_footer)

            # correction factor (other)
            row_footer += 1
            _string(B0, row_footer, ' + other', f_caption)             # B0 (caption)
            _number(B2, row_footer, 0.0, f_estimates, f_percentage)    # B2 (kappa)
            kappa_rows.append(row_footer)

            # all together
            kappa_rows = [ min(kappa_rows), max(kappa_rows) ]

            # correction factor (total multiplier)
            row_footer += 1
            _string(B0, row_footer, ' * correction', f_caption)        # B0 (caption)
            _number(B2, row_footer, 1.0, f_estimates)                  # B2 (kappa)
            kappa_rows.append(row_footer)

            # kappa: correction factor (total, formula)
            row_footer += 1
            row_kappa = row_footer
            _string(B0, row_kappa, 'K (total)', f_caption)             # B0 (caption)
            _formula(B2, row_kappa, '=SUM(%s)*%s' % (cells(B2, kappa_rows[0:2]), cell(B2, kappa_rows[2])), f_estimates)  # B2 (kappa)
            del kappa_rows
        else:
            # kappa: correction factor
            row_footer += 1
            row_kappa = row_footer
            _string(B0, row_kappa, 'K', f_caption)     # B0 (caption)
            _number(B2, row_kappa, 1.5, f_estimates)  # B2 (kappa)

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

        # empty line
        row_footer += 1

        # Min (P=95/99%)
        row_footer += 1
        _string(B0, row_footer, 'Min (%s)' % p_title, f_total)  # B0 (caption)
        _string(B1, row_footer, '', f_total)                    # B1
        _formula(B2, row_footer, '=%s-%s*%s' % (cell(E4, row_total), p_multiplier, cell(B2, row_sigma)), f_total, f_estimates)  # B2 (min)
        _string(S0, row_footer, '', f_total)                                                                                    # S0
        _string(S1, row_footer, '', f_total)                                                                                    # S1
        _string(S2, row_footer, '', f_total)                                                                                    # S2
        _string(S3, row_footer, '', f_total)                                                                                    # S3
        _formula(C0, row_footer, '=%s*%s' % (cell(B2, row_footer), cell(B2, row_kappa)),  f_final, f_estimates)                 # C0 (modified)

        # Max (P=95/99%)
        row_footer += 1
        _string(B0, row_footer, 'Max (%s)' % p_title, f_total)  # B0 (caption)
        _string(B1, row_footer, '', f_total)                    # B1
        _formula(B2, row_footer, '=%s+%s*%s' % (cell(E4, row_total), p_multiplier, cell(B2, row_sigma)), f_total, f_estimates)  # B2 (max)
        _string(S0, row_footer, '', f_total)                                                                                    # S0
        _string(S1, row_footer, '', f_total)                                                                                    # S1
        _string(S2, row_footer, '', f_total)                                                                                    # S2
        _string(S3, row_footer, '', f_total)                                                                                    # S3
        _formula(C0, row_footer, '=%s*%s' % (cell(B2, row_footer), cell(B2, row_kappa)),  f_final, f_estimates)                 # C0 (modified)

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
        '--factor',
        action='store',
        dest=Processor.OPT_FACTOR,
        help='''use a given factor for estimates (default = 1.0)'''
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

    # this option is required for SharePoint
    parser.add_argument(
        '--no-data-validation',
        action='store_false',
        dest=Processor.OPT_VALIDATION,
        help='''don't apply data validation for filter and MVP column (SharePoint fix)'''
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

    # TODO: implement me
    parser.add_argument(
        '--arrows',
        action='store_true',
        dest=Processor.OPT_ARROWS,
        help='''transformation: handle arrows as dependency indicators'''
    )

    parser.add_argument(
        '--stages',
        action='store_true',
        dest=Processor.OPT_STAGES,
        help='''transformation: use stages markers as nodes/subtree groups (replaces MVP feature)'''
    )

    parser.add_argument(
        '--factors',
        action='store',
        dest=Processor.OPT_FACTORS,
        help='''use extra factors with default values (in format f1:v1,f2:v2,f3:v3,...)'''
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
    root = processor.transform(root)
    processor.report(root, options.output or (filename + ".xlsx"))

