#!/usr/bin/env python

"""Setup script for packaging estimate-tools.

To build a package for distribution:
    python setup.py sdist
and upload it to the PyPI with:
    python setup.py upload

Install a link for development work:
    pip install -e .

Thee manifest.in file is used for data files.

"""

import sys
import os
import warnings

if sys.version_info < (2, 6):
    raise Exception("Python >= 2.6 is required.")
elif sys.version_info[:2] == (3, 2):
    warnings.warn("Python 3.2 is not supported")


try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command

class PyTest(Command):

    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['python',  '-m', 'unittest', 'discover'])
        raise SystemExit(errno)


setup(
    name='estimation-tools',
    version="0.0.1",
    description="A Python library to help you to generate an estimation report in xlsx format.",
    long_description="It could help you to generate a xlsx-report for a scope of work with 'Work breakdown structure' and 'Three-point estimation' techniques.",
    author="Viktor A. Danilov",
    author_email="rjabchikov.zhuj@gmail.com",
    url="https://github.com/zhuj/estimation-tools/wiki",
    license="MIT",
    packages=['estimation_tools'],
    requires=['python (>=2.6.0)'],
    install_requires=['openpyxl'],
    tests_require=['openpyxl'],
    cmdclass={'test': PyTest},
    scripts=['estimation_tools/estimate.py'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ]
)
