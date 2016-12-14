#!/usr/bin/env python

"""
Setup script for packaging estimation-tools:

To build a package for distribution (sources):
    python setup.py sdist

To build a package for distribution (egg):
    python setup.py bdist_egg

To upload it to the PyPI with:
    python setup.py upload

To start test cases:
    python setup.py test

To install a link for development work:
    pip install -e .

"""

import sys
import warnings


# check python version
if sys.version_info < (2, 6):
    raise Exception("Python >= 2.6 is required.")
elif sys.version_info[:2] == (3, 2):
    warnings.warn("Python 3.2 is not supported")


# import setup
try:
    from setuptools import setup, Command
except ImportError:
    from distutils.core import setup, Command


# it wraps 'test' command
class PyTest(Command):

    # command class must provide 'user_options' attribute (a list of tuples)
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import subprocess
        errno = subprocess.call(['python',  '-m', 'unittest', 'discover'])
        raise SystemExit(errno)


# read readme file
try:
    import os
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md'), 'r') as f:
        long_description = f.read()
except:
    long_description = ''


# let's go
setup(
    name='estimation-tools',
    version="1.0.0",
    description="A Python tool to generate a work breakdown structured three-point estimation report from mind map file.",
    long_description=long_description,
    author="Viktor A. Danilov",
    author_email="rjabchikov.zhuj@gmail.com",
    url="https://github.com/zhuj/estimation-tools/wiki",
    license="MIT",
    packages=[], # Maybe later: packages=['estimation_tools'],
    requires=['python (>=2.6.0)'],
    install_requires=['openpyxl (>=2.4.1)'],
    tests_require=['openpyxl (>=2.4.1)'],
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
