# Copyright 2017 The Abseil Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Abseil setup configuration."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import platform
import sys

try:
  import setuptools
except ImportError:
  from ez_setup import use_setuptools
  use_setuptools()
  import setuptools

py_version = platform.python_version_tuple()
if py_version < ('2', '7') or py_version[0] == '3' and py_version < ('3', '4'):
  raise RuntimeError('Python version 2.7 or 3.4+ is required.')

INSTALL_REQUIRES = [
    'six',
]

setuptools_version = tuple(
    int(x) for x in setuptools.__version__.split('.')[:2])

# A variety of environments have very, very old versions of setuptools that
# don't support the environment markers ("foo; python_version < X"). Since
# we're using sdist, this setup.py gets run directly when installing, so
# we can just manually do the dependency checking.
# See these for more info:
# https://github.com/abseil/abseil-py/issues/79
# https://hynek.me/articles/conditional-python-dependencies/
# Environment marker support was added in setuptools 36.2, see
# https://github.com/pypa/setuptools/blob/master/CHANGES.rst#v3620
if setuptools_version < (36, 2):
  if sys.version_info[0:2] < (3, 4):
    INSTALL_REQUIRES.append('enum34')
else:
  # Environment markers are the preferred way: it allows correct non-source
  # distributions (i.e., wheels) to be generated.
  INSTALL_REQUIRES.append("enum34; python_version < '3.4'")

_README_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'README.md')
with open(_README_PATH, 'rb') as fp:
  LONG_DESCRIPTION = fp.read().decode('utf-8')

setuptools.setup(
    name='absl-py',
    version='0.8.1',
    description=(
        'Abseil Python Common Libraries, '
        'see https://github.com/abseil/abseil-py.'),
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    author='The Abseil Authors',
    url='https://github.com/abseil/abseil-py',
    packages=setuptools.find_packages(exclude=[
        '*.tests', '*.tests.*', 'tests.*', 'tests',
    ]),
    install_requires=INSTALL_REQUIRES,
    include_package_data=True,
    license='Apache 2.0',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
