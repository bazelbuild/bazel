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

import os
import sys

try:
  import setuptools
except ImportError:
  from ez_setup import use_setuptools
  use_setuptools()
  import setuptools

if sys.version_info < (3, 6):
  raise RuntimeError('Python version 3.6+ is required.')

setuptools_version = tuple(
    int(x) for x in setuptools.__version__.split('.')[:2])

additional_kwargs = {}
if setuptools_version >= (24, 2):
  # `python_requires` was added in 24.2, see
  # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
  additional_kwargs['python_requires'] = '>=3.6'

_README_PATH = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), 'README.md')
with open(_README_PATH, 'rb') as fp:
  LONG_DESCRIPTION = fp.read().decode('utf-8')

setuptools.setup(
    name='absl-py',
    version='1.3.0',
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
    include_package_data=True,
    license='Apache 2.0',
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
    **additional_kwargs,
)
