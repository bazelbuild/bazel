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

"""Tests for converter.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

from absl import logging as absl_logging
from absl.logging import converter
from absl.testing import absltest


class ConverterTest(absltest.TestCase):
  """Tests the converter module."""

  def test_absl_to_cpp(self):
    self.assertEqual(0, converter.absl_to_cpp(absl_logging.DEBUG))
    self.assertEqual(0, converter.absl_to_cpp(absl_logging.INFO))
    self.assertEqual(1, converter.absl_to_cpp(absl_logging.WARN))
    self.assertEqual(2, converter.absl_to_cpp(absl_logging.ERROR))
    self.assertEqual(3, converter.absl_to_cpp(absl_logging.FATAL))

    with self.assertRaises(TypeError):
      converter.absl_to_cpp('')

  def test_absl_to_standard(self):
    self.assertEqual(
        logging.DEBUG, converter.absl_to_standard(absl_logging.DEBUG))
    self.assertEqual(
        logging.INFO, converter.absl_to_standard(absl_logging.INFO))
    self.assertEqual(
        logging.WARNING, converter.absl_to_standard(absl_logging.WARN))
    self.assertEqual(
        logging.WARN, converter.absl_to_standard(absl_logging.WARN))
    self.assertEqual(
        logging.ERROR, converter.absl_to_standard(absl_logging.ERROR))
    self.assertEqual(
        logging.FATAL, converter.absl_to_standard(absl_logging.FATAL))
    self.assertEqual(
        logging.CRITICAL, converter.absl_to_standard(absl_logging.FATAL))
    # vlog levels.
    self.assertEqual(9, converter.absl_to_standard(2))
    self.assertEqual(8, converter.absl_to_standard(3))

    with self.assertRaises(TypeError):
      converter.absl_to_standard('')

  def test_standard_to_absl(self):
    self.assertEqual(
        absl_logging.DEBUG, converter.standard_to_absl(logging.DEBUG))
    self.assertEqual(
        absl_logging.INFO, converter.standard_to_absl(logging.INFO))
    self.assertEqual(
        absl_logging.WARN, converter.standard_to_absl(logging.WARN))
    self.assertEqual(
        absl_logging.WARN, converter.standard_to_absl(logging.WARNING))
    self.assertEqual(
        absl_logging.ERROR, converter.standard_to_absl(logging.ERROR))
    self.assertEqual(
        absl_logging.FATAL, converter.standard_to_absl(logging.FATAL))
    self.assertEqual(
        absl_logging.FATAL, converter.standard_to_absl(logging.CRITICAL))
    # vlog levels.
    self.assertEqual(2, converter.standard_to_absl(logging.DEBUG - 1))
    self.assertEqual(3, converter.standard_to_absl(logging.DEBUG - 2))

    with self.assertRaises(TypeError):
      converter.standard_to_absl('')

  def test_standard_to_cpp(self):
    self.assertEqual(0, converter.standard_to_cpp(logging.DEBUG))
    self.assertEqual(0, converter.standard_to_cpp(logging.INFO))
    self.assertEqual(1, converter.standard_to_cpp(logging.WARN))
    self.assertEqual(1, converter.standard_to_cpp(logging.WARNING))
    self.assertEqual(2, converter.standard_to_cpp(logging.ERROR))
    self.assertEqual(3, converter.standard_to_cpp(logging.FATAL))
    self.assertEqual(3, converter.standard_to_cpp(logging.CRITICAL))

    with self.assertRaises(TypeError):
      converter.standard_to_cpp('')

  def test_get_initial_for_level(self):
    self.assertEqual('F', converter.get_initial_for_level(logging.CRITICAL))
    self.assertEqual('E', converter.get_initial_for_level(logging.ERROR))
    self.assertEqual('W', converter.get_initial_for_level(logging.WARNING))
    self.assertEqual('I', converter.get_initial_for_level(logging.INFO))
    self.assertEqual('I', converter.get_initial_for_level(logging.DEBUG))
    self.assertEqual('I', converter.get_initial_for_level(logging.NOTSET))

    self.assertEqual('F', converter.get_initial_for_level(51))
    self.assertEqual('E', converter.get_initial_for_level(49))
    self.assertEqual('E', converter.get_initial_for_level(41))
    self.assertEqual('W', converter.get_initial_for_level(39))
    self.assertEqual('W', converter.get_initial_for_level(31))
    self.assertEqual('I', converter.get_initial_for_level(29))
    self.assertEqual('I', converter.get_initial_for_level(21))
    self.assertEqual('I', converter.get_initial_for_level(19))
    self.assertEqual('I', converter.get_initial_for_level(11))
    self.assertEqual('I', converter.get_initial_for_level(9))
    self.assertEqual('I', converter.get_initial_for_level(1))
    self.assertEqual('I', converter.get_initial_for_level(-1))

  def test_string_to_standard(self):
    self.assertEqual(logging.DEBUG, converter.string_to_standard('debug'))
    self.assertEqual(logging.INFO, converter.string_to_standard('info'))
    self.assertEqual(logging.WARNING, converter.string_to_standard('warn'))
    self.assertEqual(logging.WARNING, converter.string_to_standard('warning'))
    self.assertEqual(logging.ERROR, converter.string_to_standard('error'))
    self.assertEqual(logging.CRITICAL, converter.string_to_standard('fatal'))

    self.assertEqual(logging.DEBUG, converter.string_to_standard('DEBUG'))
    self.assertEqual(logging.INFO, converter.string_to_standard('INFO'))
    self.assertEqual(logging.WARNING, converter.string_to_standard('WARN'))
    self.assertEqual(logging.WARNING, converter.string_to_standard('WARNING'))
    self.assertEqual(logging.ERROR, converter.string_to_standard('ERROR'))
    self.assertEqual(logging.CRITICAL, converter.string_to_standard('FATAL'))


if __name__ == '__main__':
  absltest.main()
