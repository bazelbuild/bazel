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

"""Additional tests for Flag classes.

Most of the Flag classes are covered in the flags_test.py.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import pickle

from absl._enum_module import enum
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _flag
from absl.testing import absltest
from absl.testing import parameterized


class FlagTest(absltest.TestCase):

  def setUp(self):
    self.flag = _flag.Flag(
        _argument_parser.ArgumentParser(),
        _argument_parser.ArgumentSerializer(),
        'fruit', 'apple', 'help')

  def test_default_unparsed(self):
    flag = _flag.Flag(
        _argument_parser.ArgumentParser(),
        _argument_parser.ArgumentSerializer(),
        'fruit', 'apple', 'help')
    self.assertEqual('apple', flag.default_unparsed)

    flag = _flag.Flag(
        _argument_parser.IntegerParser(),
        _argument_parser.ArgumentSerializer(),
        'number', '1', 'help')
    self.assertEqual('1', flag.default_unparsed)

    flag = _flag.Flag(
        _argument_parser.IntegerParser(),
        _argument_parser.ArgumentSerializer(),
        'number', 1, 'help')
    self.assertEqual(1, flag.default_unparsed)

  def test_set_default_overrides_current_value(self):
    self.assertEqual('apple', self.flag.value)
    self.flag._set_default('orange')
    self.assertEqual('orange', self.flag.value)

  def test_set_default_overrides_current_value_when_not_using_default(self):
    self.flag.using_default_value = False
    self.assertEqual('apple', self.flag.value)
    self.flag._set_default('orange')
    self.assertEqual('apple', self.flag.value)

  def test_pickle(self):
    with self.assertRaisesRegexp(TypeError, "can't pickle Flag objects"):
      pickle.dumps(self.flag)

  def test_copy(self):
    self.flag.value = 'orange'

    with self.assertRaisesRegexp(
        TypeError, 'Flag does not support shallow copies'):
      copy.copy(self.flag)

    flag2 = copy.deepcopy(self.flag)
    self.assertEqual(flag2.value, 'orange')

    flag2.value = 'mango'
    self.assertEqual(flag2.value, 'mango')
    self.assertEqual(self.flag.value, 'orange')


class BooleanFlagTest(parameterized.TestCase):

  @parameterized.parameters(('', '(no help available)'),
                            ('Is my test brilliant?', 'Is my test brilliant?'))
  def test_help_text(self, helptext_input, helptext_output):
    f = _flag.BooleanFlag('a_bool', False, helptext_input)
    self.assertEqual(helptext_output, f.help)


class EnumFlagTest(parameterized.TestCase):

  @parameterized.parameters(
      ('', '<apple|orange>: (no help available)'),
      ('Type of fruit.', '<apple|orange>: Type of fruit.'))
  def test_help_text(self, helptext_input, helptext_output):
    f = _flag.EnumFlag('fruit', 'apple', helptext_input, ['apple', 'orange'])
    self.assertEqual(helptext_output, f.help)

  def test_empty_values(self):
    with self.assertRaises(ValueError):
      _flag.EnumFlag('fruit', None, 'help', [])


class Fruit(enum.Enum):
  APPLE = 1
  ORANGE = 2


class EmptyEnum(enum.Enum):
  pass


class EnumClassFlagTest(parameterized.TestCase):

  @parameterized.parameters(
      ('', '<APPLE|ORANGE>: (no help available)'),
      ('Type of fruit.', '<APPLE|ORANGE>: Type of fruit.'))
  def test_help_text(self, helptext_input, helptext_output):
    f = _flag.EnumClassFlag('fruit', None, helptext_input, Fruit)
    self.assertEqual(helptext_output, f.help)

  def test_requires_enum(self):
    with self.assertRaises(TypeError):
      _flag.EnumClassFlag('fruit', None, 'help', ['apple', 'orange'])

  def test_requires_non_empty_enum_class(self):
    with self.assertRaises(ValueError):
      _flag.EnumClassFlag('empty', None, 'help', EmptyEnum)

  def test_accepts_literal_default(self):
    f = _flag.EnumClassFlag('fruit', Fruit.APPLE, 'A sample enum flag.', Fruit)
    self.assertEqual(Fruit.APPLE, f.value)

  def test_accepts_string_default(self):
    f = _flag.EnumClassFlag('fruit', 'ORANGE', 'A sample enum flag.', Fruit)
    self.assertEqual(Fruit.ORANGE, f.value)

  def test_default_value_does_not_exist(self):
    with self.assertRaises(_exceptions.IllegalFlagValueError):
      _flag.EnumClassFlag('fruit', 'BANANA', 'help', Fruit)


class MultiEnumClassFlagTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('NoHelpSupplied', '', '<APPLE|ORANGE>: (no help available);\n    '
       'repeat this option to specify a list of values'),
      ('WithHelpSupplied', 'Type of fruit.',
       '<APPLE|ORANGE>: Type of fruit.;\n    '
       'repeat this option to specify a list of values'))
  def test_help_text(self, helptext_input, helptext_output):
    f = _flag.MultiEnumClassFlag('fruit', None, helptext_input, Fruit)
    self.assertEqual(helptext_output, f.help)

  def test_requires_enum(self):
    with self.assertRaises(TypeError):
      _flag.MultiEnumClassFlag('fruit', None, 'help', ['apple', 'orange'])

  def test_requires_non_empty_enum_class(self):
    with self.assertRaises(ValueError):
      _flag.MultiEnumClassFlag('empty', None, 'help', EmptyEnum)

  def test_accepts_literal_default(self):
    f = _flag.MultiEnumClassFlag('fruit', Fruit.APPLE, 'A sample enum flag.',
                                 Fruit)
    self.assertListEqual([Fruit.APPLE], f.value)

  def test_accepts_list_of_literal_default(self):
    f = _flag.MultiEnumClassFlag('fruit', [Fruit.APPLE, Fruit.ORANGE],
                                 'A sample enum flag.', Fruit)
    self.assertListEqual([Fruit.APPLE, Fruit.ORANGE], f.value)

  def test_accepts_string_default(self):
    f = _flag.MultiEnumClassFlag('fruit', 'ORANGE', 'A sample enum flag.',
                                 Fruit)
    self.assertListEqual([Fruit.ORANGE], f.value)

  def test_accepts_list_of_string_default(self):
    f = _flag.MultiEnumClassFlag('fruit', ['ORANGE', 'APPLE'],
                                 'A sample enum flag.', Fruit)
    self.assertListEqual([Fruit.ORANGE, Fruit.APPLE], f.value)

  def test_default_value_does_not_exist(self):
    with self.assertRaises(_exceptions.IllegalFlagValueError):
      _flag.MultiEnumClassFlag('fruit', 'BANANA', 'help', Fruit)


if __name__ == '__main__':
  absltest.main()
