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

"""Tests for lower/upper bounds validators for numeric flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.flags import _validators
from absl.testing import absltest
import mock


class NumericFlagBoundsTest(absltest.TestCase):

  def setUp(self):
    super(NumericFlagBoundsTest, self).setUp()
    self.flag_values = flags.FlagValues()

  def test_no_validator_if_no_bounds(self):
    """Validator is not registered if lower and upper bound are None."""
    with mock.patch.object(_validators, 'register_validator'
                          ) as register_validator:
      flags.DEFINE_integer('positive_flag', None, 'positive int',
                           lower_bound=0, flag_values=self.flag_values)
      register_validator.assert_called_once_with(
          'positive_flag', mock.ANY, flag_values=self.flag_values)
    with mock.patch.object(_validators, 'register_validator'
                          ) as register_validator:
      flags.DEFINE_integer('int_flag', None, 'just int',
                           flag_values=self.flag_values)
      register_validator.assert_not_called()

  def test_success(self):
    flags.DEFINE_integer('int_flag', 5, 'Just integer',
                         flag_values=self.flag_values)
    argv = ('./program', '--int_flag=13')
    self.flag_values(argv)
    self.assertEqual(13, self.flag_values.int_flag)
    self.flag_values.int_flag = 25
    self.assertEqual(25, self.flag_values.int_flag)

  def test_success_if_none(self):
    flags.DEFINE_integer('int_flag', None, '',
                         lower_bound=0, upper_bound=5,
                         flag_values=self.flag_values)
    argv = ('./program',)
    self.flag_values(argv)
    self.assertIsNone(self.flag_values.int_flag)

  def test_success_if_exactly_equals(self):
    flags.DEFINE_float('float_flag', None, '',
                       lower_bound=1, upper_bound=1,
                       flag_values=self.flag_values)
    argv = ('./program', '--float_flag=1')
    self.flag_values(argv)
    self.assertEqual(1, self.flag_values.float_flag)

  def test_exception_if_smaller(self):
    flags.DEFINE_integer('int_flag', None, '',
                         lower_bound=0, upper_bound=5,
                         flag_values=self.flag_values)
    argv = ('./program', '--int_flag=-1')
    try:
      self.flag_values(argv)
    except flags.IllegalFlagValueError as e:
      text = 'flag --int_flag=-1: -1 is not an integer in the range [0, 5]'
      self.assertEqual(text, str(e))


class SettingFlagAfterStartTest(absltest.TestCase):

  def setUp(self):
    self.flag_values = flags.FlagValues()

  def test_success(self):
    flags.DEFINE_integer('int_flag', None, 'Just integer',
                         flag_values=self.flag_values)
    argv = ('./program', '--int_flag=13')
    self.flag_values(argv)
    self.assertEqual(13, self.flag_values.int_flag)
    self.flag_values.int_flag = 25
    self.assertEqual(25, self.flag_values.int_flag)

  def test_exception_if_setting_integer_flag_outside_bounds(self):
    flags.DEFINE_integer('int_flag', None, 'Just integer', lower_bound=0,
                         flag_values=self.flag_values)
    argv = ('./program', '--int_flag=13')
    self.flag_values(argv)
    self.assertEqual(13, self.flag_values.int_flag)
    with self.assertRaises(flags.IllegalFlagValueError):
      self.flag_values.int_flag = -2


if __name__ == '__main__':
  absltest.main()
