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

"""Testing that flags validators framework does work.

This file tests that each flag validator called when it should be, and that
failed validator will throw an exception, etc.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings


from absl.flags import _defines
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _validators
from absl.testing import absltest


class SingleFlagValidatorTest(absltest.TestCase):
  """Testing _validators.register_validator() method."""

  def setUp(self):
    super(SingleFlagValidatorTest, self).setUp()
    self.flag_values = _flagvalues.FlagValues()
    self.call_args = []

  def test_success(self):
    def checker(x):
      self.call_args.append(x)
      return True
    _defines.DEFINE_integer(
        'test_flag', None, 'Usual integer flag', flag_values=self.flag_values)
    _validators.register_validator(
        'test_flag',
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program',)
    self.flag_values(argv)
    self.assertEqual(None, self.flag_values.test_flag)
    self.flag_values.test_flag = 2
    self.assertEqual(2, self.flag_values.test_flag)
    self.assertEqual([None, 2], self.call_args)

  def test_default_value_not_used_success(self):
    def checker(x):
      self.call_args.append(x)
      return True
    _defines.DEFINE_integer(
        'test_flag', None, 'Usual integer flag', flag_values=self.flag_values)
    _validators.register_validator(
        'test_flag',
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program', '--test_flag=1')
    self.flag_values(argv)
    self.assertEqual(1, self.flag_values.test_flag)
    self.assertEqual([1], self.call_args)

  def test_validator_not_called_when_other_flag_is_changed(self):
    def checker(x):
      self.call_args.append(x)
      return True
    _defines.DEFINE_integer(
        'test_flag', 1, 'Usual integer flag', flag_values=self.flag_values)
    _defines.DEFINE_integer(
        'other_flag', 2, 'Other integer flag', flag_values=self.flag_values)
    _validators.register_validator(
        'test_flag',
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program',)
    self.flag_values(argv)
    self.assertEqual(1, self.flag_values.test_flag)
    self.flag_values.other_flag = 3
    self.assertEqual([1], self.call_args)

  def test_exception_raised_if_checker_fails(self):
    def checker(x):
      self.call_args.append(x)
      return x == 1
    _defines.DEFINE_integer(
        'test_flag', None, 'Usual integer flag', flag_values=self.flag_values)
    _validators.register_validator(
        'test_flag',
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program', '--test_flag=1')
    self.flag_values(argv)
    try:
      self.flag_values.test_flag = 2
      raise AssertionError('IllegalFlagValueError expected')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual('flag --test_flag=2: Errors happen', str(e))
    self.assertEqual([1, 2], self.call_args)

  def test_exception_raised_if_checker_raises_exception(self):
    def checker(x):
      self.call_args.append(x)
      if x == 1:
        return True
      raise _exceptions.ValidationError('Specific message')

    _defines.DEFINE_integer(
        'test_flag', None, 'Usual integer flag', flag_values=self.flag_values)
    _validators.register_validator(
        'test_flag',
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program', '--test_flag=1')
    self.flag_values(argv)
    try:
      self.flag_values.test_flag = 2
      raise AssertionError('IllegalFlagValueError expected')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual('flag --test_flag=2: Specific message', str(e))
    self.assertEqual([1, 2], self.call_args)

  def test_error_message_when_checker_returns_false_on_start(self):
    def checker(x):
      self.call_args.append(x)
      return False
    _defines.DEFINE_integer(
        'test_flag', None, 'Usual integer flag', flag_values=self.flag_values)
    _validators.register_validator(
        'test_flag',
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program', '--test_flag=1')
    try:
      self.flag_values(argv)
      raise AssertionError('IllegalFlagValueError expected')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual('flag --test_flag=1: Errors happen', str(e))
    self.assertEqual([1], self.call_args)

  def test_error_message_when_checker_raises_exception_on_start(self):
    def checker(x):
      self.call_args.append(x)
      raise _exceptions.ValidationError('Specific message')

    _defines.DEFINE_integer(
        'test_flag', None, 'Usual integer flag', flag_values=self.flag_values)
    _validators.register_validator(
        'test_flag',
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program', '--test_flag=1')
    try:
      self.flag_values(argv)
      raise AssertionError('IllegalFlagValueError expected')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual('flag --test_flag=1: Specific message', str(e))
    self.assertEqual([1], self.call_args)

  def test_validators_checked_in_order(self):

    def required(x):
      self.calls.append('required')
      return x is not None

    def even(x):
      self.calls.append('even')
      return x % 2 == 0

    self.calls = []
    self._define_flag_and_validators(required, even)
    self.assertEqual(['required', 'even'], self.calls)

    self.calls = []
    self._define_flag_and_validators(even, required)
    self.assertEqual(['even', 'required'], self.calls)

  def _define_flag_and_validators(self, first_validator, second_validator):
    local_flags = _flagvalues.FlagValues()
    _defines.DEFINE_integer(
        'test_flag', 2, 'test flag', flag_values=local_flags)
    _validators.register_validator(
        'test_flag', first_validator, message='', flag_values=local_flags)
    _validators.register_validator(
        'test_flag', second_validator, message='', flag_values=local_flags)
    argv = ('./program',)
    local_flags(argv)

  def test_validator_as_decorator(self):
    _defines.DEFINE_integer(
        'test_flag', None, 'Simple integer flag', flag_values=self.flag_values)

    @_validators.validator('test_flag', flag_values=self.flag_values)
    def checker(x):
      self.call_args.append(x)
      return True

    argv = ('./program',)
    self.flag_values(argv)
    self.assertEqual(None, self.flag_values.test_flag)
    self.flag_values.test_flag = 2
    self.assertEqual(2, self.flag_values.test_flag)
    self.assertEqual([None, 2], self.call_args)
    # Check that 'Checker' is still a function and has not been replaced.
    self.assertTrue(checker(3))
    self.assertEqual([None, 2, 3], self.call_args)


class MultiFlagsValidatorTest(absltest.TestCase):
  """Test flags multi-flag validators."""

  def setUp(self):
    super(MultiFlagsValidatorTest, self).setUp()
    self.flag_values = _flagvalues.FlagValues()
    self.call_args = []
    _defines.DEFINE_integer(
        'foo', 1, 'Usual integer flag', flag_values=self.flag_values)
    _defines.DEFINE_integer(
        'bar', 2, 'Usual integer flag', flag_values=self.flag_values)

  def test_success(self):
    def checker(flags_dict):
      self.call_args.append(flags_dict)
      return True
    _validators.register_multi_flags_validator(
        ['foo', 'bar'], checker, flag_values=self.flag_values)

    argv = ('./program', '--bar=2')
    self.flag_values(argv)
    self.assertEqual(1, self.flag_values.foo)
    self.assertEqual(2, self.flag_values.bar)
    self.assertEqual([{'foo': 1, 'bar': 2}], self.call_args)
    self.flag_values.foo = 3
    self.assertEqual(3, self.flag_values.foo)
    self.assertEqual([{'foo': 1, 'bar': 2}, {'foo': 3, 'bar': 2}],
                     self.call_args)

  def test_validator_not_called_when_other_flag_is_changed(self):
    def checker(flags_dict):
      self.call_args.append(flags_dict)
      return True
    _defines.DEFINE_integer(
        'other_flag', 3, 'Other integer flag', flag_values=self.flag_values)
    _validators.register_multi_flags_validator(
        ['foo', 'bar'], checker, flag_values=self.flag_values)

    argv = ('./program',)
    self.flag_values(argv)
    self.flag_values.other_flag = 3
    self.assertEqual([{'foo': 1, 'bar': 2}], self.call_args)

  def test_exception_raised_if_checker_fails(self):
    def checker(flags_dict):
      self.call_args.append(flags_dict)
      values = flags_dict.values()
      # Make sure all the flags have different values.
      return len(set(values)) == len(values)
    _validators.register_multi_flags_validator(
        ['foo', 'bar'],
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program',)
    self.flag_values(argv)
    try:
      self.flag_values.bar = 1
      raise AssertionError('IllegalFlagValueError expected')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual('flags foo=1, bar=1: Errors happen', str(e))
    self.assertEqual([{'foo': 1, 'bar': 2}, {'foo': 1, 'bar': 1}],
                     self.call_args)

  def test_exception_raised_if_checker_raises_exception(self):
    def checker(flags_dict):
      self.call_args.append(flags_dict)
      values = flags_dict.values()
      # Make sure all the flags have different values.
      if len(set(values)) != len(values):
        raise _exceptions.ValidationError('Specific message')
      return True

    _validators.register_multi_flags_validator(
        ['foo', 'bar'],
        checker,
        message='Errors happen',
        flag_values=self.flag_values)

    argv = ('./program',)
    self.flag_values(argv)
    try:
      self.flag_values.bar = 1
      raise AssertionError('IllegalFlagValueError expected')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual('flags foo=1, bar=1: Specific message', str(e))
    self.assertEqual([{'foo': 1, 'bar': 2}, {'foo': 1, 'bar': 1}],
                     self.call_args)

  def test_decorator(self):
    @_validators.multi_flags_validator(
        ['foo', 'bar'], message='Errors happen', flag_values=self.flag_values)
    def checker(flags_dict):  # pylint: disable=unused-variable
      self.call_args.append(flags_dict)
      values = flags_dict.values()
      # Make sure all the flags have different values.
      return len(set(values)) == len(values)

    argv = ('./program',)
    self.flag_values(argv)
    try:
      self.flag_values.bar = 1
      raise AssertionError('IllegalFlagValueError expected')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual('flags foo=1, bar=1: Errors happen', str(e))
    self.assertEqual([{'foo': 1, 'bar': 2}, {'foo': 1, 'bar': 1}],
                     self.call_args)


class MarkFlagsAsMutualExclusiveTest(absltest.TestCase):

  def setUp(self):
    super(MarkFlagsAsMutualExclusiveTest, self).setUp()
    self.flag_values = _flagvalues.FlagValues()

    _defines.DEFINE_string(
        'flag_one', None, 'flag one', flag_values=self.flag_values)
    _defines.DEFINE_string(
        'flag_two', None, 'flag two', flag_values=self.flag_values)
    _defines.DEFINE_string(
        'flag_three', None, 'flag three', flag_values=self.flag_values)
    _defines.DEFINE_integer(
        'int_flag_one', None, 'int flag one', flag_values=self.flag_values)
    _defines.DEFINE_integer(
        'int_flag_two', None, 'int flag two', flag_values=self.flag_values)
    _defines.DEFINE_multi_string(
        'multi_flag_one', None, 'multi flag one', flag_values=self.flag_values)
    _defines.DEFINE_multi_string(
        'multi_flag_two', None, 'multi flag two', flag_values=self.flag_values)
    _defines.DEFINE_boolean(
        'flag_not_none', False, 'false default', flag_values=self.flag_values)

  def _mark_flags_as_mutually_exclusive(self, flag_names, required):
    _validators.mark_flags_as_mutual_exclusive(
        flag_names, required=required, flag_values=self.flag_values)

  def test_no_flags_present(self):
    self._mark_flags_as_mutually_exclusive(['flag_one', 'flag_two'], False)
    argv = ('./program',)

    self.flag_values(argv)
    self.assertEqual(None, self.flag_values.flag_one)
    self.assertEqual(None, self.flag_values.flag_two)

  def test_no_flags_present_required(self):
    self._mark_flags_as_mutually_exclusive(['flag_one', 'flag_two'], True)
    argv = ('./program',)
    expected = (
        'flags flag_one=None, flag_two=None: '
        'Exactly one of (flag_one, flag_two) must have a value other than '
        'None.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_one_flag_present(self):
    self._mark_flags_as_mutually_exclusive(['flag_one', 'flag_two'], False)
    self.flag_values(('./program', '--flag_one=1'))
    self.assertEqual('1', self.flag_values.flag_one)

  def test_one_flag_present_required(self):
    self._mark_flags_as_mutually_exclusive(['flag_one', 'flag_two'], True)
    self.flag_values(('./program', '--flag_two=2'))
    self.assertEqual('2', self.flag_values.flag_two)

  def test_one_flag_zero_required(self):
    self._mark_flags_as_mutually_exclusive(
        ['int_flag_one', 'int_flag_two'], True)
    self.flag_values(('./program', '--int_flag_one=0'))
    self.assertEqual(0, self.flag_values.int_flag_one)

  def test_mutual_exclusion_with_extra_flags(self):
    self._mark_flags_as_mutually_exclusive(['flag_one', 'flag_two'], True)
    argv = ('./program', '--flag_two=2', '--flag_three=3')

    self.flag_values(argv)
    self.assertEqual('2', self.flag_values.flag_two)
    self.assertEqual('3', self.flag_values.flag_three)

  def test_mutual_exclusion_with_zero(self):
    self._mark_flags_as_mutually_exclusive(
        ['int_flag_one', 'int_flag_two'], False)
    argv = ('./program', '--int_flag_one=0', '--int_flag_two=0')
    expected = (
        'flags int_flag_one=0, int_flag_two=0: '
        'At most one of (int_flag_one, int_flag_two) must have a value other '
        'than None.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_multiple_flags_present(self):
    self._mark_flags_as_mutually_exclusive(
        ['flag_one', 'flag_two', 'flag_three'], False)
    argv = ('./program', '--flag_one=1', '--flag_two=2', '--flag_three=3')
    expected = (
        'flags flag_one=1, flag_two=2, flag_three=3: '
        'At most one of (flag_one, flag_two, flag_three) must have a value '
        'other than None.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_multiple_flags_present_required(self):
    self._mark_flags_as_mutually_exclusive(
        ['flag_one', 'flag_two', 'flag_three'], True)
    argv = ('./program', '--flag_one=1', '--flag_two=2', '--flag_three=3')
    expected = (
        'flags flag_one=1, flag_two=2, flag_three=3: '
        'Exactly one of (flag_one, flag_two, flag_three) must have a value '
        'other than None.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_no_multiflags_present(self):
    self._mark_flags_as_mutually_exclusive(
        ['multi_flag_one', 'multi_flag_two'], False)
    argv = ('./program',)
    self.flag_values(argv)
    self.assertEqual(None, self.flag_values.multi_flag_one)
    self.assertEqual(None, self.flag_values.multi_flag_two)

  def test_no_multistring_flags_present_required(self):
    self._mark_flags_as_mutually_exclusive(
        ['multi_flag_one', 'multi_flag_two'], True)
    argv = ('./program',)
    expected = (
        'flags multi_flag_one=None, multi_flag_two=None: '
        'Exactly one of (multi_flag_one, multi_flag_two) must have a value '
        'other than None.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_one_multiflag_present(self):
    self._mark_flags_as_mutually_exclusive(
        ['multi_flag_one', 'multi_flag_two'], True)
    self.flag_values(('./program', '--multi_flag_one=1'))
    self.assertEqual(['1'], self.flag_values.multi_flag_one)

  def test_one_multiflag_present_repeated(self):
    self._mark_flags_as_mutually_exclusive(
        ['multi_flag_one', 'multi_flag_two'], True)
    self.flag_values(('./program', '--multi_flag_one=1', '--multi_flag_one=1b'))
    self.assertEqual(['1', '1b'], self.flag_values.multi_flag_one)

  def test_multiple_multiflags_present(self):
    self._mark_flags_as_mutually_exclusive(
        ['multi_flag_one', 'multi_flag_two'], False)
    argv = ('./program', '--multi_flag_one=1', '--multi_flag_two=2')
    expected = (
        "flags multi_flag_one=['1'], multi_flag_two=['2']: "
        'At most one of (multi_flag_one, multi_flag_two) must have a value '
        'other than None.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_multiple_multiflags_present_required(self):
    self._mark_flags_as_mutually_exclusive(
        ['multi_flag_one', 'multi_flag_two'], True)
    argv = ('./program', '--multi_flag_one=1', '--multi_flag_two=2')
    expected = (
        "flags multi_flag_one=['1'], multi_flag_two=['2']: "
        'Exactly one of (multi_flag_one, multi_flag_two) must have a value '
        'other than None.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_flag_default_not_none_warning(self):
    with warnings.catch_warnings(record=True) as caught_warnings:
      warnings.simplefilter('always')
      self._mark_flags_as_mutually_exclusive(['flag_one', 'flag_not_none'],
                                             False)
    self.assertLen(caught_warnings, 1)
    self.assertIn('--flag_not_none has a non-None default value',
                  str(caught_warnings[0].message))


class MarkBoolFlagsAsMutualExclusiveTest(absltest.TestCase):

  def setUp(self):
    super(MarkBoolFlagsAsMutualExclusiveTest, self).setUp()
    self.flag_values = _flagvalues.FlagValues()

    _defines.DEFINE_boolean(
        'false_1', False, 'default false 1', flag_values=self.flag_values)
    _defines.DEFINE_boolean(
        'false_2', False, 'default false 2', flag_values=self.flag_values)
    _defines.DEFINE_boolean(
        'true_1', True, 'default true 1', flag_values=self.flag_values)
    _defines.DEFINE_integer(
        'non_bool', None, 'non bool', flag_values=self.flag_values)

  def _mark_bool_flags_as_mutually_exclusive(self, flag_names, required):
    _validators.mark_bool_flags_as_mutual_exclusive(
        flag_names, required=required, flag_values=self.flag_values)

  def test_no_flags_present(self):
    self._mark_bool_flags_as_mutually_exclusive(['false_1', 'false_2'], False)
    self.flag_values(('./program',))
    self.assertEqual(False, self.flag_values.false_1)
    self.assertEqual(False, self.flag_values.false_2)

  def test_no_flags_present_required(self):
    self._mark_bool_flags_as_mutually_exclusive(['false_1', 'false_2'], True)
    argv = ('./program',)
    expected = (
        'flags false_1=False, false_2=False: '
        'Exactly one of (false_1, false_2) must be True.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_no_flags_present_with_default_true_required(self):
    self._mark_bool_flags_as_mutually_exclusive(['false_1', 'true_1'], True)
    self.flag_values(('./program',))
    self.assertEqual(False, self.flag_values.false_1)
    self.assertEqual(True, self.flag_values.true_1)

  def test_two_flags_true(self):
    self._mark_bool_flags_as_mutually_exclusive(['false_1', 'false_2'], False)
    argv = ('./program', '--false_1', '--false_2')
    expected = (
        'flags false_1=True, false_2=True: At most one of (false_1, false_2) '
        'must be True.')

    self.assertRaisesWithLiteralMatch(_exceptions.IllegalFlagValueError,
                                      expected, self.flag_values, argv)

  def test_non_bool_flag(self):
    expected = ('Flag --non_bool is not Boolean, which is required for flags '
                'used in mark_bool_flags_as_mutual_exclusive.')
    with self.assertRaisesWithLiteralMatch(_exceptions.ValidationError,
                                           expected):
      self._mark_bool_flags_as_mutually_exclusive(['false_1', 'non_bool'],
                                                  False)


class MarkFlagAsRequiredTest(absltest.TestCase):

  def setUp(self):
    super(MarkFlagAsRequiredTest, self).setUp()
    self.flag_values = _flagvalues.FlagValues()

  def test_success(self):
    _defines.DEFINE_string(
        'string_flag', None, 'string flag', flag_values=self.flag_values)
    _validators.mark_flag_as_required(
        'string_flag', flag_values=self.flag_values)
    argv = ('./program', '--string_flag=value')
    self.flag_values(argv)
    self.assertEqual('value', self.flag_values.string_flag)

  def test_catch_none_as_default(self):
    _defines.DEFINE_string(
        'string_flag', None, 'string flag', flag_values=self.flag_values)
    _validators.mark_flag_as_required(
        'string_flag', flag_values=self.flag_values)
    argv = ('./program',)
    expected = (
        r'flag --string_flag=None: Flag --string_flag must have a value other '
        r'than None\.')
    with self.assertRaisesRegex(_exceptions.IllegalFlagValueError, expected):
      self.flag_values(argv)

  def test_catch_setting_none_after_program_start(self):
    _defines.DEFINE_string(
        'string_flag', 'value', 'string flag', flag_values=self.flag_values)
    _validators.mark_flag_as_required(
        'string_flag', flag_values=self.flag_values)
    argv = ('./program',)
    self.flag_values(argv)
    self.assertEqual('value', self.flag_values.string_flag)
    expected = ('flag --string_flag=None: Flag --string_flag must have a value '
                'other than None.')
    try:
      self.flag_values.string_flag = None
      raise AssertionError('Failed to detect non-set required flag.')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual(expected, str(e))

  def test_flag_default_not_none_warning(self):
    _defines.DEFINE_string(
        'flag_not_none', '', 'empty default', flag_values=self.flag_values)
    with warnings.catch_warnings(record=True) as caught_warnings:
      warnings.simplefilter('always')
      _validators.mark_flag_as_required(
          'flag_not_none', flag_values=self.flag_values)

    self.assertLen(caught_warnings, 1)
    self.assertIn('--flag_not_none has a non-None default value',
                  str(caught_warnings[0].message))


class MarkFlagsAsRequiredTest(absltest.TestCase):

  def setUp(self):
    super(MarkFlagsAsRequiredTest, self).setUp()
    self.flag_values = _flagvalues.FlagValues()

  def test_success(self):
    _defines.DEFINE_string(
        'string_flag_1', None, 'string flag 1', flag_values=self.flag_values)
    _defines.DEFINE_string(
        'string_flag_2', None, 'string flag 2', flag_values=self.flag_values)
    flag_names = ['string_flag_1', 'string_flag_2']
    _validators.mark_flags_as_required(flag_names, flag_values=self.flag_values)
    argv = ('./program', '--string_flag_1=value_1', '--string_flag_2=value_2')
    self.flag_values(argv)
    self.assertEqual('value_1', self.flag_values.string_flag_1)
    self.assertEqual('value_2', self.flag_values.string_flag_2)

  def test_catch_none_as_default(self):
    _defines.DEFINE_string(
        'string_flag_1', None, 'string flag 1', flag_values=self.flag_values)
    _defines.DEFINE_string(
        'string_flag_2', None, 'string flag 2', flag_values=self.flag_values)
    _validators.mark_flags_as_required(
        ['string_flag_1', 'string_flag_2'], flag_values=self.flag_values)
    argv = ('./program', '--string_flag_1=value_1')
    expected = (
        r'flag --string_flag_2=None: Flag --string_flag_2 must have a value '
        r'other than None\.')
    with self.assertRaisesRegex(_exceptions.IllegalFlagValueError, expected):
      self.flag_values(argv)

  def test_catch_setting_none_after_program_start(self):
    _defines.DEFINE_string(
        'string_flag_1',
        'value_1',
        'string flag 1',
        flag_values=self.flag_values)
    _defines.DEFINE_string(
        'string_flag_2',
        'value_2',
        'string flag 2',
        flag_values=self.flag_values)
    _validators.mark_flags_as_required(
        ['string_flag_1', 'string_flag_2'], flag_values=self.flag_values)
    argv = ('./program', '--string_flag_1=value_1')
    self.flag_values(argv)
    self.assertEqual('value_1', self.flag_values.string_flag_1)
    expected = (
        'flag --string_flag_1=None: Flag --string_flag_1 must have a value '
        'other than None.')
    try:
      self.flag_values.string_flag_1 = None
      raise AssertionError('Failed to detect non-set required flag.')
    except _exceptions.IllegalFlagValueError as e:
      self.assertEqual(expected, str(e))

if __name__ == '__main__':
  absltest.main()
