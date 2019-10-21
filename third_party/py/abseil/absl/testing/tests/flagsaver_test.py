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

"""Tests for flagsaver."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl.testing import absltest
from absl.testing import flagsaver

flags.DEFINE_string('flagsaver_test_flag0', 'unchanged0', 'flag to test with')
flags.DEFINE_string('flagsaver_test_flag1', 'unchanged1', 'flag to test with')
flags.DEFINE_string('flagsaver_test_validated_flag', None, 'flag to test with')
flags.register_validator('flagsaver_test_validated_flag', lambda x: not x)

FLAGS = flags.FLAGS


@flags.validator('flagsaver_test_flag0')
def check_no_upper_case(value):
  return value == value.lower()


class _TestError(Exception):
  """Exception class for use in these tests."""


class FlagSaverTest(absltest.TestCase):

  def setUp(self):
    # Save the value of the instance of FLAGS local to this module.
    global FLAGS  # pylint: disable=global-statement
    self.flags = FLAGS
    # pylint: disable=g-bad-name
    FLAGS = flags.FlagValues()
    FLAGS.append_flag_values(self.flags)
    FLAGS.mark_as_parsed()

  def tearDown(self):
    global FLAGS  # pylint: disable=global-statement
    FLAGS = self.flags  # pylint: disable=g-bad-name

  def test_context_manager_without_parameters(self):
    with flagsaver.flagsaver():
      FLAGS.flagsaver_test_flag0 = 'new value'
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)

  def test_context_manager_with_overrides(self):
    with flagsaver.flagsaver(flagsaver_test_flag0='new value'):
      self.assertEqual('new value', FLAGS.flagsaver_test_flag0)
      FLAGS.flagsaver_test_flag1 = 'another value'
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    self.assertEqual('unchanged1', FLAGS.flagsaver_test_flag1)

  def test_context_manager_with_exception(self):
    with self.assertRaises(_TestError):
      with flagsaver.flagsaver(flagsaver_test_flag0='new value'):
        self.assertEqual('new value', FLAGS.flagsaver_test_flag0)
        FLAGS.flagsaver_test_flag1 = 'another value'
        raise _TestError('oops')
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    self.assertEqual('unchanged1', FLAGS.flagsaver_test_flag1)

  def test_context_manager_with_validation_exception(self):
    with self.assertRaises(flags.IllegalFlagValueError):
      with flagsaver.flagsaver(
          flagsaver_test_flag0='new value',
          flagsaver_test_validated_flag='new value'):
        pass
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    self.assertEqual('unchanged1', FLAGS.flagsaver_test_flag1)
    self.assertEqual(None, FLAGS.flagsaver_test_validated_flag)

  def test_decorator_without_call(self):

    @flagsaver.flagsaver
    def mutate_flags(value):
      """Test function that mutates a flag."""
      # The undecorated method mutates --flagsaver_test_flag0 to the given value
      # and then returns the value of that flag.  If the @flagsaver.flagsaver
      # decorator works as designed, then this mutation will be reverted after
      # this method returns.
      FLAGS.flagsaver_test_flag0 = value
      return FLAGS.flagsaver_test_flag0

    # mutate_flags returns the flag value before it gets restored by
    # the flagsaver decorator.  So we check that flag value was
    # actually changed in the method's scope.
    self.assertEqual('new value',
                     mutate_flags('new value'))
    # But... notice that the flag is now unchanged0.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)

  def test_decorator_without_parameters(self):

    @flagsaver.flagsaver()
    def mutate_flags(value):
      FLAGS.flagsaver_test_flag0 = value
      return FLAGS.flagsaver_test_flag0

    self.assertEqual('new value', mutate_flags('new value'))
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)

  def test_decorator_with_overrides(self):

    @flagsaver.flagsaver(flagsaver_test_flag0='new value')
    def mutate_flags():
      """Test function expecting new value."""
      # If the @flagsaver.decorator decorator works as designed,
      # then the value of the flag should be changed in the scope of
      # the method but the change will be reverted after this method
      # returns.
      return FLAGS.flagsaver_test_flag0

    # mutate_flags returns the flag value before it gets restored by
    # the flagsaver decorator.  So we check that flag value was
    # actually changed in the method's scope.
    self.assertEqual('new value', mutate_flags())
    # But... notice that the flag is now unchanged0.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)

  def test_save_flag_value(self):
    # First save the flag values.
    saved_flag_values = flagsaver.save_flag_values()

    # Now mutate the flag's value field and check that it changed.
    FLAGS.flagsaver_test_flag0 = 'new value'
    self.assertEqual('new value', FLAGS.flagsaver_test_flag0)

    # Now restore the flag to its original value.
    flagsaver.restore_flag_values(saved_flag_values)
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)

  def test_save_flag_default(self):
    # First save the flag.
    saved_flag_values = flagsaver.save_flag_values()

    # Now mutate the flag's default field and check that it changed.
    FLAGS.set_default('flagsaver_test_flag0', 'new_default')
    self.assertEqual('new_default', FLAGS['flagsaver_test_flag0'].default)

    # Now restore the flag's default field.
    flagsaver.restore_flag_values(saved_flag_values)
    self.assertEqual('unchanged0', FLAGS['flagsaver_test_flag0'].default)

  def test_restore_after_parse(self):
    # First save the flag.
    saved_flag_values = flagsaver.save_flag_values()

    # Sanity check (would fail if called with --flagsaver_test_flag0).
    self.assertEqual(0, FLAGS['flagsaver_test_flag0'].present)
    # Now populate the flag and check that it changed.
    FLAGS['flagsaver_test_flag0'].parse('new value')
    self.assertEqual('new value', FLAGS['flagsaver_test_flag0'].value)
    self.assertEqual(1, FLAGS['flagsaver_test_flag0'].present)

    # Now restore the flag to its original value.
    flagsaver.restore_flag_values(saved_flag_values)
    self.assertEqual('unchanged0', FLAGS['flagsaver_test_flag0'].value)
    self.assertEqual(0, FLAGS['flagsaver_test_flag0'].present)

  def test_decorator_with_exception(self):

    @flagsaver.flagsaver
    def raise_exception():
      FLAGS.flagsaver_test_flag0 = 'new value'
      # Simulate a failed test.
      raise _TestError('something happened')

    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    self.assertRaises(_TestError, raise_exception)
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)

  def test_validator_list_is_restored(self):

    self.assertLen(FLAGS['flagsaver_test_flag0'].validators, 1)
    original_validators = list(FLAGS['flagsaver_test_flag0'].validators)

    @flagsaver.flagsaver
    def modify_validators():

      def no_space(value):
        return ' ' not in value

      flags.register_validator('flagsaver_test_flag0', no_space)
      self.assertLen(FLAGS['flagsaver_test_flag0'].validators, 2)

    modify_validators()
    self.assertEqual(
        original_validators, FLAGS['flagsaver_test_flag0'].validators)


class FlagSaverDecoratorUsageTest(absltest.TestCase):

  @flagsaver.flagsaver
  def test_mutate1(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'

  @flagsaver.flagsaver
  def test_mutate2(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'

  @flagsaver.flagsaver
  def test_mutate3(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'

  @flagsaver.flagsaver
  def test_mutate4(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'


class FlagSaverSetUpTearDownUsageTest(absltest.TestCase):

  def setUp(self):
    self.saved_flag_values = flagsaver.save_flag_values()

  def tearDown(self):
    flagsaver.restore_flag_values(self.saved_flag_values)

  def test_mutate1(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'

  def test_mutate2(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'

  def test_mutate3(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'

  def test_mutate4(self):
    # Even though other test cases change the flag, it should be
    # restored to 'unchanged0' if the flagsaver is working.
    self.assertEqual('unchanged0', FLAGS.flagsaver_test_flag0)
    FLAGS.flagsaver_test_flag0 = 'changed0'


class FlagSaverBadUsageTest(absltest.TestCase):
  """Tests that certain kinds of improper usages raise errors."""

  def test_flag_saver_on_class(self):
    with self.assertRaises(TypeError):

      # WRONG. Don't do this.
      # Consider the correct usage example in FlagSaverSetUpTearDownUsageTest.
      @flagsaver.flagsaver
      class FooTest(absltest.TestCase):

        def test_tautology(self):
          pass

      del FooTest

  def test_flag_saver_call_on_class(self):
    with self.assertRaises(TypeError):

      # WRONG. Don't do this.
      # Consider the correct usage example in FlagSaverSetUpTearDownUsageTest.
      @flagsaver.flagsaver()
      class FooTest(absltest.TestCase):

        def test_tautology(self):
          pass

      del FooTest

  def test_flag_saver_with_overrides_on_class(self):
    with self.assertRaises(TypeError):

      # WRONG. Don't do this.
      # Consider the correct usage example in FlagSaverSetUpTearDownUsageTest.
      @flagsaver.flagsaver(foo='bar')
      class FooTest(absltest.TestCase):

        def test_tautology(self):
          pass

      del FooTest

  def test_multiple_positional_parameters(self):
    with self.assertRaises(ValueError):
      func_a = lambda: None
      func_b = lambda: None
      flagsaver.flagsaver(func_a, func_b)

  def test_both_positional_and_keyword_parameters(self):
    with self.assertRaises(ValueError):
      func_a = lambda: None
      flagsaver.flagsaver(func_a, flagsaver_test_flag0='new value')


if __name__ == '__main__':
  absltest.main()
