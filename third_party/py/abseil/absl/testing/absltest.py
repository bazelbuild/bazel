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

"""Base functionality for Abseil Python tests.

This module contains base classes and high-level functions for Abseil-style
tests.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import difflib
import errno
import getpass
import inspect
import itertools
import json
import os
import random
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest

try:
  import faulthandler
except ImportError:
  # We use faulthandler if it is available.
  faulthandler = None

from absl import app
from absl import flags
from absl import logging
from absl.testing import xml_reporter
import six
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin


FLAGS = flags.FLAGS

_TEXT_OR_BINARY_TYPES = (six.text_type, six.binary_type)


# Many of the methods in this module have names like assertSameElements.
# This kind of name does not comply with PEP8 style,
# but it is consistent with the naming of methods in unittest.py.
# pylint: disable=invalid-name


def _get_default_test_random_seed():
  random_seed = 301
  value = os.environ.get('TEST_RANDOM_SEED', '')
  try:
    random_seed = int(value)
  except ValueError:
    pass
  return random_seed


def get_default_test_srcdir():
  """Returns default test source dir."""
  return os.environ.get('TEST_SRCDIR', '')


def get_default_test_tmpdir():
  """Returns default test temp dir."""
  tmpdir = os.environ.get('TEST_TMPDIR', '')
  if not tmpdir:
    tmpdir = os.path.join(tempfile.gettempdir(), 'absl_testing')

  return tmpdir


def _get_default_randomize_ordering_seed():
  """Returns default seed to use for randomizing test order.

  This function first checks the --test_randomize_ordering_seed flag, and then
  the TEST_RANDOMIZE_ORDERING_SEED environment variable. If the first value
  we find is:
    * (not set): disable test randomization
    * 0: disable test randomization
    * 'random': choose a random seed in [1, 4294967295] for test order
      randomization
    * positive integer: use this seed for test order randomization

  (The values used are patterned after
  https://docs.python.org/3/using/cmdline.html#envvar-PYTHONHASHSEED).

  In principle, it would be simpler to return None if no override is provided;
  however, the python random module has no `get_seed()`, only `getstate()`,
  which returns far more data than we want to pass via an environment variable
  or flag.

  Returns:
    A default value for test case randomization (int). 0 means do not randomize.

  Raises:
    ValueError: Raised when the flag or env value is not one of the options
        above.
  """
  if FLAGS.test_randomize_ordering_seed is not None:
    randomize = FLAGS.test_randomize_ordering_seed
  else:
    randomize = os.environ.get('TEST_RANDOMIZE_ORDERING_SEED')
  if randomize is None:
    return 0
  if randomize == 'random':
    return random.Random().randint(1, 4294967295)
  if randomize == '0':
    return 0
  try:
    seed = int(randomize)
    if seed > 0:
      return seed
  except ValueError:
    pass
  raise ValueError(
      'Unknown test randomization seed value: {}'.format(randomize))


flags.DEFINE_integer('test_random_seed', _get_default_test_random_seed(),
                     'Random seed for testing. Some test frameworks may '
                     'change the default value of this flag between runs, so '
                     'it is not appropriate for seeding probabilistic tests.',
                     allow_override_cpp=True)
flags.DEFINE_string('test_srcdir',
                    get_default_test_srcdir(),
                    'Root of directory tree where source files live',
                    allow_override_cpp=True)
flags.DEFINE_string('test_tmpdir', get_default_test_tmpdir(),
                    'Directory for temporary testing files',
                    allow_override_cpp=True)
flags.DEFINE_string('test_randomize_ordering_seed', None,
                    'If positive, use this as a seed to randomize the '
                    'execution order for test cases. If "random", pick a '
                    'random seed to use. If 0 or not set, do not randomize '
                    'test case execution order. This flag also overrides '
                    'the TEST_RANDOMIZE_ORDERING_SEED environment variable.')
flags.DEFINE_string('xml_output_file', '',
                    'File to store XML test results')


# We might need to monkey-patch TestResult so that it stops considering an
# unexpected pass as a as a "successful result".  For details, see
# http://bugs.python.org/issue20165
def _monkey_patch_test_result_for_unexpected_passes():
  """Workaround for <http://bugs.python.org/issue20165>."""

  def wasSuccessful(self):
    """Tells whether or not this result was a success.

    Any unexpected pass is to be counted as a non-success.

    Args:
      self: The TestResult instance.

    Returns:
      Whether or not this result was a success.
    """
    return (len(self.failures) == len(self.errors) ==
            len(self.unexpectedSuccesses) == 0)

  test_result = unittest.result.TestResult()
  test_result.addUnexpectedSuccess('test')
  if test_result.wasSuccessful():  # The bug is present.
    unittest.result.TestResult.wasSuccessful = wasSuccessful
    if test_result.wasSuccessful():  # Warn the user if our hot-fix failed.
      sys.stderr.write('unittest.result.TestResult monkey patch to report'
                       ' unexpected passes as failures did not work.\n')


_monkey_patch_test_result_for_unexpected_passes()


class TestCase(unittest.TestCase):
  """Extension of unittest.TestCase providing more powerful assertions."""

  maxDiff = 80 * 20

  def shortDescription(self):
    """Formats both the test method name and the first line of its docstring.

    If no docstring is given, only returns the method name.

    This method overrides unittest.TestCase.shortDescription(), which
    only returns the first line of the docstring, obscuring the name
    of the test upon failure.

    Returns:
      desc: A short description of a test method.
    """
    desc = str(self)
    # NOTE: super() is used here instead of directly invoking
    # unittest.TestCase.shortDescription(self), because of the
    # following line that occurs later on:
    #       unittest.TestCase = TestCase
    # Because of this, direct invocation of what we think is the
    # superclass will actually cause infinite recursion.
    doc_first_line = super(TestCase, self).shortDescription()
    if doc_first_line is not None:
      desc = '\n'.join((desc, doc_first_line))
    return desc

  def assertStartsWith(self, actual, expected_start, msg=None):
    """Asserts that actual.startswith(expected_start) is True.

    Args:
      actual: str
      expected_start: str
      msg: Optional message to report on failure.
    """
    if not actual.startswith(expected_start):
      self.fail('%r does not start with %r' % (actual, expected_start), msg)

  def assertNotStartsWith(self, actual, unexpected_start, msg=None):
    """Asserts that actual.startswith(unexpected_start) is False.

    Args:
      actual: str
      unexpected_start: str
      msg: Optional message to report on failure.
    """
    if actual.startswith(unexpected_start):
      self.fail('%r does start with %r' % (actual, unexpected_start), msg)

  def assertEndsWith(self, actual, expected_end, msg=None):
    """Asserts that actual.endswith(expected_end) is True.

    Args:
      actual: str
      expected_end: str
      msg: Optional message to report on failure.
    """
    if not actual.endswith(expected_end):
      self.fail('%r does not end with %r' % (actual, expected_end), msg)

  def assertNotEndsWith(self, actual, unexpected_end, msg=None):
    """Asserts that actual.endswith(unexpected_end) is False.

    Args:
      actual: str
      unexpected_end: str
      msg: Optional message to report on failure.
    """
    if actual.endswith(unexpected_end):
      self.fail('%r does end with %r' % (actual, unexpected_end), msg)

  def assertSequenceStartsWith(self, prefix, whole, msg=None):
    """An equality assertion for the beginning of ordered sequences.

    If prefix is an empty sequence, it will raise an error unless whole is also
    an empty sequence.

    If prefix is not a sequence, it will raise an error if the first element of
    whole does not match.

    Args:
      prefix: A sequence expected at the beginning of the whole parameter.
      whole: The sequence in which to look for prefix.
      msg: Optional message to report on failure.
    """
    try:
      prefix_len = len(prefix)
    except (TypeError, NotImplementedError):
      prefix = [prefix]
      prefix_len = 1

    try:
      whole_len = len(whole)
    except (TypeError, NotImplementedError):
      self.fail('For whole: len(%s) is not supported, it appears to be type: '
                '%s' % (whole, type(whole)), msg)

    assert prefix_len <= whole_len, self._formatMessage(
        msg,
        'Prefix length (%d) is longer than whole length (%d).' %
        (prefix_len, whole_len)
    )

    if not prefix_len and whole_len:
      self.fail('Prefix length is 0 but whole length is %d: %s' %
                (len(whole), whole), msg)

    try:
      self.assertSequenceEqual(prefix, whole[:prefix_len], msg)
    except AssertionError:
      self.fail('prefix: %s not found at start of whole: %s.' %
                (prefix, whole), msg)

  def assertEmpty(self, container, msg=None):
    """Asserts that an object has zero length.

    Args:
      container: Anything that implements the collections.Sized interface.
      msg: Optional message to report on failure.
    """
    if not isinstance(container, collections.Sized):
      self.fail('Expected a Sized object, got: '
                '{!r}'.format(type(container).__name__), msg)

    # explicitly check the length since some Sized objects (e.g. numpy.ndarray)
    # have strange __nonzero__/__bool__ behavior.
    if len(container):  # pylint: disable=g-explicit-length-test
      self.fail('{!r} has length of {}.'.format(container, len(container)), msg)

  def assertNotEmpty(self, container, msg=None):
    """Asserts that an object has non-zero length.

    Args:
      container: Anything that implements the collections.Sized interface.
      msg: Optional message to report on failure.
    """
    if not isinstance(container, collections.Sized):
      self.fail('Expected a Sized object, got: '
                '{!r}'.format(type(container).__name__), msg)

    # explicitly check the length since some Sized objects (e.g. numpy.ndarray)
    # have strange __nonzero__/__bool__ behavior.
    if not len(container):  # pylint: disable=g-explicit-length-test
      self.fail('{!r} has length of 0.'.format(container), msg)

  def assertLen(self, container, expected_len, msg=None):
    """Asserts that an object has the expected length.

    Args:
      container: Anything that implements the collections.Sized interface.
      expected_len: The expected length of the container.
      msg: Optional message to report on failure.
    """
    if not isinstance(container, collections.Sized):
      self.fail('Expected a Sized object, got: '
                '{!r}'.format(type(container).__name__), msg)
    if len(container) != expected_len:
      container_repr = unittest.util.safe_repr(container)
      self.fail('{} has length of {}, expected {}.'.format(
          container_repr, len(container), expected_len), msg)

  def assertSequenceAlmostEqual(self, expected_seq, actual_seq, places=None,
                                msg=None, delta=None):
    """An approximate equality assertion for ordered sequences.

    Fail if the two sequences are unequal as determined by their value
    differences rounded to the given number of decimal places (default 7) and
    comparing to zero, or by comparing that the difference between each value
    in the two sequences is more than the given delta.

    Note that decimal places (from zero) are usually not the same as significant
    digits (measured from the most signficant digit).

    If the two sequences compare equal then they will automatically compare
    almost equal.

    Args:
      expected_seq: A sequence containing elements we are expecting.
      actual_seq: The sequence that we are testing.
      places: The number of decimal places to compare.
      msg: The message to be printed if the test fails.
      delta: The OK difference between compared values.
    """
    if len(expected_seq) != len(actual_seq):
      self.fail('Sequence size mismatch: {} vs {}'.format(
          len(expected_seq), len(actual_seq)), msg)

    err_list = []
    for idx, (exp_elem, act_elem) in enumerate(zip(expected_seq, actual_seq)):
      try:
        self.assertAlmostEqual(exp_elem, act_elem, places=places, msg=msg,
                               delta=delta)
      except self.failureException as err:
        err_list.append('At index {}: {}'.format(idx, err))

    if err_list:
      if len(err_list) > 30:
        err_list = err_list[:30] + ['...']
      msg = self._formatMessage(msg, '\n'.join(err_list))
      self.fail(msg)

  def assertContainsSubset(self, expected_subset, actual_set, msg=None):
    """Checks whether actual iterable is a superset of expected iterable."""
    missing = set(expected_subset) - set(actual_set)
    if not missing:
      return

    self.fail('Missing elements %s\nExpected: %s\nActual: %s' % (
        missing, expected_subset, actual_set), msg)

  def assertNoCommonElements(self, expected_seq, actual_seq, msg=None):
    """Checks whether actual iterable and expected iterable are disjoint."""
    common = set(expected_seq) & set(actual_seq)
    if not common:
      return

    self.fail('Common elements %s\nExpected: %s\nActual: %s' % (
        common, expected_seq, actual_seq), msg)

  def assertItemsEqual(self, expected_seq, actual_seq, msg=None):
    """An unordered sequence specific comparison.

    Equivalent to assertCountEqual(). This method is a compatibility layer
    for Python 3k, since 2to3 does not convert assertItemsEqual() calls into
    assertCountEqual() calls.

    Args:
      expected_seq: A sequence containing elements we are expecting.
      actual_seq: The sequence that we are testing.
      msg: The message to be printed if the test fails.
    """

    if not hasattr(super(TestCase, self), 'assertItemsEqual'):
      # The assertItemsEqual method was renamed assertCountEqual in Python 3.2
      super(TestCase, self).assertCountEqual(expected_seq, actual_seq, msg)
      return

    super(TestCase, self).assertItemsEqual(expected_seq, actual_seq, msg)

  def assertCountEqual(self, expected_seq, actual_seq, msg=None):
    """An unordered sequence specific comparison.

    It asserts that actual_seq and expected_seq have the same element counts.
    Equivalent to::

        self.assertEqual(Counter(iter(actual_seq)),
                         Counter(iter(expected_seq)))

    Asserts that each element has the same count in both sequences.
    Example:
        - [0, 1, 1] and [1, 0, 1] compare equal.
        - [0, 0, 1] and [0, 1] compare unequal.

    Args:
      expected_seq: A sequence containing elements we are expecting.
      actual_seq: The sequence that we are testing.
      msg: The message to be printed if the test fails.

    """
    self.assertItemsEqual(expected_seq, actual_seq, msg)

  def assertSameElements(self, expected_seq, actual_seq, msg=None):
    """Asserts that two sequences have the same elements (in any order).

    This method, unlike assertCountEqual, doesn't care about any
    duplicates in the expected and actual sequences.

      >> assertSameElements([1, 1, 1, 0, 0, 0], [0, 1])
      # Doesn't raise an AssertionError

    If possible, you should use assertCountEqual instead of
    assertSameElements.

    Args:
      expected_seq: A sequence containing elements we are expecting.
      actual_seq: The sequence that we are testing.
      msg: The message to be printed if the test fails.
    """
    # `unittest2.TestCase` used to have assertSameElements, but it was
    # removed in favor of assertItemsEqual. As there's a unit test
    # that explicitly checks this behavior, I am leaving this method
    # alone.
    # Fail on strings: empirically, passing strings to this test method
    # is almost always a bug. If comparing the character sets of two strings
    # is desired, cast the inputs to sets or lists explicitly.
    if (isinstance(expected_seq, _TEXT_OR_BINARY_TYPES) or
        isinstance(actual_seq, _TEXT_OR_BINARY_TYPES)):
      self.fail('Passing string/bytes to assertSameElements is usually a bug. '
                'Did you mean to use assertEqual?\n'
                'Expected: %s\nActual: %s' % (expected_seq, actual_seq))
    try:
      expected = dict([(element, None) for element in expected_seq])
      actual = dict([(element, None) for element in actual_seq])
      missing = [element for element in expected if element not in actual]
      unexpected = [element for element in actual if element not in expected]
      missing.sort()
      unexpected.sort()
    except TypeError:
      # Fall back to slower list-compare if any of the objects are
      # not hashable.
      expected = list(expected_seq)
      actual = list(actual_seq)
      expected.sort()
      actual.sort()
      missing, unexpected = _sorted_list_difference(expected, actual)
    errors = []
    if msg:
      errors.extend((msg, ':\n'))
    if missing:
      errors.append('Expected, but missing:\n  %r\n' % missing)
    if unexpected:
      errors.append('Unexpected, but present:\n  %r\n' % unexpected)
    if missing or unexpected:
      self.fail(''.join(errors))

  # unittest.TestCase.assertMultiLineEqual works very similarly, but it
  # has a different error format. However, I find this slightly more readable.
  def assertMultiLineEqual(self, first, second, msg=None, **kwargs):
    """Asserts that two multi-line strings are equal."""
    assert isinstance(first, six.string_types), (
        'First argument is not a string: %r' % (first,))
    assert isinstance(second, six.string_types), (
        'Second argument is not a string: %r' % (second,))
    line_limit = kwargs.pop('line_limit', 0)
    if kwargs:
      raise TypeError('Unexpected keyword args {}'.format(tuple(kwargs)))

    if first == second:
      return
    if msg:
      failure_message = [msg + ':\n']
    else:
      failure_message = ['\n']
    if line_limit:
      line_limit += len(failure_message)
    for line in difflib.ndiff(first.splitlines(True), second.splitlines(True)):
      failure_message.append(line)
      if not line.endswith('\n'):
        failure_message.append('\n')
    if line_limit and len(failure_message) > line_limit:
      n_omitted = len(failure_message) - line_limit
      failure_message = failure_message[:line_limit]
      failure_message.append(
          '(... and {} more delta lines omitted for brevity.)\n'.format(
              n_omitted))

    raise self.failureException(''.join(failure_message))

  def assertBetween(self, value, minv, maxv, msg=None):
    """Asserts that value is between minv and maxv (inclusive)."""
    msg = self._formatMessage(msg,
                              '"%r" unexpectedly not between "%r" and "%r"' %
                              (value, minv, maxv))
    self.assertTrue(minv <= value, msg)
    self.assertTrue(maxv >= value, msg)

  def assertRegexMatch(self, actual_str, regexes, message=None):
    r"""Asserts that at least one regex in regexes matches str.

    If possible you should use assertRegexpMatches, which is a simpler
    version of this method. assertRegexpMatches takes a single regular
    expression (a string or re compiled object) instead of a list.

    Notes:
    1. This function uses substring matching, i.e. the matching
       succeeds if *any* substring of the error message matches *any*
       regex in the list.  This is more convenient for the user than
       full-string matching.

    2. If regexes is the empty list, the matching will always fail.

    3. Use regexes=[''] for a regex that will always pass.

    4. '.' matches any single character *except* the newline.  To
       match any character, use '(.|\n)'.

    5. '^' matches the beginning of each line, not just the beginning
       of the string.  Similarly, '$' matches the end of each line.

    6. An exception will be thrown if regexes contains an invalid
       regex.

    Args:
      actual_str:  The string we try to match with the items in regexes.
      regexes:  The regular expressions we want to match against str.
          See "Notes" above for detailed notes on how this is interpreted.
      message:  The message to be printed if the test fails.
    """
    if isinstance(regexes, _TEXT_OR_BINARY_TYPES):
      self.fail('regexes is string or bytes; use assertRegexpMatches instead.',
                message)
    if not regexes:
      self.fail('No regexes specified.', message)

    regex_type = type(regexes[0])
    for regex in regexes[1:]:
      if type(regex) is not regex_type:  # pylint: disable=unidiomatic-typecheck
        self.fail('regexes list must all be the same type.', message)

    if regex_type is bytes and isinstance(actual_str, six.text_type):
      regexes = [regex.decode('utf-8') for regex in regexes]
      regex_type = six.text_type
    elif regex_type is six.text_type and isinstance(actual_str, bytes):
      regexes = [regex.encode('utf-8') for regex in regexes]
      regex_type = bytes

    if regex_type is six.text_type:
      regex = u'(?:%s)' % u')|(?:'.join(regexes)
    elif regex_type is bytes:
      regex = b'(?:' + (b')|(?:'.join(regexes)) + b')'
    else:
      self.fail('Only know how to deal with unicode str or bytes regexes.',
                message)

    if not re.search(regex, actual_str, re.MULTILINE):
      self.fail('"%s" does not contain any of these regexes: %s.' %
                (actual_str, regexes), message)

  def assertCommandSucceeds(self, command, regexes=(b'',), env=None,
                            close_fds=True, msg=None):
    """Asserts that a shell command succeeds (i.e. exits with code 0).

    Args:
      command: List or string representing the command to run.
      regexes: List of regular expression byte strings that match success.
      env: Dictionary of environment variable settings. If None, no environment
          variables will be set for the child process. This is to make tests
          more hermetic. NOTE: this behavior is different than the standard
          subprocess module.
      close_fds: Whether or not to close all open fd's in the child after
          forking.
      msg: Optional message to report on failure.
    """
    (ret_code, err) = get_command_stderr(command, env, close_fds)

    # We need bytes regexes here because `err` is bytes.
    # Accommodate code which listed their output regexes w/o the b'' prefix by
    # converting them to bytes for the user.
    if isinstance(regexes[0], six.text_type):
      regexes = [regex.encode('utf-8') for regex in regexes]

    command_string = get_command_string(command)
    self.assertEqual(
        ret_code, 0,
        self._formatMessage(msg,
                            'Running command\n'
                            '%s failed with error code %s and message\n'
                            '%s' % (_quote_long_string(command_string),
                                    ret_code,
                                    _quote_long_string(err)))
    )
    self.assertRegexMatch(
        err,
        regexes,
        message=self._formatMessage(
            msg,
            'Running command\n'
            '%s failed with error code %s and message\n'
            '%s which matches no regex in %s' % (
                _quote_long_string(command_string),
                ret_code,
                _quote_long_string(err),
                regexes)))

  def assertCommandFails(self, command, regexes, env=None, close_fds=True,
                         msg=None):
    """Asserts a shell command fails and the error matches a regex in a list.

    Args:
      command: List or string representing the command to run.
      regexes: the list of regular expression strings.
      env: Dictionary of environment variable settings. If None, no environment
          variables will be set for the child process. This is to make tests
          more hermetic. NOTE: this behavior is different than the standard
          subprocess module.
      close_fds: Whether or not to close all open fd's in the child after
          forking.
      msg: Optional message to report on failure.
    """
    (ret_code, err) = get_command_stderr(command, env, close_fds)

    # We need bytes regexes here because `err` is bytes.
    # Accommodate code which listed their output regexes w/o the b'' prefix by
    # converting them to bytes for the user.
    if isinstance(regexes[0], six.text_type):
      regexes = [regex.encode('utf-8') for regex in regexes]

    command_string = get_command_string(command)
    self.assertNotEqual(
        ret_code, 0,
        self._formatMessage(msg, 'The following command succeeded '
                            'while expected to fail:\n%s' %
                            _quote_long_string(command_string)))
    self.assertRegexMatch(
        err,
        regexes,
        message=self._formatMessage(
            msg,
            'Running command\n'
            '%s failed with error code %s and message\n'
            '%s which matches no regex in %s' % (
                _quote_long_string(command_string),
                ret_code,
                _quote_long_string(err),
                regexes)))

  class _AssertRaisesContext(object):

    def __init__(self, expected_exception, test_case, test_func, msg=None):
      self.expected_exception = expected_exception
      self.test_case = test_case
      self.test_func = test_func
      self.msg = msg

    def __enter__(self):
      return self

    def __exit__(self, exc_type, exc_value, tb):
      if exc_type is None:
        self.test_case.fail(self.expected_exception.__name__ + ' not raised',
                            self.msg)
      if not issubclass(exc_type, self.expected_exception):
        return False
      self.test_func(exc_value)
      return True

  def assertRaisesWithPredicateMatch(self, expected_exception, predicate,
                                     callable_obj=None, *args, **kwargs):
    """Asserts that exception is thrown and predicate(exception) is true.

    Args:
      expected_exception: Exception class expected to be raised.
      predicate: Function of one argument that inspects the passed-in exception
          and returns True (success) or False (please fail the test).
      callable_obj: Function to be called.
      *args: Extra args.
      **kwargs: Extra keyword args.

    Returns:
      A context manager if callable_obj is None. Otherwise, None.

    Raises:
      self.failureException if callable_obj does not raise a matching exception.
    """
    def Check(err):
      self.assertTrue(predicate(err),
                      '%r does not match predicate %r' % (err, predicate))

    context = self._AssertRaisesContext(expected_exception, self, Check)
    if callable_obj is None:
      return context
    with context:
      callable_obj(*args, **kwargs)

  def assertRaisesWithLiteralMatch(self, expected_exception,
                                   expected_exception_message,
                                   callable_obj=None, *args, **kwargs):
    """Asserts that the message in a raised exception equals the given string.

    Unlike assertRaisesRegexp, this method takes a literal string, not
    a regular expression.

    with self.assertRaisesWithLiteralMatch(ExType, 'message'):
      DoSomething()

    Args:
      expected_exception: Exception class expected to be raised.
      expected_exception_message: String message expected in the raised
          exception.  For a raise exception e, expected_exception_message must
          equal str(e).
      callable_obj: Function to be called, or None to return a context.
      *args: Extra args.
      **kwargs: Extra kwargs.

    Returns:
      A context manager if callable_obj is None. Otherwise, None.

    Raises:
      self.failureException if callable_obj does not raise a matching exception.
    """
    def Check(err):
      actual_exception_message = str(err)
      self.assertTrue(expected_exception_message == actual_exception_message,
                      'Exception message does not match.\n'
                      'Expected: %r\n'
                      'Actual: %r' % (expected_exception_message,
                                      actual_exception_message))

    context = self._AssertRaisesContext(expected_exception, self, Check)
    if callable_obj is None:
      return context
    with context:
      callable_obj(*args, **kwargs)

  def assertContainsInOrder(self, strings, target, msg=None):
    """Asserts that the strings provided are found in the target in order.

    This may be useful for checking HTML output.

    Args:
      strings: A list of strings, such as [ 'fox', 'dog' ]
      target: A target string in which to look for the strings, such as
          'The quick brown fox jumped over the lazy dog'.
      msg: Optional message to report on failure.
    """
    if isinstance(strings, (bytes, unicode if str is bytes else str)):
      strings = (strings,)

    current_index = 0
    last_string = None
    for string in strings:
      index = target.find(str(string), current_index)
      if index == -1 and current_index == 0:
        self.fail("Did not find '%s' in '%s'" %
                  (string, target), msg)
      elif index == -1:
        self.fail("Did not find '%s' after '%s' in '%s'" %
                  (string, last_string, target), msg)
      last_string = string
      current_index = index

  def assertContainsSubsequence(self, container, subsequence, msg=None):
    """Asserts that "container" contains "subsequence" as a subsequence.

    Asserts that "container" contains all the elements of "subsequence", in
    order, but possibly with other elements interspersed. For example, [1, 2, 3]
    is a subsequence of [0, 0, 1, 2, 0, 3, 0] but not of [0, 0, 1, 3, 0, 2, 0].

    Args:
      container: the list we're testing for subsequence inclusion.
      subsequence: the list we hope will be a subsequence of container.
      msg: Optional message to report on failure.
    """
    first_nonmatching = None
    reversed_container = list(reversed(container))
    subsequence = list(subsequence)

    for e in subsequence:
      if e not in reversed_container:
        first_nonmatching = e
        break
      while e != reversed_container.pop():
        pass

    if first_nonmatching is not None:
      self.fail('%s not a subsequence of %s. First non-matching element: %s' %
                (subsequence, container, first_nonmatching), msg)

  def assertContainsExactSubsequence(self, container, subsequence, msg=None):
    """Asserts that "container" contains "subsequence" as an exact subsequence.

    Asserts that "container" contains all the elements of "subsequence", in
    order, and without other elements interspersed. For example, [1, 2, 3] is an
    exact subsequence of [0, 0, 1, 2, 3, 0] but not of [0, 0, 1, 2, 0, 3, 0].

    Args:
      container: the list we're testing for subsequence inclusion.
      subsequence: the list we hope will be an exact subsequence of container.
      msg: Optional message to report on failure.
    """
    container = list(container)
    subsequence = list(subsequence)
    longest_match = 0

    for start in xrange(1 + len(container) - len(subsequence)):
      if longest_match == len(subsequence):
        break
      index = 0
      while (index < len(subsequence) and
             subsequence[index] == container[start + index]):
        index += 1
      longest_match = max(longest_match, index)

    if longest_match < len(subsequence):
      self.fail('%s not an exact subsequence of %s. '
                'Longest matching prefix: %s' %
                (subsequence, container, subsequence[:longest_match]), msg)

  def assertTotallyOrdered(self, *groups, **kwargs):
    """Asserts that total ordering has been implemented correctly.

    For example, say you have a class A that compares only on its attribute x.
    Comparators other than __lt__ are omitted for brevity.

    class A(object):
      def __init__(self, x, y):
        self.x = x
        self.y = y

      def __hash__(self):
        return hash(self.x)

      def __lt__(self, other):
        try:
          return self.x < other.x
        except AttributeError:
          return NotImplemented

    assertTotallyOrdered will check that instances can be ordered correctly.
    For example,

    self.assertTotallyOrdered(
      [None],  # None should come before everything else.
      [1],     # Integers sort earlier.
      [A(1, 'a')],
      [A(2, 'b')],  # 2 is after 1.
      [A(3, 'c'), A(3, 'd')],  # The second argument is irrelevant.
      [A(4, 'z')],
      ['foo'])  # Strings sort last.

    Args:
     *groups: A list of groups of elements.  Each group of elements is a list
         of objects that are equal.  The elements in each group must be less
         than the elements in the group after it.  For example, these groups are
         totally ordered: [None], [1], [2, 2], [3].
      **kwargs: optional msg keyword argument can be passed.
    """

    def CheckOrder(small, big):
      """Ensures small is ordered before big."""
      self.assertFalse(small == big,
                       self._formatMessage(msg, '%r unexpectedly equals %r' %
                                           (small, big)))
      self.assertTrue(small != big,
                      self._formatMessage(msg, '%r unexpectedly equals %r' %
                                          (small, big)))
      self.assertLess(small, big, msg)
      self.assertFalse(big < small,
                       self._formatMessage(msg,
                                           '%r unexpectedly less than %r' %
                                           (big, small)))
      self.assertLessEqual(small, big, msg)
      self.assertFalse(big <= small, self._formatMessage(
          '%r unexpectedly less than or equal to %r' % (big, small), msg
      ))
      self.assertGreater(big, small, msg)
      self.assertFalse(small > big,
                       self._formatMessage(msg,
                                           '%r unexpectedly greater than %r' %
                                           (small, big)))
      self.assertGreaterEqual(big, small)
      self.assertFalse(small >= big, self._formatMessage(
          msg,
          '%r unexpectedly greater than or equal to %r' % (small, big)))

    def CheckEqual(a, b):
      """Ensures that a and b are equal."""
      self.assertEqual(a, b, msg)
      self.assertFalse(a != b,
                       self._formatMessage(msg, '%r unexpectedly unequals %r' %
                                           (a, b)))
      self.assertEqual(hash(a), hash(b), self._formatMessage(
          msg,
          'hash %d of %r unexpectedly not equal to hash %d of %r' %
          (hash(a), a, hash(b), b)))
      self.assertFalse(a < b,
                       self._formatMessage(msg,
                                           '%r unexpectedly less than %r' %
                                           (a, b)))
      self.assertFalse(b < a,
                       self._formatMessage(msg,
                                           '%r unexpectedly less than %r' %
                                           (b, a)))
      self.assertLessEqual(a, b, msg)
      self.assertLessEqual(b, a, msg)
      self.assertFalse(a > b,
                       self._formatMessage(msg,
                                           '%r unexpectedly greater than %r' %
                                           (a, b)))
      self.assertFalse(b > a,
                       self._formatMessage(msg,
                                           '%r unexpectedly greater than %r' %
                                           (b, a)))
      self.assertGreaterEqual(a, b, msg)
      self.assertGreaterEqual(b, a, msg)

    msg = kwargs.get('msg')

    # For every combination of elements, check the order of every pair of
    # elements.
    for elements in itertools.product(*groups):
      elements = list(elements)
      for index, small in enumerate(elements[:-1]):
        for big in elements[index + 1:]:
          CheckOrder(small, big)

    # Check that every element in each group is equal.
    for group in groups:
      for a in group:
        CheckEqual(a, a)
      for a, b in itertools.product(group, group):
        CheckEqual(a, b)

  def assertDictEqual(self, a, b, msg=None):
    """Raises AssertionError if a and b are not equal dictionaries.

    Args:
      a: A dict, the expected value.
      b: A dict, the actual value.
      msg: An optional str, the associated message.

    Raises:
      AssertionError: if the dictionaries are not equal.
    """
    self.assertIsInstance(a, dict, self._formatMessage(
        msg,
        'First argument is not a dictionary'
    ))
    self.assertIsInstance(b, dict, self._formatMessage(
        msg,
        'Second argument is not a dictionary'
    ))

    def Sorted(list_of_items):
      try:
        return sorted(list_of_items)  # In 3.3, unordered are possible.
      except TypeError:
        return list_of_items

    if a == b:
      return
    a_items = Sorted(list(six.iteritems(a)))
    b_items = Sorted(list(six.iteritems(b)))

    unexpected = []
    missing = []
    different = []

    safe_repr = unittest.util.safe_repr

    def Repr(dikt):
      """Deterministic repr for dict."""
      # Sort the entries based on their repr, not based on their sort order,
      # which will be non-deterministic across executions, for many types.
      entries = sorted((safe_repr(k), safe_repr(v))
                       for k, v in six.iteritems(dikt))
      return '{%s}' % (', '.join('%s: %s' % pair for pair in entries))

    message = ['%s != %s%s' % (Repr(a), Repr(b), ' (%s)' % msg if msg else '')]

    # The standard library default output confounds lexical difference with
    # value difference; treat them separately.
    for a_key, a_value in a_items:
      if a_key not in b:
        missing.append((a_key, a_value))
      elif a_value != b[a_key]:
        different.append((a_key, a_value, b[a_key]))

    for b_key, b_value in b_items:
      if b_key not in a:
        unexpected.append((b_key, b_value))

    if unexpected:
      message.append(
          'Unexpected, but present entries:\n%s' % ''.join(
              '%s: %s\n' % (safe_repr(k), safe_repr(v)) for k, v in unexpected))

    if different:
      message.append(
          'repr() of differing entries:\n%s' % ''.join(
              '%s: %s != %s\n' % (safe_repr(k), safe_repr(a_value),
                                  safe_repr(b_value))
              for k, a_value, b_value in different))

    if missing:
      message.append(
          'Missing entries:\n%s' % ''.join(
              ('%s: %s\n' % (safe_repr(k), safe_repr(v)) for k, v in missing)))

    raise self.failureException('\n'.join(message))

  def assertUrlEqual(self, a, b, msg=None):
    """Asserts that urls are equal, ignoring ordering of query params."""
    parsed_a = urllib.parse.urlparse(a)
    parsed_b = urllib.parse.urlparse(b)
    self.assertEqual(parsed_a.scheme, parsed_b.scheme, msg)
    self.assertEqual(parsed_a.netloc, parsed_b.netloc, msg)
    self.assertEqual(parsed_a.path, parsed_b.path, msg)
    self.assertEqual(parsed_a.fragment, parsed_b.fragment, msg)
    self.assertEqual(sorted(parsed_a.params.split(';')),
                     sorted(parsed_b.params.split(';')), msg)
    self.assertDictEqual(
        urllib.parse.parse_qs(parsed_a.query, keep_blank_values=True),
        urllib.parse.parse_qs(parsed_b.query, keep_blank_values=True), msg)

  def assertSameStructure(self, a, b, aname='a', bname='b', msg=None):
    """Asserts that two values contain the same structural content.

    The two arguments should be data trees consisting of trees of dicts and
    lists. They will be deeply compared by walking into the contents of dicts
    and lists; other items will be compared using the == operator.
    If the two structures differ in content, the failure message will indicate
    the location within the structures where the first difference is found.
    This may be helpful when comparing large structures.

    Args:
      a: The first structure to compare.
      b: The second structure to compare.
      aname: Variable name to use for the first structure in assertion messages.
      bname: Variable name to use for the second structure.
      msg: Additional text to include in the failure message.
    """

    # Accumulate all the problems found so we can report all of them at once
    # rather than just stopping at the first
    problems = []

    _walk_structure_for_problems(a, b, aname, bname, problems)

    # Avoid spamming the user toooo much
    if self.maxDiff is not None:
      max_problems_to_show = self.maxDiff // 80
      if len(problems) > max_problems_to_show:
        problems = problems[0:max_problems_to_show-1] + ['...']

    if problems:
      self.fail('; '.join(problems), msg)

  def assertJsonEqual(self, first, second, msg=None):
    """Asserts that the JSON objects defined in two strings are equal.

    A summary of the differences will be included in the failure message
    using assertSameStructure.

    Args:
      first: A string contining JSON to decode and compare to second.
      second: A string contining JSON to decode and compare to first.
      msg: Additional text to include in the failure message.
    """
    try:
      first_structured = json.loads(first)
    except ValueError as e:
      raise ValueError(self._formatMessage(
          msg,
          'could not decode first JSON value %s: %s' % (first, e)))

    try:
      second_structured = json.loads(second)
    except ValueError as e:
      raise ValueError(self._formatMessage(
          msg,
          'could not decode second JSON value %s: %s' % (second, e)))

    self.assertSameStructure(first_structured, second_structured,
                             aname='first', bname='second', msg=msg)

  def _getAssertEqualityFunc(self, first, second):
    try:
      return super(TestCase, self)._getAssertEqualityFunc(first, second)
    except AttributeError:
      # This is a workaround if unittest.TestCase.__init__ was never run.
      # It usually means that somebody created a subclass just for the
      # assertions and has overridden __init__. "assertTrue" is a safe
      # value that will not make __init__ raise a ValueError.
      test_method = getattr(self, '_testMethodName', 'assertTrue')
      super(TestCase, self).__init__(test_method)

    return super(TestCase, self)._getAssertEqualityFunc(first, second)

  def fail(self, msg=None, prefix=None):
    """Fail immediately with the given message, optionally prefixed."""
    return super(TestCase, self).fail(self._formatMessage(prefix, msg))


def _sorted_list_difference(expected, actual):
  """Finds elements in only one or the other of two, sorted input lists.

  Returns a two-element tuple of lists.  The first list contains those
  elements in the "expected" list but not in the "actual" list, and the
  second contains those elements in the "actual" list but not in the
  "expected" list.  Duplicate elements in either input list are ignored.

  Args:
    expected:  The list we expected.
    actual:  The list we actualy got.
  Returns:
    (missing, unexpected)
    missing: items in expected that are not in actual.
    unexpected: items in actual that are not in expected.
  """
  i = j = 0
  missing = []
  unexpected = []
  while True:
    try:
      e = expected[i]
      a = actual[j]
      if e < a:
        missing.append(e)
        i += 1
        while expected[i] == e:
          i += 1
      elif e > a:
        unexpected.append(a)
        j += 1
        while actual[j] == a:
          j += 1
      else:
        i += 1
        try:
          while expected[i] == e:
            i += 1
        finally:
          j += 1
          while actual[j] == a:
            j += 1
    except IndexError:
      missing.extend(expected[i:])
      unexpected.extend(actual[j:])
      break
  return missing, unexpected


def _walk_structure_for_problems(a, b, aname, bname, problem_list):
  """The recursive comparison behind assertSameStructure."""
  if type(a) != type(b) and not (  # pylint: disable=unidiomatic-typecheck
      isinstance(a, six.integer_types) and isinstance(b, six.integer_types)):
    # We do not distinguish between int and long types as 99.99% of Python 2
    # code should never care.  They collapse into a single type in Python 3.
    problem_list.append('%s is a %r but %s is a %r' %
                        (aname, type(a), bname, type(b)))
    # If they have different types there's no point continuing
    return

  if isinstance(a, collections.Mapping):
    for k in a:
      if k in b:
        _walk_structure_for_problems(
            a[k], b[k], '%s[%r]' % (aname, k), '%s[%r]' % (bname, k),
            problem_list)
      else:
        problem_list.append(
            "%s has [%r] with value %r but it's missing in %s" %
            (aname, k, a[k], bname))
    for k in b:
      if k not in a:
        problem_list.append(
            '%s lacks [%r] but %s has it with value %r' %
            (aname, k, bname, b[k]))

  # Strings/bytes are Sequences but we'll just do those with regular !=
  elif (isinstance(a, collections.Sequence) and
        not isinstance(a, _TEXT_OR_BINARY_TYPES)):
    minlen = min(len(a), len(b))
    for i in xrange(minlen):
      _walk_structure_for_problems(
          a[i], b[i], '%s[%d]' % (aname, i), '%s[%d]' % (bname, i),
          problem_list)
    for i in xrange(minlen, len(a)):
      problem_list.append('%s has [%i] with value %r but %s does not' %
                          (aname, i, a[i], bname))
    for i in xrange(minlen, len(b)):
      problem_list.append('%s lacks [%i] but %s has it with value %r' %
                          (aname, i, bname, b[i]))

  else:
    if a != b:
      problem_list.append('%s is %r but %s is %r' % (aname, a, bname, b))


def get_command_string(command):
  """Returns an escaped string that can be used as a shell command.

  Args:
    command: List or string representing the command to run.
  Returns:
    A string suitable for use as a shell command.
  """
  if isinstance(command, six.string_types):
    return command
  else:
    if os.name == 'nt':
      return ' '.join(command)
    else:
      # The following is identical to Python 3's shlex.quote function.
      command_string = ''
      for word in command:
        # Single quote word, and replace each ' in word with '"'"'
        command_string += "'" + word.replace("'", "'\"'\"'") + "' "
      return command_string[:-1]


def get_command_stderr(command, env=None, close_fds=True):
  """Runs the given shell command and returns a tuple.

  Args:
    command: List or string representing the command to run.
    env: Dictionary of environment variable settings. If None, no environment
        variables will be set for the child process. This is to make tests
        more hermetic. NOTE: this behavior is different than the standard
        subprocess module.
    close_fds: Whether or not to close all open fd's in the child after forking.
        On Windows, this is ignored and close_fds is always False.

  Returns:
    Tuple of (exit status, text printed to stdout and stderr by the command).
  """
  if env is None: env = {}
  if os.name == 'nt':
    # Windows does not support setting close_fds to True while also redirecting
    # standard handles.
    close_fds = False

  use_shell = isinstance(command, six.string_types)
  process = subprocess.Popen(
      command,
      close_fds=close_fds,
      env=env,
      shell=use_shell,
      stderr=subprocess.STDOUT,
      stdout=subprocess.PIPE)
  output = process.communicate()[0]
  exit_status = process.wait()
  return (exit_status, output)


def _quote_long_string(s):
  """Quotes a potentially multi-line string to make the start and end obvious.

  Args:
    s: A string.

  Returns:
    The quoted string.
  """
  if isinstance(s, (bytes, bytearray)):
    try:
      s = s.decode('utf-8')
    except UnicodeDecodeError:
      s = str(s)
  return ('8<-----------\n' +
          s + '\n' +
          '----------->8\n')


class _TestProgramManualRun(unittest.TestProgram):
  """A TestProgram which runs the tests manually."""

  def runTests(self, do_run=False):
    """Runs the tests."""
    if do_run:
      unittest.TestProgram.runTests(self)


def print_python_version():
  # Having this in the test output logs by default helps debugging when all
  # you've got is the log and no other idea of which Python was used.
  sys.stderr.write('Running tests under Python {0[0]}.{0[1]}.{0[2]}: '
                   '{1}\n'.format(
                       sys.version_info,
                       sys.executable if sys.executable else 'embedded.'))


def main(*args, **kwargs):
  """Executes a set of Python unit tests.

  Usually this function is called without arguments, so the
  unittest.TestProgram instance will get created with the default settings,
  so it will run all test methods of all TestCase classes in the __main__
  module.

  Args:
    *args: Positional arguments passed through to unittest.TestProgram.__init__.
    **kwargs: Keyword arguments passed through to unittest.TestProgram.__init__.
  """
  print_python_version()
  _run_in_app(run_tests, args, kwargs)


def _is_in_app_main():
  """Returns True iff app.run is active."""
  f = sys._getframe().f_back  # pylint: disable=protected-access
  while f:
    if f.f_code == six.get_function_code(app.run):
      return True
    f = f.f_back
  return False


class _SavedFlag(object):
  """Helper class for saving and restoring a flag value."""

  def __init__(self, flag):
    self.flag = flag
    self.value = flag.value
    self.present = flag.present

  def restore_flag(self):
    self.flag.value = self.value
    self.flag.present = self.present


def _register_sigterm_with_faulthandler():
  """Have faulthandler dump stacks on SIGTERM.  Useful to diagnose timeouts."""
  if faulthandler and getattr(faulthandler, 'register', None):
    # faulthandler.register is not avaiable on Windows.
    # faulthandler.enable() is already called by app.run.
    try:
      faulthandler.register(signal.SIGTERM, chain=True)
    except Exception as e:  # pylint: disable=broad-except
      sys.stderr.write('faulthandler.register(SIGTERM) failed '
                       '%r; ignoring.\n' % e)


def _run_in_app(function, args, kwargs):
  """Executes a set of Python unit tests, ensuring app.run.

  This is a private function, users should call absltest.main().

  _run_in_app calculates argv to be the command-line arguments of this program
  (without the flags), sets the default of FLAGS.alsologtostderr to True,
  then it calls function(argv, args, kwargs), making sure that `function'
  will get called within app.run(). _run_in_app does this by checking whether
  it is called by app.run(), or by calling app.run() explicitly.

  The reason why app.run has to be ensured is to make sure that
  flags are parsed and stripped properly, and other initializations done by
  the app module are also carried out, no matter if absltest.run() is called
  from within or outside app.run().

  If _run_in_app is called from within app.run(), then it will reparse
  sys.argv and pass the result without command-line flags into the argv
  argument of `function'. The reason why this parsing is needed is that
  __main__.main() calls absltest.main() without passing its argv. So the
  only way _run_in_app could get to know the argv without the flags is that
  it reparses sys.argv.

  _run_in_app changes the default of FLAGS.alsologtostderr to True so that the
  test program's stderr will contain all the log messages unless otherwise
  specified on the command-line. This overrides any explicit assignment to
  FLAGS.alsologtostderr by the test program prior to the call to _run_in_app()
  (e.g. in __main__.main).

  Please note that _run_in_app (and the function it calls) is allowed to make
  changes to kwargs.

  Args:
    function: absltest.run_tests or a similar function. It will be called as
        function(argv, args, kwargs) where argv is a list containing the
        elements of sys.argv without the command-line flags.
    args: Positional arguments passed through to unittest.TestProgram.__init__.
    kwargs: Keyword arguments passed through to unittest.TestProgram.__init__.
  """
  if _is_in_app_main():
    _register_sigterm_with_faulthandler()

    # Save command-line flags so the side effects of FLAGS(sys.argv) can be
    # undone.
    flag_objects = (FLAGS[name] for name in FLAGS)
    saved_flags = dict((f.name, _SavedFlag(f)) for f in flag_objects)

    # Change the default of alsologtostderr from False to True, so the test
    # programs's stderr will contain all the log messages.
    # If --alsologtostderr=false is specified in the command-line, or user
    # has called FLAGS.alsologtostderr = False before, then the value is kept
    # False.
    FLAGS.set_default('alsologtostderr', True)
    # Remove it from saved flags so it doesn't get restored later.
    del saved_flags['alsologtostderr']

    # The call FLAGS(sys.argv) parses sys.argv, returns the arguments
    # without the flags, and -- as a side effect -- modifies flag values in
    # FLAGS. We don't want the side effect, because we don't want to
    # override flag changes the program did (e.g. in __main__.main)
    # after the command-line has been parsed. So we have the for loop below
    # to change back flags to their old values.
    argv = FLAGS(sys.argv)
    for saved_flag in six.itervalues(saved_flags):
      saved_flag.restore_flag()


    function(argv, args, kwargs)
  else:
    # Send logging to stderr. Use --alsologtostderr instead of --logtostderr
    # in case tests are reading their own logs.
    FLAGS.set_default('alsologtostderr', True)

    def main_function(argv):
      _register_sigterm_with_faulthandler()
      function(argv, args, kwargs)

    app.run(main=main_function)


def _is_suspicious_attribute(testCaseClass, name):
  """Returns True if an attribute is a method named like a test method."""
  if name.startswith('Test') and len(name) > 4 and name[4].isupper():
    attr = getattr(testCaseClass, name)
    if inspect.isfunction(attr) or inspect.ismethod(attr):
      args = inspect.getargspec(attr)
      return (len(args.args) == 1 and args.args[0] == 'self'
              and args.varargs is None and args.keywords is None)
  return False


class TestLoader(unittest.TestLoader):
  """A test loader which supports common test features.

  Supported features include:
   * Banning untested methods with test-like names: methods attached to this
     testCase with names starting with `Test` are ignored by the test runner,
     and often represent mistakenly-omitted test cases. This loader will raise
     a TypeError when attempting to load a TestCase with such methods.
   * Randomization of test case execution order (optional).
  """

  _ERROR_MSG = textwrap.dedent("""Method '%s' is named like a test case but
  is not one. This is often a bug. If you want it to be a test method,
  name it with 'test' in lowercase. If not, rename the method to not begin
  with 'Test'.""")

  def __init__(self, *args, **kwds):
    super(TestLoader, self).__init__(*args, **kwds)
    seed = _get_default_randomize_ordering_seed()
    if seed:
      self._seed = seed
      self._random = random.Random(self._seed)
    else:
      self._seed = None
      self._random = None

  def getTestCaseNames(self, testCaseClass):  # pylint:disable=invalid-name
    """Validates and returns a (possibly randomized) list of test case names."""
    for name in dir(testCaseClass):
      if _is_suspicious_attribute(testCaseClass, name):
        raise TypeError(TestLoader._ERROR_MSG % name)
    names = super(TestLoader, self).getTestCaseNames(testCaseClass)
    if self._seed is not None:
      logging.info('Randomizing test order with seed: %d', self._seed)
      logging.info('To reproduce this order, re-run with '
                   '--test_randomize_ordering_seed=%d', self._seed)
      self._random.shuffle(names)
    return names


def get_default_xml_output_filename():
  if os.environ.get('XML_OUTPUT_FILE'):
    return os.environ['XML_OUTPUT_FILE']
  elif os.environ.get('RUNNING_UNDER_TEST_DAEMON'):
    return os.path.join(os.path.dirname(FLAGS.test_tmpdir), 'test_detail.xml')
  elif os.environ.get('TEST_XMLOUTPUTDIR'):
    return os.path.join(
        os.environ['TEST_XMLOUTPUTDIR'],
        os.path.splitext(os.path.basename(sys.argv[0]))[0] + '.xml')


def _setup_filtering(argv):
  """Implements the bazel test filtering protocol.

  The following environment variable is used in this method:

    TESTBRIDGE_TEST_ONLY: string, if set, is forwarded to the unittest
      framework to use as a test filter. Its value is split with shlex
      before being passed as positional arguments on argv.

  Args:
    argv: the argv to mutate in-place.
  """
  test_filter = os.environ.get('TESTBRIDGE_TEST_ONLY')
  if argv is None or not test_filter:
    return

  argv[1:1] = shlex.split(test_filter)


def _setup_sharding(custom_loader=None):
  """Implements the bazel sharding protocol.

  The following environment variables are used in this method:

    TEST_SHARD_STATUS_FILE: string, if set, points to a file. We write a blank
      file to tell the test runner that this test implements the test sharding
      protocol.

    TEST_TOTAL_SHARDS: int, if set, sharding is requested.

    TEST_SHARD_INDEX: int, must be set if TEST_TOTAL_SHARDS is set. Specifies
      the shard index for this instance of the test process. Must satisfy:
      0 <= TEST_SHARD_INDEX < TEST_TOTAL_SHARDS.

  Args:
    custom_loader: A TestLoader to be made sharded.

  Returns:
    The test loader for shard-filtering or the standard test loader, depending
    on the sharding environment variables.
  """

  # It may be useful to write the shard file even if the other sharding
  # environment variables are not set. Test runners may use this functionality
  # to query whether a test binary implements the test sharding protocol.
  if 'TEST_SHARD_STATUS_FILE' in os.environ:
    try:
      f = None
      try:
        f = open(os.environ['TEST_SHARD_STATUS_FILE'], 'w')
        f.write('')
      except IOError:
        sys.stderr.write('Error opening TEST_SHARD_STATUS_FILE (%s). Exiting.'
                         % os.environ['TEST_SHARD_STATUS_FILE'])
        sys.exit(1)
    finally:
      if f is not None: f.close()

  base_loader = custom_loader or TestLoader()
  if 'TEST_TOTAL_SHARDS' not in os.environ:
    # Not using sharding, use the expected test loader.
    return base_loader

  total_shards = int(os.environ['TEST_TOTAL_SHARDS'])
  shard_index = int(os.environ['TEST_SHARD_INDEX'])

  if shard_index < 0 or shard_index >= total_shards:
    sys.stderr.write('ERROR: Bad sharding values. index=%d, total=%d\n' %
                     (shard_index, total_shards))
    sys.exit(1)

  # Replace the original getTestCaseNames with one that returns
  # the test case names for this shard.
  delegate_get_names = base_loader.getTestCaseNames

  bucket_iterator = itertools.cycle(xrange(total_shards))

  def getShardedTestCaseNames(testCaseClass):
    filtered_names = []
    for testcase in sorted(delegate_get_names(testCaseClass)):
      bucket = next(bucket_iterator)
      if bucket == shard_index:
        filtered_names.append(testcase)
    return filtered_names

  base_loader.getTestCaseNames = getShardedTestCaseNames
  return base_loader


def _run_and_get_tests_result(argv, args, kwargs, xml_test_runner_class):
  """Executes a set of Python unit tests and returns the result."""

  # Set up test filtering if requested in environment.
  _setup_filtering(argv)

  # Shard the (default or custom) loader if sharding is turned on.
  kwargs['testLoader'] = _setup_sharding(kwargs.get('testLoader', None))

  # XML file name is based upon (sorted by priority):
  # --xml_output_file flag, XML_OUTPUT_FILE variable,
  # TEST_XMLOUTPUTDIR variable or RUNNING_UNDER_TEST_DAEMON variable.
  if not FLAGS.xml_output_file:
    FLAGS.xml_output_file = get_default_xml_output_filename()
  xml_output_file = FLAGS.xml_output_file

  xml_output = None
  if xml_output_file:
    xml_output_dir = os.path.dirname(xml_output_file)
    if xml_output_dir and not os.path.isdir(xml_output_dir):
      try:
        os.makedirs(xml_output_dir)
      except OSError as e:
        # File exists error can occur with concurrent tests
        if e.errno != errno.EEXIST:
          raise
    if sys.version_info.major == 2:
      xml_output = open(xml_output_file, 'w')
    else:
      xml_output = open(xml_output_file, 'w', encoding='utf-8')
    # We can reuse testRunner if it supports XML output (e. g. by inheriting
    # from xml_reporter.TextAndXMLTestRunner). Otherwise we need to use
    # xml_reporter.TextAndXMLTestRunner.
    if (kwargs.get('testRunner') is not None
        and not hasattr(kwargs['testRunner'], 'set_default_xml_stream')):
      sys.stderr.write('WARNING: XML_OUTPUT_FILE or --xml_output_file setting '
                       'overrides testRunner=%r setting (possibly from --pdb)'
                       % (kwargs['testRunner']))
      # Passing a class object here allows TestProgram to initialize
      # instances based on its kwargs and/or parsed command-line args.
      kwargs['testRunner'] = xml_test_runner_class
    if kwargs.get('testRunner') is None:
      kwargs['testRunner'] = xml_test_runner_class
    kwargs['testRunner'].set_default_xml_stream(xml_output)

  # Make sure tmpdir exists.
  if not os.path.isdir(FLAGS.test_tmpdir):
    try:
      os.makedirs(FLAGS.test_tmpdir)
    except OSError as e:
      # Concurrent test might have created the directory.
      if e.errno != errno.EEXIST:
        raise

  # Let unittest.TestProgram.__init__ do its own argv parsing, e.g. for '-v',
  # on argv, which is sys.argv without the command-line flags.
  kwargs.setdefault('argv', argv)

  try:
    test_program = unittest.TestProgram(*args, **kwargs)
    return test_program.result
  finally:
    if xml_output:
      xml_output.close()


def run_tests(argv, args, kwargs):
  """Executes a set of Python unit tests.

  Most users should call absltest.main() instead of run_tests.

  Please note that run_tests should be called from app.run.
  Calling absltest.main() would ensure that.

  Please note that run_tests is allowed to make changes to kwargs.

  Args:
    argv: sys.argv with the command-line flags removed from the front, i.e. the
      argv with which app.run() has called __main__.main.
    args: Positional arguments passed through to unittest.TestProgram.__init__.
    kwargs: Keyword arguments passed through to unittest.TestProgram.__init__.
  """
  result = _run_and_get_tests_result(
      argv, args, kwargs, xml_reporter.TextAndXMLTestRunner)
  sys.exit(not result.wasSuccessful())
