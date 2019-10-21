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

"""Tests for absltest."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import io
import os
import re
import string
import subprocess
import sys
import tempfile

from absl import flags
from absl.testing import _bazelize_command
from absl.testing import absltest
from absl.testing import parameterized
import six

PY_VERSION_2 = sys.version_info[0] == 2

FLAGS = flags.FLAGS


class HelperMixin(object):

  def _get_helper_exec_path(self):
    helper = 'absl/testing/tests/absltest_test_helper'
    return _bazelize_command.get_executable_path(helper)

  def run_helper(self, test_id, args, env_overrides, expect_success):
    env = os.environ.copy()
    for key, value in six.iteritems(env_overrides):
      if value is None:
        if key in env:
          del env[key]
      else:
        env[key] = value

    command = [self._get_helper_exec_path(),
               '--test_id={}'.format(test_id)] + args
    process = subprocess.Popen(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env,
        universal_newlines=True)
    stdout, stderr = process.communicate()
    if expect_success:
      self.assertEqual(
          0, process.returncode,
          'Expected success, but failed with '
          'stdout:\n{}\nstderr:\n{}\n'.format(stdout, stderr))
    else:
      self.assertEqual(
          1, process.returncode,
          'Expected failure, but succeeded with '
          'stdout:\n{}\nstderr:\n{}\n'.format(stdout, stderr))
    return stdout, stderr


class TestCaseTest(absltest.TestCase, HelperMixin):
  longMessage = True

  def run_helper(self, test_id, args, env_overrides, expect_success):
    return super(TestCaseTest, self).run_helper(test_id, args + ['HelperTest'],
                                                env_overrides, expect_success)

  def test_flags_no_env_var_no_flags(self):
    self.run_helper(
        1,
        [],
        {'TEST_RANDOM_SEED': None,
         'TEST_SRCDIR': None,
         'TEST_TMPDIR': None,
        },
        expect_success=True)

  def test_flags_env_var_no_flags(self):
    tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    srcdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.run_helper(
        2,
        [],
        {'TEST_RANDOM_SEED': '321',
         'TEST_SRCDIR': srcdir,
         'TEST_TMPDIR': tmpdir,
         'ABSLTEST_TEST_HELPER_EXPECTED_TEST_SRCDIR': srcdir,
         'ABSLTEST_TEST_HELPER_EXPECTED_TEST_TMPDIR': tmpdir,
        },
        expect_success=True)

  def test_flags_no_env_var_flags(self):
    tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    srcdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.run_helper(
        3,
        ['--test_random_seed=123', '--test_srcdir={}'.format(srcdir),
         '--test_tmpdir={}'.format(tmpdir)],
        {'TEST_RANDOM_SEED': None,
         'TEST_SRCDIR': None,
         'TEST_TMPDIR': None,
         'ABSLTEST_TEST_HELPER_EXPECTED_TEST_SRCDIR': srcdir,
         'ABSLTEST_TEST_HELPER_EXPECTED_TEST_TMPDIR': tmpdir,
        },
        expect_success=True)

  def test_flags_env_var_flags(self):
    tmpdir_from_flag = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    srcdir_from_flag = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    tmpdir_from_env_var = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    srcdir_from_env_var = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.run_helper(
        4,
        ['--test_random_seed=221', '--test_srcdir={}'.format(srcdir_from_flag),
         '--test_tmpdir={}'.format(tmpdir_from_flag)],
        {'TEST_RANDOM_SEED': '123',
         'TEST_SRCDIR': srcdir_from_env_var,
         'TEST_TMPDIR': tmpdir_from_env_var,
         'ABSLTEST_TEST_HELPER_EXPECTED_TEST_SRCDIR': srcdir_from_flag,
         'ABSLTEST_TEST_HELPER_EXPECTED_TEST_TMPDIR': tmpdir_from_flag,
        },
        expect_success=True)

  def test_xml_output_file_from_xml_output_file_env(self):
    xml_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    xml_output_file_env = os.path.join(xml_dir, 'xml_output_file.xml')
    random_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.run_helper(
        6,
        [],
        {'XML_OUTPUT_FILE': xml_output_file_env,
         'RUNNING_UNDER_TEST_DAEMON': '1',
         'TEST_XMLOUTPUTDIR': random_dir,
         'ABSLTEST_TEST_HELPER_EXPECTED_XML_OUTPUT_FILE': xml_output_file_env,
        },
        expect_success=True)

  def test_xml_output_file_from_daemon(self):
    tmpdir = os.path.join(tempfile.mkdtemp(dir=FLAGS.test_tmpdir), 'sub_dir')
    random_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.run_helper(
        6,
        ['--test_tmpdir', tmpdir],
        {'XML_OUTPUT_FILE': None,
         'RUNNING_UNDER_TEST_DAEMON': '1',
         'TEST_XMLOUTPUTDIR': random_dir,
         'ABSLTEST_TEST_HELPER_EXPECTED_XML_OUTPUT_FILE': os.path.join(
             os.path.dirname(tmpdir), 'test_detail.xml'),
        },
        expect_success=True)

  def test_xml_output_file_from_test_xmloutputdir_env(self):
    xml_output_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    expected_xml_file = 'absltest_test_helper.xml'
    self.run_helper(
        6,
        [],
        {'XML_OUTPUT_FILE': None,
         'RUNNING_UNDER_TEST_DAEMON': None,
         'TEST_XMLOUTPUTDIR': xml_output_dir,
         'ABSLTEST_TEST_HELPER_EXPECTED_XML_OUTPUT_FILE': os.path.join(
             xml_output_dir, expected_xml_file),
        },
        expect_success=True)

  def test_xml_output_file_from_flag(self):
    random_dir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    flag_file = os.path.join(
        tempfile.mkdtemp(dir=FLAGS.test_tmpdir), 'output.xml')
    self.run_helper(
        6,
        ['--xml_output_file', flag_file],
        {'XML_OUTPUT_FILE': os.path.join(random_dir, 'output.xml'),
         'RUNNING_UNDER_TEST_DAEMON': '1',
         'TEST_XMLOUTPUTDIR': random_dir,
         'ABSLTEST_TEST_HELPER_EXPECTED_XML_OUTPUT_FILE': flag_file,
        },
        expect_success=True)

  def test_assert_in(self):
    animals = {'monkey': 'banana', 'cow': 'grass', 'seal': 'fish'}

    self.assertIn('a', 'abc')
    self.assertIn(2, [1, 2, 3])
    self.assertIn('monkey', animals)

    self.assertNotIn('d', 'abc')
    self.assertNotIn(0, [1, 2, 3])
    self.assertNotIn('otter', animals)

    self.assertRaises(AssertionError, self.assertIn, 'x', 'abc')
    self.assertRaises(AssertionError, self.assertIn, 4, [1, 2, 3])
    self.assertRaises(AssertionError, self.assertIn, 'elephant', animals)

    self.assertRaises(AssertionError, self.assertNotIn, 'c', 'abc')
    self.assertRaises(AssertionError, self.assertNotIn, 1, [1, 2, 3])
    self.assertRaises(AssertionError, self.assertNotIn, 'cow', animals)

  @absltest.expectedFailure
  def test_expected_failure(self):
    self.assertEqual(1, 2)  # the expected failure

  @absltest.expectedFailureIf(True, 'always true')
  def test_expected_failure_if(self):
    self.assertEqual(1, 2)  # the expected failure

  def test_expected_failure_success(self):
    _, stderr = self.run_helper(5, ['--', '-v'], {}, expect_success=False)
    self.assertRegex(stderr, r'FAILED \(.*unexpected successes=1\)')

  def test_assert_equal(self):
    self.assertListEqual([], [])
    self.assertTupleEqual((), ())
    self.assertSequenceEqual([], ())

    a = [0, 'a', []]
    b = []
    self.assertRaises(absltest.TestCase.failureException,
                      self.assertListEqual, a, b)
    self.assertRaises(absltest.TestCase.failureException,
                      self.assertListEqual, tuple(a), tuple(b))
    self.assertRaises(absltest.TestCase.failureException,
                      self.assertSequenceEqual, a, tuple(b))

    b.extend(a)
    self.assertListEqual(a, b)
    self.assertTupleEqual(tuple(a), tuple(b))
    self.assertSequenceEqual(a, tuple(b))
    self.assertSequenceEqual(tuple(a), b)

    self.assertRaises(AssertionError, self.assertListEqual, a, tuple(b))
    self.assertRaises(AssertionError, self.assertTupleEqual, tuple(a), b)
    self.assertRaises(AssertionError, self.assertListEqual, None, b)
    self.assertRaises(AssertionError, self.assertTupleEqual, None, tuple(b))
    self.assertRaises(AssertionError, self.assertSequenceEqual, None, tuple(b))
    self.assertRaises(AssertionError, self.assertListEqual, 1, 1)
    self.assertRaises(AssertionError, self.assertTupleEqual, 1, 1)
    self.assertRaises(AssertionError, self.assertSequenceEqual, 1, 1)

    self.assertSameElements([1, 2, 3], [3, 2, 1])
    self.assertSameElements([1, 2] + [3] * 100, [1] * 100 + [2, 3])
    self.assertSameElements(['foo', 'bar', 'baz'], ['bar', 'baz', 'foo'])
    self.assertRaises(AssertionError, self.assertSameElements, [10], [10, 11])
    self.assertRaises(AssertionError, self.assertSameElements, [10, 11], [10])

    # Test that sequences of unhashable objects can be tested for sameness:
    self.assertSameElements([[1, 2], [3, 4]], [[3, 4], [1, 2]])
    if PY_VERSION_2:
      # dict's are no longer valid for < comparison in Python 3 making them
      # unsortable (yay, sanity!).  But we need to preserve this old behavior
      # when running under Python 2.
      self.assertSameElements([{'a': 1}, {'b': 2}], [{'b': 2}, {'a': 1}])
    self.assertRaises(AssertionError, self.assertSameElements, [[1]], [[2]])

  def test_assert_items_equal_hotfix(self):
    """Confirm that http://bugs.python.org/issue14832 - b/10038517 is gone."""
    for assert_items_method in (self.assertItemsEqual, self.assertCountEqual):
      with self.assertRaises(self.failureException) as error_context:
        assert_items_method([4], [2])
      error_message = str(error_context.exception)
      # Confirm that the bug is either no longer present in Python or that our
      # assertItemsEqual patching version of the method in absltest.TestCase
      # doesn't get used.
      self.assertIn('First has 1, Second has 0:  4', error_message)
      self.assertIn('First has 0, Second has 1:  2', error_message)

  def test_assert_dict_equal(self):
    self.assertDictEqual({}, {})

    c = {'x': 1}
    d = {}
    self.assertRaises(absltest.TestCase.failureException,
                      self.assertDictEqual, c, d)

    d.update(c)
    self.assertDictEqual(c, d)

    d['x'] = 0
    self.assertRaises(absltest.TestCase.failureException,
                      self.assertDictEqual, c, d, 'These are unequal')

    self.assertRaises(AssertionError, self.assertDictEqual, None, d)
    self.assertRaises(AssertionError, self.assertDictEqual, [], d)
    self.assertRaises(AssertionError, self.assertDictEqual, 1, 1)

    try:
      # Ensure we use equality as the sole measure of elements, not type, since
      # that is consistent with dict equality.
      self.assertDictEqual({1: 1.0, 2: 2}, {1: 1, 2: 3})
    except AssertionError as e:
      self.assertMultiLineEqual('{1: 1.0, 2: 2} != {1: 1, 2: 3}\n'
                                'repr() of differing entries:\n2: 2 != 3\n',
                                str(e))

    try:
      self.assertDictEqual({}, {'x': 1})
    except AssertionError as e:
      self.assertMultiLineEqual("{} != {'x': 1}\n"
                                "Unexpected, but present entries:\n'x': 1\n",
                                str(e))
    else:
      self.fail('Expecting AssertionError')

    try:
      self.assertDictEqual({}, {'x': 1}, 'a message')
    except AssertionError as e:
      self.assertIn('a message', str(e))
    else:
      self.fail('Expecting AssertionError')

    expected = {'a': 1, 'b': 2, 'c': 3}
    seen = {'a': 2, 'c': 3, 'd': 4}
    try:
      self.assertDictEqual(expected, seen)
    except AssertionError as e:
      self.assertMultiLineEqual("""\
{'a': 1, 'b': 2, 'c': 3} != {'a': 2, 'c': 3, 'd': 4}
Unexpected, but present entries:
'd': 4

repr() of differing entries:
'a': 1 != 2

Missing entries:
'b': 2
""", str(e))
    else:
      self.fail('Expecting AssertionError')

    self.assertRaises(AssertionError, self.assertDictEqual, (1, 2), {})
    self.assertRaises(AssertionError, self.assertDictEqual, {}, (1, 2))

    # Ensure deterministic output of keys in dictionaries whose sort order
    # doesn't match the lexical ordering of repr -- this is most Python objects,
    # which are keyed by memory address.
    class Obj(object):

      def __init__(self, name):
        self.name = name

      def __repr__(self):
        return self.name

    try:
      self.assertDictEqual(
          {'a': Obj('A'), Obj('b'): Obj('B'), Obj('c'): Obj('C')},
          {'a': Obj('A'), Obj('d'): Obj('D'), Obj('e'): Obj('E')})
    except AssertionError as e:
      # Do as best we can not to be misleading when objects have the same repr
      # but aren't equal.
      err_str = str(e)
      self.assertStartsWith(err_str,
                            "{'a': A, b: B, c: C} != {'a': A, d: D, e: E}\n")
      self.assertRegex(
          err_str, r'(?ms).*^Unexpected, but present entries:\s+'
          r'^(d: D$\s+^e: E|e: E$\s+^d: D)$')
      self.assertRegex(
          err_str, r'(?ms).*^repr\(\) of differing entries:\s+'
          r'^.a.: A != A$', err_str)
      self.assertRegex(
          err_str, r'(?ms).*^Missing entries:\s+'
          r'^(b: B$\s+^c: C|c: C$\s+^b: B)$')
    else:
      self.fail('Expecting AssertionError')

    # Confirm that safe_repr, not repr, is being used.
    class RaisesOnRepr(object):

      def __repr__(self):
        return 1/0  # Intentionally broken __repr__ implementation.

    try:
      self.assertDictEqual(
          {RaisesOnRepr(): RaisesOnRepr()},
          {RaisesOnRepr(): RaisesOnRepr()}
          )
      self.fail('Expected dicts not to match')
    except AssertionError as e:
      # Depending on the testing environment, the object may get a __main__
      # prefix or a absltest_test prefix, so strip that for comparison.
      error_msg = re.sub(
          r'( at 0x[^>]+)|__main__\.|absltest_test\.', '', str(e))
      self.assertRegex(error_msg, """(?m)\
{<.*RaisesOnRepr object.*>: <.*RaisesOnRepr object.*>} != \
{<.*RaisesOnRepr object.*>: <.*RaisesOnRepr object.*>}
Unexpected, but present entries:
<.*RaisesOnRepr object.*>: <.*RaisesOnRepr object.*>

Missing entries:
<.*RaisesOnRepr object.*>: <.*RaisesOnRepr object.*>
""")

    # Confirm that safe_repr, not repr, is being used.
    class RaisesOnLt(object):

      def __lt__(self, unused_other):
        raise TypeError('Object is unordered.')

      def __repr__(self):
        return '<RaisesOnLt object>'

    try:
      self.assertDictEqual(
          {RaisesOnLt(): RaisesOnLt()},
          {RaisesOnLt(): RaisesOnLt()})
    except AssertionError as e:
      self.assertIn('Unexpected, but present entries:\n<RaisesOnLt', str(e))
      self.assertIn('Missing entries:\n<RaisesOnLt', str(e))

  def test_assert_set_equal(self):
    set1 = set()
    set2 = set()
    self.assertSetEqual(set1, set2)

    self.assertRaises(AssertionError, self.assertSetEqual, None, set2)
    self.assertRaises(AssertionError, self.assertSetEqual, [], set2)
    self.assertRaises(AssertionError, self.assertSetEqual, set1, None)
    self.assertRaises(AssertionError, self.assertSetEqual, set1, [])

    set1 = set(['a'])
    set2 = set()
    self.assertRaises(AssertionError, self.assertSetEqual, set1, set2)

    set1 = set(['a'])
    set2 = set(['a'])
    self.assertSetEqual(set1, set2)

    set1 = set(['a'])
    set2 = set(['a', 'b'])
    self.assertRaises(AssertionError, self.assertSetEqual, set1, set2)

    set1 = set(['a'])
    set2 = frozenset(['a', 'b'])
    self.assertRaises(AssertionError, self.assertSetEqual, set1, set2)

    set1 = set(['a', 'b'])
    set2 = frozenset(['a', 'b'])
    self.assertSetEqual(set1, set2)

    set1 = set()
    set2 = 'foo'
    self.assertRaises(AssertionError, self.assertSetEqual, set1, set2)
    self.assertRaises(AssertionError, self.assertSetEqual, set2, set1)

    # make sure any string formatting is tuple-safe
    set1 = set([(0, 1), (2, 3)])
    set2 = set([(4, 5)])
    self.assertRaises(AssertionError, self.assertSetEqual, set1, set2)

  def test_assert_dict_contains_subset(self):
    self.assertDictContainsSubset({}, {})

    self.assertDictContainsSubset({}, {'a': 1})

    self.assertDictContainsSubset({'a': 1}, {'a': 1})

    self.assertDictContainsSubset({'a': 1}, {'a': 1, 'b': 2})

    self.assertDictContainsSubset({'a': 1, 'b': 2}, {'a': 1, 'b': 2})

    self.assertRaises(absltest.TestCase.failureException,
                      self.assertDictContainsSubset, {'a': 2}, {'a': 1},
                      '.*Mismatched values:.*')

    self.assertRaises(absltest.TestCase.failureException,
                      self.assertDictContainsSubset, {'c': 1}, {'a': 1},
                      '.*Missing:.*')

    self.assertRaises(absltest.TestCase.failureException,
                      self.assertDictContainsSubset, {'a': 1, 'c': 1}, {'a': 1},
                      '.*Missing:.*')

    self.assertRaises(absltest.TestCase.failureException,
                      self.assertDictContainsSubset, {'a': 1, 'c': 1}, {'a': 1},
                      '.*Missing:.*Mismatched values:.*')

  def test_assert_sequence_almost_equal(self):
    actual = (1.1, 1.2, 1.4)

    # Test across sequence types.
    self.assertSequenceAlmostEqual((1.1, 1.2, 1.4), actual)
    self.assertSequenceAlmostEqual([1.1, 1.2, 1.4], actual)

    # Test sequence size mismatch.
    with self.assertRaises(AssertionError):
      self.assertSequenceAlmostEqual([1.1, 1.2], actual)
    with self.assertRaises(AssertionError):
      self.assertSequenceAlmostEqual([1.1, 1.2, 1.4, 1.5], actual)

    # Test delta.
    with self.assertRaises(AssertionError):
      self.assertSequenceAlmostEqual((1.15, 1.15, 1.4), actual)
    self.assertSequenceAlmostEqual((1.15, 1.15, 1.4), actual, delta=0.1)

    # Test places.
    with self.assertRaises(AssertionError):
      self.assertSequenceAlmostEqual((1.1001, 1.2001, 1.3999), actual)
    self.assertSequenceAlmostEqual((1.1001, 1.2001, 1.3999), actual, places=3)

  def test_assert_contains_subset(self):
    # sets, lists, tuples, dicts all ok.  Types of set and subset do not have to
    # match.
    actual = ('a', 'b', 'c')
    self.assertContainsSubset({'a', 'b'}, actual)
    self.assertContainsSubset(('b', 'c'), actual)
    self.assertContainsSubset({'b': 1, 'c': 2}, list(actual))
    self.assertContainsSubset(['c', 'a'], set(actual))
    self.assertContainsSubset([], set())
    self.assertContainsSubset([], {'a': 1})

    self.assertRaises(AssertionError, self.assertContainsSubset, ('d',), actual)
    self.assertRaises(AssertionError, self.assertContainsSubset, ['d'],
                      set(actual))
    self.assertRaises(AssertionError, self.assertContainsSubset, {'a': 1}, [])

    with self.assertRaisesRegex(AssertionError, 'Missing elements'):
      self.assertContainsSubset({1, 2, 3}, {1, 2})

    with self.assertRaisesRegex(
        AssertionError,
        re.compile('Missing elements .* Custom message', re.DOTALL)):
      self.assertContainsSubset({1, 2}, {1}, 'Custom message')

  def test_assert_no_common_elements(self):
    actual = ('a', 'b', 'c')
    self.assertNoCommonElements((), actual)
    self.assertNoCommonElements(('d', 'e'), actual)
    self.assertNoCommonElements({'d', 'e'}, actual)

    with self.assertRaisesRegex(
        AssertionError,
        re.compile('Common elements .* Custom message', re.DOTALL)):
      self.assertNoCommonElements({1, 2}, {1}, 'Custom message')

    with self.assertRaises(AssertionError):
      self.assertNoCommonElements(['a'], actual)

    with self.assertRaises(AssertionError):
      self.assertNoCommonElements({'a', 'b', 'c'}, actual)

    with self.assertRaises(AssertionError):
      self.assertNoCommonElements({'b', 'c'}, set(actual))

  def test_assert_almost_equal(self):
    self.assertAlmostEqual(1.00000001, 1.0)
    self.assertNotAlmostEqual(1.0000001, 1.0)

  def test_assert_almost_equals_with_delta(self):
    self.assertAlmostEquals(3.14, 3, delta=0.2)
    self.assertAlmostEquals(2.81, 3.14, delta=1)
    self.assertAlmostEquals(-1, 1, delta=3)
    self.assertRaises(AssertionError, self.assertAlmostEquals,
                      3.14, 2.81, delta=0.1)
    self.assertRaises(AssertionError, self.assertAlmostEquals,
                      1, 2, delta=0.5)
    self.assertNotAlmostEquals(3.14, 2.81, delta=0.1)

  def test_assert_starts_with(self):
    self.assertStartsWith('foobar', 'foo')
    self.assertStartsWith('foobar', 'foobar')
    msg = 'This is a useful message'
    whole_msg = "'foobar' does not start with 'bar' : This is a useful message"
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertStartsWith,
                                      'foobar', 'bar', msg)
    self.assertRaises(AssertionError, self.assertStartsWith, 'foobar', 'blah')

  def test_assert_not_starts_with(self):
    self.assertNotStartsWith('foobar', 'bar')
    self.assertNotStartsWith('foobar', 'blah')
    msg = 'This is a useful message'
    whole_msg = "'foobar' does start with 'foo' : This is a useful message"
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertNotStartsWith,
                                      'foobar', 'foo', msg)
    self.assertRaises(AssertionError, self.assertNotStartsWith, 'foobar',
                      'foobar')

  def test_assert_ends_with(self):
    self.assertEndsWith('foobar', 'bar')
    self.assertEndsWith('foobar', 'foobar')
    msg = 'This is a useful message'
    whole_msg = "'foobar' does not end with 'foo' : This is a useful message"
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertEndsWith,
                                      'foobar', 'foo', msg)
    self.assertRaises(AssertionError, self.assertEndsWith, 'foobar', 'blah')

  def test_assert_not_ends_with(self):
    self.assertNotEndsWith('foobar', 'foo')
    self.assertNotEndsWith('foobar', 'blah')
    msg = 'This is a useful message'
    whole_msg = "'foobar' does end with 'bar' : This is a useful message"
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertNotEndsWith,
                                      'foobar', 'bar', msg)
    self.assertRaises(AssertionError, self.assertNotEndsWith, 'foobar',
                      'foobar')

  def test_assert_regex_backports(self):
    self.assertRegex('regex', 'regex')
    self.assertNotRegex('not-regex', 'no-match')
    with self.assertRaisesRegex(ValueError, 'pattern'):
      raise ValueError('pattern')

  def test_assert_regex_match_matches(self):
    self.assertRegexMatch('str', ['str'])

  def test_assert_regex_match_matches_substring(self):
    self.assertRegexMatch('pre-str-post', ['str'])

  def test_assert_regex_match_multiple_regex_matches(self):
    self.assertRegexMatch('str', ['rts', 'str'])

  def test_assert_regex_match_empty_list_fails(self):
    expected_re = re.compile(r'No regexes specified\.', re.MULTILINE)

    with self.assertRaisesRegex(AssertionError, expected_re):
      self.assertRegexMatch('str', regexes=[])

  def test_assert_regex_match_bad_arguments(self):
    with self.assertRaisesRegex(AssertionError,
                                'regexes is string or bytes;.*'):
      self.assertRegexMatch('1.*2', '1 2')

  def test_assert_regex_match_unicode_vs_bytes(self):
    """Ensure proper utf-8 encoding or decoding happens automatically."""
    self.assertRegexMatch(u'str', [b'str'])
    self.assertRegexMatch(b'str', [u'str'])

  def test_assert_regex_match_unicode(self):
    self.assertRegexMatch(u'foo str', [u'str'])

  def test_assert_regex_match_bytes(self):
    self.assertRegexMatch(b'foo str', [b'str'])

  def test_assert_regex_match_all_the_same_type(self):
    with self.assertRaisesRegex(AssertionError, 'regexes .* same type'):
      self.assertRegexMatch('foo str', [b'str', u'foo'])

  def test_assert_command_fails_stderr(self):
    tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.assertCommandFails(
        ['cat', os.path.join(tmpdir, 'file.txt')],
        ['No such file or directory'])

  def test_assert_command_fails_with_list_of_string(self):
    self.assertCommandFails(['false'], [''])

  def test_assert_command_fails_with_list_of_unicode_string(self):
    self.assertCommandFails([u'false'], [''])

  def test_assert_command_fails_with_unicode_string(self):
    self.assertCommandFails(u'false', [u''])

  def test_assert_command_fails_with_unicode_string_bytes_regex(self):
    self.assertCommandFails(u'false', [b''])

  def test_assert_command_fails_with_message(self):
    msg = 'This is a useful message'
    expected_re = re.compile('The following command succeeded while expected to'
                             ' fail:.* This is a useful message', re.DOTALL)

    with self.assertRaisesRegex(AssertionError, expected_re):
      self.assertCommandFails([u'true'], [''], msg=msg)

  def test_assert_command_succeeds_stderr(self):
    expected_re = re.compile('No such file or directory')
    tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)

    with self.assertRaisesRegex(AssertionError, expected_re):
      self.assertCommandSucceeds(['cat', os.path.join(tmpdir, 'file.txt')])

  def test_assert_command_succeeds_with_matching_unicode_regexes(self):
    self.assertCommandSucceeds(['echo', 'SUCCESS'], regexes=[u'SUCCESS'])

  def test_assert_command_succeeds_with_matching_bytes_regexes(self):
    self.assertCommandSucceeds(['echo', 'SUCCESS'], regexes=[b'SUCCESS'])

  def test_assert_command_succeeds_with_non_matching_regexes(self):
    expected_re = re.compile('Running command.* This is a useful message',
                             re.DOTALL)
    msg = 'This is a useful message'

    with self.assertRaisesRegex(AssertionError, expected_re):
      self.assertCommandSucceeds(['echo', 'FAIL'], regexes=['SUCCESS'], msg=msg)

  def test_assert_command_succeeds_with_list_of_string(self):
    self.assertCommandSucceeds(['true'])

  def test_assert_command_succeeds_with_list_of_unicode_string(self):
    self.assertCommandSucceeds([u'true'])

  def test_assert_command_succeeds_with_unicode_string(self):
    # This uses shell=True. On Windows Bazel, it requires environment
    # variables from the current process otherwise it can't find msys 'true'.
    self.assertCommandSucceeds(u'true', env=dict(os.environ))

  def test_inequality(self):
    # Try ints
    self.assertGreater(2, 1)
    self.assertGreaterEqual(2, 1)
    self.assertGreaterEqual(1, 1)
    self.assertLess(1, 2)
    self.assertLessEqual(1, 2)
    self.assertLessEqual(1, 1)
    self.assertRaises(AssertionError, self.assertGreater, 1, 2)
    self.assertRaises(AssertionError, self.assertGreater, 1, 1)
    self.assertRaises(AssertionError, self.assertGreaterEqual, 1, 2)
    self.assertRaises(AssertionError, self.assertLess, 2, 1)
    self.assertRaises(AssertionError, self.assertLess, 1, 1)
    self.assertRaises(AssertionError, self.assertLessEqual, 2, 1)

    # Try Floats
    self.assertGreater(1.1, 1.0)
    self.assertGreaterEqual(1.1, 1.0)
    self.assertGreaterEqual(1.0, 1.0)
    self.assertLess(1.0, 1.1)
    self.assertLessEqual(1.0, 1.1)
    self.assertLessEqual(1.0, 1.0)
    self.assertRaises(AssertionError, self.assertGreater, 1.0, 1.1)
    self.assertRaises(AssertionError, self.assertGreater, 1.0, 1.0)
    self.assertRaises(AssertionError, self.assertGreaterEqual, 1.0, 1.1)
    self.assertRaises(AssertionError, self.assertLess, 1.1, 1.0)
    self.assertRaises(AssertionError, self.assertLess, 1.0, 1.0)
    self.assertRaises(AssertionError, self.assertLessEqual, 1.1, 1.0)

    # Try Strings
    self.assertGreater('bug', 'ant')
    self.assertGreaterEqual('bug', 'ant')
    self.assertGreaterEqual('ant', 'ant')
    self.assertLess('ant', 'bug')
    self.assertLessEqual('ant', 'bug')
    self.assertLessEqual('ant', 'ant')
    self.assertRaises(AssertionError, self.assertGreater, 'ant', 'bug')
    self.assertRaises(AssertionError, self.assertGreater, 'ant', 'ant')
    self.assertRaises(AssertionError, self.assertGreaterEqual, 'ant', 'bug')
    self.assertRaises(AssertionError, self.assertLess, 'bug', 'ant')
    self.assertRaises(AssertionError, self.assertLess, 'ant', 'ant')
    self.assertRaises(AssertionError, self.assertLessEqual, 'bug', 'ant')

    # Try Unicode
    self.assertGreater(u'bug', u'ant')
    self.assertGreaterEqual(u'bug', u'ant')
    self.assertGreaterEqual(u'ant', u'ant')
    self.assertLess(u'ant', u'bug')
    self.assertLessEqual(u'ant', u'bug')
    self.assertLessEqual(u'ant', u'ant')
    self.assertRaises(AssertionError, self.assertGreater, u'ant', u'bug')
    self.assertRaises(AssertionError, self.assertGreater, u'ant', u'ant')
    self.assertRaises(AssertionError, self.assertGreaterEqual, u'ant', u'bug')
    self.assertRaises(AssertionError, self.assertLess, u'bug', u'ant')
    self.assertRaises(AssertionError, self.assertLess, u'ant', u'ant')
    self.assertRaises(AssertionError, self.assertLessEqual, u'bug', u'ant')

    # Try Mixed String/Unicode
    self.assertGreater('bug', u'ant')
    self.assertGreater(u'bug', 'ant')
    self.assertGreaterEqual('bug', u'ant')
    self.assertGreaterEqual(u'bug', 'ant')
    self.assertGreaterEqual('ant', u'ant')
    self.assertGreaterEqual(u'ant', 'ant')
    self.assertLess('ant', u'bug')
    self.assertLess(u'ant', 'bug')
    self.assertLessEqual('ant', u'bug')
    self.assertLessEqual(u'ant', 'bug')
    self.assertLessEqual('ant', u'ant')
    self.assertLessEqual(u'ant', 'ant')
    self.assertRaises(AssertionError, self.assertGreater, 'ant', u'bug')
    self.assertRaises(AssertionError, self.assertGreater, u'ant', 'bug')
    self.assertRaises(AssertionError, self.assertGreater, 'ant', u'ant')
    self.assertRaises(AssertionError, self.assertGreater, u'ant', 'ant')
    self.assertRaises(AssertionError, self.assertGreaterEqual, 'ant', u'bug')
    self.assertRaises(AssertionError, self.assertGreaterEqual, u'ant', 'bug')
    self.assertRaises(AssertionError, self.assertLess, 'bug', u'ant')
    self.assertRaises(AssertionError, self.assertLess, u'bug', 'ant')
    self.assertRaises(AssertionError, self.assertLess, 'ant', u'ant')
    self.assertRaises(AssertionError, self.assertLess, u'ant', 'ant')
    self.assertRaises(AssertionError, self.assertLessEqual, 'bug', u'ant')
    self.assertRaises(AssertionError, self.assertLessEqual, u'bug', 'ant')

  def test_assert_multi_line_equal(self):
    sample_text = """\
http://www.python.org/doc/2.3/lib/module-unittest.html
test case
    A test case is the smallest unit of testing. [...]
"""
    revised_sample_text = """\
http://www.python.org/doc/2.4.1/lib/module-unittest.html
test case
    A test case is the smallest unit of testing. [...] You may provide your
    own implementation that does not subclass from TestCase, of course.
"""
    sample_text_error = """
- http://www.python.org/doc/2.3/lib/module-unittest.html
?                             ^
+ http://www.python.org/doc/2.4.1/lib/module-unittest.html
?                             ^^^
  test case
-     A test case is the smallest unit of testing. [...]
+     A test case is the smallest unit of testing. [...] You may provide your
?                                                       +++++++++++++++++++++
+     own implementation that does not subclass from TestCase, of course.
"""
    types = (str, unicode) if PY_VERSION_2 else (str,)

    for type1 in types:
      for type2 in types:
        self.assertRaisesWithLiteralMatch(AssertionError, sample_text_error,
                                          self.assertMultiLineEqual,
                                          type1(sample_text),
                                          type2(revised_sample_text))

    self.assertRaises(AssertionError, self.assertMultiLineEqual, (1, 2), 'str')
    self.assertRaises(AssertionError, self.assertMultiLineEqual, 'str', (1, 2))

  def test_assert_multi_line_equal_adds_newlines_if_needed(self):
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        '\n'
        '  line1\n'
        '- line2\n'
        '?     ^\n'
        '+ line3\n'
        '?     ^\n',
        self.assertMultiLineEqual,
        'line1\n'
        'line2',
        'line1\n'
        'line3')

  def test_assert_multi_line_equal_shows_missing_newlines(self):
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        '\n'
        '  line1\n'
        '- line2\n'
        '?      -\n'
        '+ line2\n',
        self.assertMultiLineEqual,
        'line1\n'
        'line2\n',
        'line1\n'
        'line2')

  def test_assert_multi_line_equal_shows_extra_newlines(self):
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        '\n'
        '  line1\n'
        '- line2\n'
        '+ line2\n'
        '?      +\n',
        self.assertMultiLineEqual,
        'line1\n'
        'line2',
        'line1\n'
        'line2\n')

  def test_assert_multi_line_equal_line_limit_limits(self):
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        '\n'
        '  line1\n'
        '(... and 4 more delta lines omitted for brevity.)\n',
        self.assertMultiLineEqual,
        'line1\n'
        'line2\n',
        'line1\n'
        'line3\n',
        line_limit=1)

  def test_assert_multi_line_equal_line_limit_limits_with_message(self):
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        'Prefix:\n'
        '  line1\n'
        '(... and 4 more delta lines omitted for brevity.)\n',
        self.assertMultiLineEqual,
        'line1\n'
        'line2\n',
        'line1\n'
        'line3\n',
        'Prefix',
        line_limit=1)

  def test_assert_is_none(self):
    self.assertIsNone(None)
    self.assertRaises(AssertionError, self.assertIsNone, False)
    self.assertIsNotNone('Google')
    self.assertRaises(AssertionError, self.assertIsNotNone, None)
    self.assertRaises(AssertionError, self.assertIsNone, (1, 2))

  def test_assert_is(self):
    self.assertIs(object, object)
    self.assertRaises(AssertionError, self.assertIsNot, object, object)
    self.assertIsNot(True, False)
    self.assertRaises(AssertionError, self.assertIs, True, False)

  def test_assert_between(self):
    self.assertBetween(3.14, 3.1, 3.141)
    self.assertBetween(4, 4, 1e10000)
    self.assertBetween(9.5, 9.4, 9.5)
    self.assertBetween(-1e10, -1e10000, 0)
    self.assertRaises(AssertionError, self.assertBetween, 9.4, 9.3, 9.3999)
    self.assertRaises(AssertionError, self.assertBetween, -1e10000, -1e10, 0)

  def test_assert_raises_with_predicate_match_no_raise(self):
    with self.assertRaisesRegex(AssertionError, '^Exception not raised$'):
      self.assertRaisesWithPredicateMatch(Exception,
                                          lambda e: True,
                                          lambda: 1)  # don't raise

    with self.assertRaisesRegex(AssertionError, '^Exception not raised$'):
      with self.assertRaisesWithPredicateMatch(Exception, lambda e: True):
        pass  # don't raise

  def test_assert_raises_with_predicate_match_raises_wrong_exception(self):
    def _raise_value_error():
      raise ValueError

    with self.assertRaises(ValueError):
      self.assertRaisesWithPredicateMatch(IOError,
                                          lambda e: True,
                                          _raise_value_error)

    with self.assertRaises(ValueError):
      with self.assertRaisesWithPredicateMatch(IOError, lambda e: True):
        raise ValueError

  def test_assert_raises_with_predicate_match_predicate_fails(self):
    def _raise_value_error():
      raise ValueError
    with self.assertRaisesRegex(AssertionError, ' does not match predicate '):
      self.assertRaisesWithPredicateMatch(ValueError,
                                          lambda e: False,
                                          _raise_value_error)

    with self.assertRaisesRegex(AssertionError, ' does not match predicate '):
      with self.assertRaisesWithPredicateMatch(ValueError, lambda e: False):
        raise ValueError

  def test_assert_raises_with_predicate_match_predicate_passes(self):
    def _raise_value_error():
      raise ValueError

    self.assertRaisesWithPredicateMatch(ValueError,
                                        lambda e: True,
                                        _raise_value_error)

    with self.assertRaisesWithPredicateMatch(ValueError, lambda e: True):
      raise ValueError

  def test_assert_contains_in_order(self):
    # Valids
    self.assertContainsInOrder(
        ['fox', 'dog'], 'The quick brown fox jumped over the lazy dog.')
    self.assertContainsInOrder(
        ['quick', 'fox', 'dog'],
        'The quick brown fox jumped over the lazy dog.')
    self.assertContainsInOrder(
        ['The', 'fox', 'dog.'], 'The quick brown fox jumped over the lazy dog.')
    self.assertContainsInOrder(
        ['fox'], 'The quick brown fox jumped over the lazy dog.')
    self.assertContainsInOrder(
        'fox', 'The quick brown fox jumped over the lazy dog.')
    self.assertContainsInOrder(
        ['fox', 'dog'], 'fox dog fox')
    self.assertContainsInOrder(
        [], 'The quick brown fox jumped over the lazy dog.')
    self.assertContainsInOrder(
        [], '')

    # Invalids
    msg = 'This is a useful message'
    whole_msg = ("Did not find 'fox' after 'dog' in 'The quick brown fox"
                 " jumped over the lazy dog' : This is a useful message")
    self.assertRaisesWithLiteralMatch(
        AssertionError, whole_msg, self.assertContainsInOrder,
        ['dog', 'fox'], 'The quick brown fox jumped over the lazy dog', msg=msg)
    self.assertRaises(
        AssertionError, self.assertContainsInOrder,
        ['The', 'dog', 'fox'], 'The quick brown fox jumped over the lazy dog')
    self.assertRaises(
        AssertionError, self.assertContainsInOrder, ['dog'], '')

  def test_assert_contains_subsequence_for_numbers(self):
    self.assertContainsSubsequence([1, 2, 3], [1])
    self.assertContainsSubsequence([1, 2, 3], [1, 2])
    self.assertContainsSubsequence([1, 2, 3], [1, 3])

    with self.assertRaises(AssertionError):
      self.assertContainsSubsequence([1, 2, 3], [4])
    msg = 'This is a useful message'
    whole_msg = ('[3, 1] not a subsequence of [1, 2, 3]. '
                 'First non-matching element: 1 : This is a useful message')
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertContainsSubsequence,
                                      [1, 2, 3], [3, 1], msg=msg)

  def test_assert_contains_subsequence_for_strings(self):
    self.assertContainsSubsequence(['foo', 'bar', 'blorp'], ['foo', 'blorp'])
    with self.assertRaises(AssertionError):
      self.assertContainsSubsequence(
          ['foo', 'bar', 'blorp'], ['blorp', 'foo'])

  def test_assert_contains_subsequence_with_empty_subsequence(self):
    self.assertContainsSubsequence([1, 2, 3], [])
    self.assertContainsSubsequence(['foo', 'bar', 'blorp'], [])
    self.assertContainsSubsequence([], [])

  def test_assert_contains_subsequence_with_empty_container(self):
    with self.assertRaises(AssertionError):
      self.assertContainsSubsequence([], [1])
    with self.assertRaises(AssertionError):
      self.assertContainsSubsequence([], ['foo'])

  def test_assert_contains_exact_subsequence_for_numbers(self):
    self.assertContainsExactSubsequence([1, 2, 3], [1])
    self.assertContainsExactSubsequence([1, 2, 3], [1, 2])
    self.assertContainsExactSubsequence([1, 2, 3], [2, 3])

    with self.assertRaises(AssertionError):
      self.assertContainsExactSubsequence([1, 2, 3], [4])
    msg = 'This is a useful message'
    whole_msg = ('[1, 2, 4] not an exact subsequence of [1, 2, 3, 4]. '
                 'Longest matching prefix: [1, 2] : This is a useful message')
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertContainsExactSubsequence,
                                      [1, 2, 3, 4], [1, 2, 4], msg=msg)

  def test_assert_contains_exact_subsequence_for_strings(self):
    self.assertContainsExactSubsequence(
        ['foo', 'bar', 'blorp'], ['foo', 'bar'])
    with self.assertRaises(AssertionError):
      self.assertContainsExactSubsequence(
          ['foo', 'bar', 'blorp'], ['blorp', 'foo'])

  def test_assert_contains_exact_subsequence_with_empty_subsequence(self):
    self.assertContainsExactSubsequence([1, 2, 3], [])
    self.assertContainsExactSubsequence(['foo', 'bar', 'blorp'], [])
    self.assertContainsExactSubsequence([], [])

  def test_assert_contains_exact_subsequence_with_empty_container(self):
    with self.assertRaises(AssertionError):
      self.assertContainsExactSubsequence([], [3])
    with self.assertRaises(AssertionError):
      self.assertContainsExactSubsequence([], ['foo', 'bar'])
    self.assertContainsExactSubsequence([], [])

  def test_assert_totally_ordered(self):
    # Valid.
    self.assertTotallyOrdered()
    self.assertTotallyOrdered([1])
    self.assertTotallyOrdered([1], [2])
    self.assertTotallyOrdered([1, 1, 1])
    self.assertTotallyOrdered([(1, 1)], [(1, 2)], [(2, 1)])
    if PY_VERSION_2:
      # In Python 3 comparing different types of elements is not supported.
      self.assertTotallyOrdered([None], [1], [2])
      self.assertTotallyOrdered([1, 1, 1], ['a string'])

    # From the docstring.
    class A(object):

      def __init__(self, x, y):
        self.x = x
        self.y = y

      def __hash__(self):
        return hash(self.x)

      def __repr__(self):
        return 'A(%r, %r)' % (self.x, self.y)

      def __eq__(self, other):
        try:
          return self.x == other.x
        except AttributeError:
          return NotImplemented

      def __ne__(self, other):
        try:
          return self.x != other.x
        except AttributeError:
          return NotImplemented

      def __lt__(self, other):
        try:
          return self.x < other.x
        except AttributeError:
          return NotImplemented

      def __le__(self, other):
        try:
          return self.x <= other.x
        except AttributeError:
          return NotImplemented

      def __gt__(self, other):
        try:
          return self.x > other.x
        except AttributeError:
          return NotImplemented

      def __ge__(self, other):
        try:
          return self.x >= other.x
        except AttributeError:
          return NotImplemented

    class B(A):
      """Like A, but not hashable."""
      __hash__ = None

    if PY_VERSION_2:
      self.assertTotallyOrdered(
          [None],  # None should come before everything else.
          [1],  # Integers sort earlier.
          [A(1, 'a')],
          [A(2, 'b')],  # 2 is after 1.
          [
              A(3, 'c'),
              B(3, 'd'),
              B(3, 'e')  # The second argument is irrelevant.
          ],
          [A(4, 'z')],
          ['foo'])  # Strings sort last.
    else:
      # Python 3 does not define ordering across different types.
      self.assertTotallyOrdered(
          [A(1, 'a')],
          [A(2, 'b')],  # 2 is after 1.
          [
              A(3, 'c'),
              B(3, 'd'),
              B(3, 'e')  # The second argument is irrelevant.
          ],
          [A(4, 'z')])

    # Invalid.
    msg = 'This is a useful message'
    whole_msg = '2 not less than 1 : This is a useful message'
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertTotallyOrdered, [2], [1],
                                      msg=msg)
    self.assertRaises(AssertionError, self.assertTotallyOrdered, [2], [1])
    self.assertRaises(AssertionError, self.assertTotallyOrdered, [2], [1], [3])
    self.assertRaises(AssertionError, self.assertTotallyOrdered, [1, 2])

  def test_short_description_without_docstring(self):
    self.assertEquals(
        self.shortDescription(),
        ('test_short_description_without_docstring '
         '(%s.TestCaseTest)' % __name__))

  def test_short_description_with_one_line_docstring(self):
    """Tests shortDescription() for a method with a docstring."""
    self.assertEquals(
        self.shortDescription(),
        ('test_short_description_with_one_line_docstring '
         '(%s.TestCaseTest)\n'
         'Tests shortDescription() for a method with a docstring.' % __name__))

  def test_short_description_with_multi_line_docstring(self):
    """Tests shortDescription() for a method with a longer docstring.

    This method ensures that only the first line of a docstring is
    returned used in the short description, no matter how long the
    whole thing is.
    """
    self.assertEquals(
        self.shortDescription(),
        ('test_short_description_with_multi_line_docstring '
         '(%s.TestCaseTest)\n'
         'Tests shortDescription() for a method with a longer docstring.'
         % __name__))

  def test_assert_url_equal_same(self):
    self.assertUrlEqual('http://a', 'http://a')
    self.assertUrlEqual('http://a/path/test', 'http://a/path/test')
    self.assertUrlEqual('#fragment', '#fragment')
    self.assertUrlEqual('http://a/?q=1', 'http://a/?q=1')
    self.assertUrlEqual('http://a/?q=1&v=5', 'http://a/?v=5&q=1')
    self.assertUrlEqual('/logs?v=1&a=2&t=labels&f=path%3A%22foo%22',
                        '/logs?a=2&f=path%3A%22foo%22&v=1&t=labels')
    self.assertUrlEqual('http://a/path;p1', 'http://a/path;p1')
    self.assertUrlEqual('http://a/path;p2;p3;p1', 'http://a/path;p1;p2;p3')
    self.assertUrlEqual('sip:alice@atlanta.com;maddr=239.255.255.1;ttl=15',
                        'sip:alice@atlanta.com;ttl=15;maddr=239.255.255.1')
    self.assertUrlEqual('http://nyan/cat?p=1&b=', 'http://nyan/cat?b=&p=1')

  def test_assert_url_equal_different(self):
    msg = 'This is a useful message'
    if PY_VERSION_2:
      whole_msg = "'a' != 'b' : This is a useful message"
    else:
      whole_msg = 'This is a useful message:\n- a\n+ b\n'
    self.assertRaisesWithLiteralMatch(AssertionError, whole_msg,
                                      self.assertUrlEqual,
                                      'http://a', 'http://b', msg=msg)
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://a/x', 'http://a:8080/x')
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://a/x', 'http://a/y')
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://a/?q=2', 'http://a/?q=1')
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://a/?q=1&v=5', 'http://a/?v=2&q=1')
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://a', 'sip://b')
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://a#g', 'sip://a#f')
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://a/path;p1;p3;p1', 'http://a/path;p1;p2;p3')
    self.assertRaises(AssertionError, self.assertUrlEqual,
                      'http://nyan/cat?p=1&b=', 'http://nyan/cat?p=1')

  def test_same_structure_same(self):
    self.assertSameStructure(0, 0)
    self.assertSameStructure(1, 1)
    self.assertSameStructure('', '')
    self.assertSameStructure('hello', 'hello', msg='This Should not fail')
    self.assertSameStructure(set(), set())
    self.assertSameStructure(set([1, 2]), set([1, 2]))
    self.assertSameStructure(set(), frozenset())
    self.assertSameStructure(set([1, 2]), frozenset([1, 2]))
    self.assertSameStructure([], [])
    self.assertSameStructure(['a'], ['a'])
    self.assertSameStructure([], ())
    self.assertSameStructure(['a'], ('a',))
    self.assertSameStructure({}, {})
    self.assertSameStructure({'one': 1}, {'one': 1})
    self.assertSameStructure(collections.defaultdict(None, {'one': 1}),
                             {'one': 1})
    self.assertSameStructure(collections.OrderedDict({'one': 1}),
                             collections.defaultdict(None, {'one': 1}))
    # int and long should always be treated as the same type.
    if PY_VERSION_2:
      self.assertSameStructure({long(3): 3}, {3: long(3)})

  def test_same_structure_different(self):
    # Different type
    with self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'int'> but b is a <(type|class) 'str'>"):
      self.assertSameStructure(0, 'hello')
    with self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'int'> but b is a <(type|class) 'list'>"):
      self.assertSameStructure(0, [])
    with self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'int'> but b is a <(type|class) 'float'>"):
      self.assertSameStructure(2, 2.0)

    with self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'list'> but b is a <(type|class) 'dict'>"):
      self.assertSameStructure([], {})

    with self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'list'> but b is a <(type|class) 'set'>"):
      self.assertSameStructure([], set())

    with self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'dict'> but b is a <(type|class) 'set'>"):
      self.assertSameStructure({}, set())

    # Different scalar values
    self.assertRaisesWithLiteralMatch(
        AssertionError, 'a is 0 but b is 1',
        self.assertSameStructure, 0, 1)
    self.assertRaisesWithLiteralMatch(
        AssertionError, "a is 'hello' but b is 'goodbye' : This was expected",
        self.assertSameStructure, 'hello', 'goodbye', msg='This was expected')

    # Different sets
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        r'AA has 2 but BB does not',
        self.assertSameStructure,
        set([1, 2]),
        set([1]),
        aname='AA',
        bname='BB')
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        r'AA lacks 2 but BB has it',
        self.assertSameStructure,
        set([1]),
        set([1, 2]),
        aname='AA',
        bname='BB')

    # Different lists
    self.assertRaisesWithLiteralMatch(
        AssertionError, "a has [2] with value 'z' but b does not",
        self.assertSameStructure, ['x', 'y', 'z'], ['x', 'y'])
    self.assertRaisesWithLiteralMatch(
        AssertionError, "a lacks [2] but b has it with value 'z'",
        self.assertSameStructure, ['x', 'y'], ['x', 'y', 'z'])
    self.assertRaisesWithLiteralMatch(
        AssertionError, "a[2] is 'z' but b[2] is 'Z'",
        self.assertSameStructure, ['x', 'y', 'z'], ['x', 'y', 'Z'])

    # Different dicts
    self.assertRaisesWithLiteralMatch(
        AssertionError, "a has ['two'] with value 2 but it's missing in b",
        self.assertSameStructure, {'one': 1, 'two': 2}, {'one': 1})
    self.assertRaisesWithLiteralMatch(
        AssertionError, "a lacks ['two'] but b has it with value 2",
        self.assertSameStructure, {'one': 1}, {'one': 1, 'two': 2})
    self.assertRaisesWithLiteralMatch(
        AssertionError, "a['two'] is 2 but b['two'] is 3",
        self.assertSameStructure, {'one': 1, 'two': 2}, {'one': 1, 'two': 3})

    # String and byte types should not be considered equivalent to other
    # sequences
    self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'list'> but b is a <(type|class) 'str'>",
        self.assertSameStructure, [], '')
    self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'str'> but b is a <(type|class) 'tuple'>",
        self.assertSameStructure, '', ())
    self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'list'> but b is a <(type|class) 'str'>",
        self.assertSameStructure, ['a', 'b', 'c'], 'abc')
    self.assertRaisesRegex(
        AssertionError,
        r"a is a <(type|class) 'str'> but b is a <(type|class) 'tuple'>",
        self.assertSameStructure, 'abc', ('a', 'b', 'c'))

    # Deep key generation
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        "a[0][0]['x']['y']['z'][0] is 1 but b[0][0]['x']['y']['z'][0] is 2",
        self.assertSameStructure,
        [[{'x': {'y': {'z': [1]}}}]], [[{'x': {'y': {'z': [2]}}}]])

    # Multiple problems
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        'a[0] is 1 but b[0] is 3; a[1] is 2 but b[1] is 4',
        self.assertSameStructure, [1, 2], [3, 4])
    with self.assertRaisesRegex(
        AssertionError,
        re.compile(r"^a\[0] is 'a' but b\[0] is 'A'; .*"
                   r"a\[18] is 's' but b\[18] is 'S'; \.\.\.$")):
      self.assertSameStructure(
          list(string.ascii_lowercase), list(string.ascii_uppercase))

    # Verify same behavior with self.maxDiff = None
    self.maxDiff = None
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        'a[0] is 1 but b[0] is 3; a[1] is 2 but b[1] is 4',
        self.assertSameStructure, [1, 2], [3, 4])

  def test_same_structure_mapping_unchanged(self):
    default_a = collections.defaultdict(lambda: 'BAD MODIFICATION', {})
    dict_b = {'one': 'z'}
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        r"a lacks ['one'] but b has it with value 'z'",
        self.assertSameStructure, default_a, dict_b)
    self.assertEmpty(default_a)

    dict_a = {'one': 'z'}
    default_b = collections.defaultdict(lambda: 'BAD MODIFICATION', {})
    self.assertRaisesWithLiteralMatch(
        AssertionError,
        r"a has ['one'] with value 'z' but it's missing in b",
        self.assertSameStructure, dict_a, default_b)
    self.assertEmpty(default_b)

  def test_assert_json_equal_same(self):
    self.assertJsonEqual('{"success": true}', '{"success": true}')
    self.assertJsonEqual('{"success": true}', '{"success":true}')
    self.assertJsonEqual('true', 'true')
    self.assertJsonEqual('null', 'null')
    self.assertJsonEqual('false', 'false')
    self.assertJsonEqual('34', '34')
    self.assertJsonEqual('[1, 2, 3]', '[1,2,3]', msg='please PASS')
    self.assertJsonEqual('{"sequence": [1, 2, 3], "float": 23.42}',
                         '{"float": 23.42, "sequence": [1,2,3]}')
    self.assertJsonEqual('{"nest": {"spam": "eggs"}, "float": 23.42}',
                         '{"float": 23.42, "nest": {"spam":"eggs"}}')

  def test_assert_json_equal_different(self):
    with self.assertRaises(AssertionError):
      self.assertJsonEqual('{"success": true}', '{"success": false}')
    with self.assertRaises(AssertionError):
      self.assertJsonEqual('{"success": false}', '{"Success": false}')
    with self.assertRaises(AssertionError):
      self.assertJsonEqual('false', 'true')
    with self.assertRaises(AssertionError) as error_context:
      self.assertJsonEqual('null', '0', msg='I demand FAILURE')
    self.assertIn('I demand FAILURE', error_context.exception.args[0])
    self.assertIn('None', error_context.exception.args[0])
    with self.assertRaises(AssertionError):
      self.assertJsonEqual('[1, 0, 3]', '[1,2,3]')
    with self.assertRaises(AssertionError):
      self.assertJsonEqual('{"sequence": [1, 2, 3], "float": 23.42}',
                           '{"float": 23.42, "sequence": [1,0,3]}')
    with self.assertRaises(AssertionError):
      self.assertJsonEqual('{"nest": {"spam": "eggs"}, "float": 23.42}',
                           '{"float": 23.42, "nest": {"Spam":"beans"}}')

  def test_assert_json_equal_bad_json(self):
    with self.assertRaises(ValueError) as error_context:
      self.assertJsonEqual("alhg'2;#", '{"a": true}')
    self.assertIn('first', error_context.exception.args[0])
    self.assertIn('alhg', error_context.exception.args[0])

    with self.assertRaises(ValueError) as error_context:
      self.assertJsonEqual('{"a": true}', "alhg'2;#")
    self.assertIn('second', error_context.exception.args[0])
    self.assertIn('alhg', error_context.exception.args[0])

    with self.assertRaises(ValueError) as error_context:
      self.assertJsonEqual('', '')


class GetCommandStderrTestCase(absltest.TestCase):

  def setUp(self):
    self.original_environ = os.environ.copy()

  def tearDown(self):
    os.environ = self.original_environ

  def test_return_status(self):
    tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    returncode = (
        absltest.get_command_stderr(
            ['cat', os.path.join(tmpdir, 'file.txt')])[0])
    self.assertEqual(1, returncode)

  def test_stderr(self):
    tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    stderr = (
        absltest.get_command_stderr(
            ['cat', os.path.join(tmpdir, 'file.txt')])[1])
    if not PY_VERSION_2:
      stderr = stderr.decode('utf-8')
    self.assertRegex(stderr, 'No such file or directory')


class EqualityAssertionTest(absltest.TestCase):
  """This test verifies that absltest.failIfEqual actually tests __ne__.

  If a user class implements __eq__, unittest.failUnlessEqual will call it
  via first == second.   However, failIfEqual also calls
  first == second.   This means that while the caller may believe
  their __ne__ method is being tested, it is not.
  """

  class NeverEqual(object):
    """Objects of this class behave like NaNs."""

    def __eq__(self, unused_other):
      return False

    def __ne__(self, unused_other):
      return False

  class AllSame(object):
    """All objects of this class compare as equal."""

    def __eq__(self, unused_other):
      return True

    def __ne__(self, unused_other):
      return False

  class EqualityTestsWithEq(object):
    """Performs all equality and inequality tests with __eq__."""

    def __init__(self, value):
      self._value = value

    def __eq__(self, other):
      return self._value == other._value

    def __ne__(self, other):
      return not self.__eq__(other)

  class EqualityTestsWithNe(object):
    """Performs all equality and inequality tests with __ne__."""

    def __init__(self, value):
      self._value = value

    def __eq__(self, other):
      return not self.__ne__(other)

    def __ne__(self, other):
      return self._value != other._value

  class EqualityTestsWithCmp(object):

    def __init__(self, value):
      self._value = value

    def __cmp__(self, other):
      return cmp(self._value, other._value)

  class EqualityTestsWithLtEq(object):

    def __init__(self, value):
      self._value = value

    def __eq__(self, other):
      return self._value == other._value

    def __lt__(self, other):
      return self._value < other._value

  def test_all_comparisons_fail(self):
    i1 = self.NeverEqual()
    i2 = self.NeverEqual()
    self.assertFalse(i1 == i2)
    self.assertFalse(i1 != i2)

    # Compare two distinct objects
    self.assertFalse(i1 is i2)
    self.assertRaises(AssertionError, self.assertEqual, i1, i2)
    self.assertRaises(AssertionError, self.assertEquals, i1, i2)
    self.assertRaises(AssertionError, self.failUnlessEqual, i1, i2)
    self.assertRaises(AssertionError, self.assertNotEqual, i1, i2)
    self.assertRaises(AssertionError, self.assertNotEquals, i1, i2)
    self.assertRaises(AssertionError, self.failIfEqual, i1, i2)
    # A NeverEqual object should not compare equal to itself either.
    i2 = i1
    self.assertTrue(i1 is i2)
    self.assertFalse(i1 == i2)
    self.assertFalse(i1 != i2)
    self.assertRaises(AssertionError, self.assertEqual, i1, i2)
    self.assertRaises(AssertionError, self.assertEquals, i1, i2)
    self.assertRaises(AssertionError, self.failUnlessEqual, i1, i2)
    self.assertRaises(AssertionError, self.assertNotEqual, i1, i2)
    self.assertRaises(AssertionError, self.assertNotEquals, i1, i2)
    self.assertRaises(AssertionError, self.failIfEqual, i1, i2)

  def test_all_comparisons_succeed(self):
    a = self.AllSame()
    b = self.AllSame()
    self.assertFalse(a is b)
    self.assertTrue(a == b)
    self.assertFalse(a != b)
    self.assertEqual(a, b)
    self.assertEquals(a, b)
    self.failUnlessEqual(a, b)
    self.assertRaises(AssertionError, self.assertNotEqual, a, b)
    self.assertRaises(AssertionError, self.assertNotEquals, a, b)
    self.assertRaises(AssertionError, self.failIfEqual, a, b)

  def _perform_apple_apple_orange_checks(self, same_a, same_b, different):
    """Perform consistency checks with two apples and an orange.

    The two apples should always compare as being the same (and inequality
    checks should fail).  The orange should always compare as being different
    to each of the apples.

    Args:
      same_a: the first apple
      same_b: the second apple
      different: the orange
    """
    self.assertTrue(same_a == same_b)
    self.assertFalse(same_a != same_b)
    self.assertEqual(same_a, same_b)
    self.assertEquals(same_a, same_b)
    self.failUnlessEqual(same_a, same_b)
    if PY_VERSION_2:
      # Python 3 removes the global cmp function
      self.assertEqual(0, cmp(same_a, same_b))

    self.assertFalse(same_a == different)
    self.assertTrue(same_a != different)
    self.assertNotEqual(same_a, different)
    self.assertNotEquals(same_a, different)
    self.failIfEqual(same_a, different)
    if PY_VERSION_2:
      self.assertNotEqual(0, cmp(same_a, different))

    self.assertFalse(same_b == different)
    self.assertTrue(same_b != different)
    self.assertNotEqual(same_b, different)
    self.assertNotEquals(same_b, different)
    self.failIfEqual(same_b, different)
    if PY_VERSION_2:
      self.assertNotEqual(0, cmp(same_b, different))

  def test_comparison_with_eq(self):
    same_a = self.EqualityTestsWithEq(42)
    same_b = self.EqualityTestsWithEq(42)
    different = self.EqualityTestsWithEq(1769)
    self._perform_apple_apple_orange_checks(same_a, same_b, different)

  def test_comparison_with_ne(self):
    same_a = self.EqualityTestsWithNe(42)
    same_b = self.EqualityTestsWithNe(42)
    different = self.EqualityTestsWithNe(1769)
    self._perform_apple_apple_orange_checks(same_a, same_b, different)

  def test_comparison_with_cmp_or_lt_eq(self):
    if PY_VERSION_2:
      # In Python 3; the __cmp__ method is no longer special.
      cmp_or_lteq_class = self.EqualityTestsWithCmp
    else:
      cmp_or_lteq_class = self.EqualityTestsWithLtEq

    same_a = cmp_or_lteq_class(42)
    same_b = cmp_or_lteq_class(42)
    different = cmp_or_lteq_class(1769)
    self._perform_apple_apple_orange_checks(same_a, same_b, different)


class AssertSequenceStartsWithTest(absltest.TestCase):

  def setUp(self):
    self.a = [5, 'foo', {'c': 'd'}, None]

  def test_empty_sequence_starts_with_empty_prefix(self):
    self.assertSequenceStartsWith([], ())

  def test_sequence_prefix_is_an_empty_list(self):
    self.assertSequenceStartsWith([[]], ([], 'foo'))

  def test_raise_if_empty_prefix_with_non_empty_whole(self):
    with self.assertRaisesRegex(
        AssertionError, 'Prefix length is 0 but whole length is %d: %s' % (len(
            self.a), r"\[5, 'foo', \{'c': 'd'\}, None\]")):
      self.assertSequenceStartsWith([], self.a)

  def test_single_element_prefix(self):
    self.assertSequenceStartsWith([5], self.a)

  def test_two_element_prefix(self):
    self.assertSequenceStartsWith((5, 'foo'), self.a)

  def test_prefix_is_full_sequence(self):
    self.assertSequenceStartsWith([5, 'foo', {'c': 'd'}, None], self.a)

  def test_string_prefix(self):
    self.assertSequenceStartsWith('abc', 'abc123')

  def test_convert_non_sequence_prefix_to_sequence_and_try_again(self):
    self.assertSequenceStartsWith(5, self.a)

  def test_whole_not_asequence(self):
    msg = (r'For whole: len\(5\) is not supported, it appears to be type: '
           '<(type|class) \'int\'>')
    with self.assertRaisesRegex(AssertionError, msg):
      self.assertSequenceStartsWith(self.a, 5)

  def test_raise_if_sequence_does_not_start_with_prefix(self):
    msg = (r"prefix: \['foo', \{'c': 'd'\}\] not found at start of whole: "
           r"\[5, 'foo', \{'c': 'd'\}, None\].")
    with self.assertRaisesRegex(AssertionError, msg):
      self.assertSequenceStartsWith(['foo', {'c': 'd'}], self.a)

  def test_raise_if_types_ar_not_supported(self):
    with self.assertRaisesRegex(TypeError, 'unhashable type'):
      self.assertSequenceStartsWith({'a': 1, 2: 'b'},
                                    {'a': 1, 2: 'b', 'c': '3'})


class TestAssertEmpty(absltest.TestCase):
  longMessage = True

  def test_raises_if_not_asized_object(self):
    msg = "Expected a Sized object, got: 'int'"
    with self.assertRaisesRegex(AssertionError, msg):
      self.assertEmpty(1)

  def test_calls_len_not_bool(self):

    class BadList(list):

      def __bool__(self):
        return False

      __nonzero__ = __bool__

    bad_list = BadList()
    self.assertEmpty(bad_list)
    self.assertFalse(bad_list)

  def test_passes_when_empty(self):
    empty_containers = [
        list(),
        tuple(),
        dict(),
        set(),
        frozenset(),
        b'',
        u'',
        bytearray(),
    ]
    for container in empty_containers:
      self.assertEmpty(container)

  def test_raises_with_not_empty_containers(self):
    not_empty_containers = [
        [1],
        (1,),
        {'foo': 'bar'},
        {1},
        frozenset([1]),
        b'a',
        u'a',
        bytearray(b'a'),
    ]
    regexp = r'.* has length of 1\.$'
    for container in not_empty_containers:
      with self.assertRaisesRegex(AssertionError, regexp):
        self.assertEmpty(container)

  def test_user_message_added_to_default(self):
    msg = 'This is a useful message'
    whole_msg = re.escape('[1] has length of 1. : This is a useful message')
    with self.assertRaisesRegex(AssertionError, whole_msg):
      self.assertEmpty([1], msg=msg)


class TestAssertNotEmpty(absltest.TestCase):
  longMessage = True

  def test_raises_if_not_asized_object(self):
    msg = "Expected a Sized object, got: 'int'"
    with self.assertRaisesRegex(AssertionError, msg):
      self.assertNotEmpty(1)

  def test_calls_len_not_bool(self):

    class BadList(list):

      def __bool__(self):
        return False

      __nonzero__ = __bool__

    bad_list = BadList([1])
    self.assertNotEmpty(bad_list)
    self.assertFalse(bad_list)

  def test_passes_when_not_empty(self):
    not_empty_containers = [
        [1],
        (1,),
        {'foo': 'bar'},
        {1},
        frozenset([1]),
        b'a',
        u'a',
        bytearray(b'a'),
    ]
    for container in not_empty_containers:
      self.assertNotEmpty(container)

  def test_raises_with_empty_containers(self):
    empty_containers = [
        list(),
        tuple(),
        dict(),
        set(),
        frozenset(),
        b'',
        u'',
        bytearray(),
    ]
    regexp = r'.* has length of 0\.$'
    for container in empty_containers:
      with self.assertRaisesRegex(AssertionError, regexp):
        self.assertNotEmpty(container)

  def test_user_message_added_to_default(self):
    msg = 'This is a useful message'
    whole_msg = re.escape('[] has length of 0. : This is a useful message')
    with self.assertRaisesRegex(AssertionError, whole_msg):
      self.assertNotEmpty([], msg=msg)


class TestAssertLen(absltest.TestCase):
  longMessage = True

  def test_raises_if_not_asized_object(self):
    msg = "Expected a Sized object, got: 'int'"
    with self.assertRaisesRegex(AssertionError, msg):
      self.assertLen(1, 1)

  def test_passes_when_expected_len(self):
    containers = [
        [[1], 1],
        [(1, 2), 2],
        [{'a': 1, 'b': 2, 'c': 3}, 3],
        [{1, 2, 3, 4}, 4],
        [frozenset([1]), 1],
        [b'abc', 3],
        [u'def', 3],
        [bytearray(b'ghij'), 4],
    ]
    for container, expected_len in containers:
      self.assertLen(container, expected_len)

  def test_raises_when_unexpected_len(self):
    containers = [
        [1],
        (1, 2),
        {'a': 1, 'b': 2, 'c': 3},
        {1, 2, 3, 4},
        frozenset([1]),
        b'abc',
        u'def',
        bytearray(b'ghij'),
    ]
    for container in containers:
      regexp = r'.* has length of %d, expected 100\.$' % len(container)
      with self.assertRaisesRegex(AssertionError, regexp):
        self.assertLen(container, 100)

  def test_user_message_added_to_default(self):
    msg = 'This is a useful message'
    whole_msg = (
        r'\[1\] has length of 1, expected 100. : This is a useful message')
    with self.assertRaisesRegex(AssertionError, whole_msg):
      self.assertLen([1], 100, msg)


class TestLoaderTest(absltest.TestCase):
  """Tests that the TestLoader bans methods named TestFoo."""

  # pylint: disable=invalid-name
  class Valid(absltest.TestCase):
    """Test case containing a variety of valid names."""

    test_property = 1
    TestProperty = 2

    @staticmethod
    def TestStaticMethod():
      pass

    @staticmethod
    def TestStaticMethodWithArg(foo):
      pass

    @classmethod
    def TestClassMethod(cls):
      pass

    def Test(self):
      pass

    def TestingHelper(self):
      pass

    def testMethod(self):
      pass

    def TestHelperWithParams(self, a, b):
      pass

    def TestHelperWithVarargs(self, *args, **kwargs):
      pass

    def TestHelperWithDefaults(self, a=5):
      pass

  class Invalid(absltest.TestCase):
    """Test case containing a suspicious method."""

    def testMethod(self):
      pass

    def TestSuspiciousMethod(self):
      pass
  # pylint: enable=invalid-name

  def setUp(self):
    self.loader = absltest.TestLoader()

  def test_valid(self):
    suite = self.loader.loadTestsFromTestCase(TestLoaderTest.Valid)
    self.assertEquals(1, suite.countTestCases())

  def testInvalid(self):
    with self.assertRaisesRegex(TypeError, 'TestSuspiciousMethod'):
      self.loader.loadTestsFromTestCase(TestLoaderTest.Invalid)


class InitNotNecessaryForAssertsTest(absltest.TestCase):
  """TestCase assertions should work even if __init__ wasn't correctly called.

  This is a workaround, see comment in
  absltest.TestCase._getAssertEqualityFunc. We know that not calling
  __init__ of a superclass is a bad thing, but people keep doing them,
  and this (even if a little bit dirty) saves them from shooting
  themselves in the foot.
  """

  def test_subclass(self):

    class Subclass(absltest.TestCase):

      def __init__(self):  # pylint: disable=super-init-not-called
        pass

    Subclass().assertEquals({}, {})

  def test_multiple_inheritance(self):

    class Foo(object):

      def __init__(self, *args, **kwargs):
        pass

    class Subclass(Foo, absltest.TestCase):
      pass

    Subclass().assertEquals({}, {})


class GetCommandStringTest(parameterized.TestCase):

  @parameterized.parameters(
      ([], '', ''),
      ([''], "''", ''),
      (['command', 'arg-0'], "'command' 'arg-0'", 'command arg-0'),
      ([u'command', u'arg-0'], "'command' 'arg-0'", u'command arg-0'),
      (["foo'bar"], "'foo'\"'\"'bar'", "foo'bar"),
      (['foo"bar'], "'foo\"bar'", 'foo"bar'),
      ('command arg-0', 'command arg-0', 'command arg-0'),
      (u'command arg-0', 'command arg-0', 'command arg-0'))
  def test_get_command_string(
      self, command, expected_non_windows, expected_windows):
    expected = expected_windows if os.name == 'nt' else expected_non_windows
    self.assertEqual(expected, absltest.get_command_string(command))


class TempFileTest(absltest.TestCase, HelperMixin):

  def assert_dir_exists(self, temp_dir):
    path = temp_dir.full_path
    self.assertTrue(os.path.exists(path), 'Dir {} does not exist'.format(path))
    self.assertTrue(os.path.isdir(path),
                    'Path {} exists, but is not a directory'.format(path))

  def assert_file_exists(self, temp_file, expected_content=b''):
    path = temp_file.full_path
    self.assertTrue(os.path.exists(path), 'File {} does not exist'.format(path))
    self.assertTrue(os.path.isfile(path),
                    'Path {} exists, but is not a file'.format(path))

    mode = 'rb' if isinstance(expected_content, bytes) else 'rt'
    with io.open(path, mode) as fp:
      actual = fp.read()
    self.assertEqual(expected_content, actual)

  def run_tempfile_helper(self, cleanup, expected_paths):
    tmpdir = self.create_tempdir('helper-test-temp-dir')
    env = {
        'ABSLTEST_TEST_HELPER_TEMPFILE_CLEANUP': cleanup,
        'TEST_TMPDIR': tmpdir.full_path,
        }
    stdout, stderr = self.run_helper(0, ['TempFileHelperTest'], env,
                                     expect_success=False)
    output = ('\n=== Helper output ===\n'
              '----- stdout -----\n{}\n'
              '----- end stdout -----\n'
              '----- stderr -----\n{}\n'
              '----- end stderr -----\n'
              '===== end helper output =====').format(stdout, stderr)
    self.assertIn('test_failure', stderr, output)

    # Adjust paths to match on Windows
    expected_paths = {path.replace('/', os.sep) for path in expected_paths}

    actual = {
        os.path.relpath(f, tmpdir.full_path)
        for f in _listdir_recursive(tmpdir.full_path)
        if f != tmpdir.full_path
    }
    self.assertEqual(expected_paths, actual, output)

  def test_create_file_pre_existing_readonly(self):
    first = self.create_tempfile('foo', content='first')
    os.chmod(first.full_path, 0o444)
    second = self.create_tempfile('foo', content='second')
    self.assertEqual('second', first.read_text())
    self.assertEqual('second', second.read_text())

  def test_unnamed(self):
    td = self.create_tempdir()
    self.assert_dir_exists(td)

    tdf = td.create_file()
    self.assert_file_exists(tdf)

    tdd = td.mkdir()
    self.assert_dir_exists(tdd)

    tf = self.create_tempfile()
    self.assert_file_exists(tf)

  def test_named(self):
    td = self.create_tempdir('d')
    self.assert_dir_exists(td)

    tdf = td.create_file('df')
    self.assert_file_exists(tdf)

    tdd = td.mkdir('dd')
    self.assert_dir_exists(tdd)

    tf = self.create_tempfile('f')
    self.assert_file_exists(tf)

  def test_nested_paths(self):
    td = self.create_tempdir('d1/d2')
    self.assert_dir_exists(td)

    tdf = td.create_file('df1/df2')
    self.assert_file_exists(tdf)

    tdd = td.mkdir('dd1/dd2')
    self.assert_dir_exists(tdd)

    tf = self.create_tempfile('f1/f2')
    self.assert_file_exists(tf)

  def test_tempdir_create_file(self):
    td = self.create_tempdir()
    td.create_file(content='text')

  def test_tempfile_text(self):
    tf = self.create_tempfile(content='text')
    self.assert_file_exists(tf, 'text')
    self.assertEqual('text', tf.read_text())

    with tf.open_text() as fp:
      self.assertEqual('text', fp.read())

    with tf.open_text('w') as fp:
      fp.write(u'text-from-open-write')
    self.assertEqual('text-from-open-write', tf.read_text())

    tf.write_text('text-from-write-text')
    self.assertEqual('text-from-write-text', tf.read_text())

  def test_tempfile_bytes(self):
    tf = self.create_tempfile(content=b'\x00\x01\x02')
    self.assert_file_exists(tf, b'\x00\x01\x02')
    self.assertEqual(b'\x00\x01\x02', tf.read_bytes())

    with tf.open_bytes() as fp:
      self.assertEqual(b'\x00\x01\x02', fp.read())

    with tf.open_bytes('wb') as fp:
      fp.write(b'\x03')
    self.assertEqual(b'\x03', tf.read_bytes())

    tf.write_bytes(b'\x04')
    self.assertEqual(b'\x04', tf.read_bytes())

  def test_tempdir_same_name(self):
    """Make sure the same directory name can be used."""
    td1 = self.create_tempdir('foo')
    td2 = self.create_tempdir('foo')
    self.assert_dir_exists(td1)
    self.assert_dir_exists(td2)

  def test_tempfile_cleanup_success(self):
    expected = {
        'TempFileHelperTest',
        'TempFileHelperTest/test_failure',
        'TempFileHelperTest/test_failure/failure',
        'TempFileHelperTest/test_success',
    }
    self.run_tempfile_helper('SUCCESS', expected)

  def test_tempfile_cleanup_always(self):
    expected = {
        'TempFileHelperTest',
        'TempFileHelperTest/test_failure',
        'TempFileHelperTest/test_success',
    }
    self.run_tempfile_helper('ALWAYS', expected)

  def test_tempfile_cleanup_off(self):
    expected = {
        'TempFileHelperTest',
        'TempFileHelperTest/test_failure',
        'TempFileHelperTest/test_failure/failure',
        'TempFileHelperTest/test_success',
        'TempFileHelperTest/test_success/success',
    }
    self.run_tempfile_helper('OFF', expected)


def _listdir_recursive(path):
  for dirname, _, filenames in os.walk(path):
    yield dirname
    for filename in filenames:
      yield os.path.join(dirname, filename)


if __name__ == '__main__':
  absltest.main()
