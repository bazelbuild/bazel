"""Tests for absl.third_party.unittest3_backport."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from absl.testing import absltest
from absl.testing import xml_reporter
import mock
import six


class MockTestResult(xml_reporter._TextAndXMLTestResult):

  def __init__(self):
    super(MockTestResult, self).__init__(six.StringIO(), six.StringIO(),
                                         'description', False)
    self.subtest_success = []
    self.subtest_failure = []

  def addSubTest(self, test, subtest, err):  # pylint: disable=invalid-name
    super(MockTestResult, self).addSubTest(test, subtest, err)

    if six.PY2:
      params = {}
      for param in subtest.params:
        for param_name, param_value in param.items():
          params[param_name] = param_value
    else:
      params = dict(subtest.params)
    if err is not None:
      self.addSubTestFailure(params)
    else:
      self.addSubTestSuccess(params)

  def addSubTestFailure(self, params):  # pylint: disable=invalid-name
    self.subtest_failure.append(params)

  def addSubTestSuccess(self, params):  # pylint: disable=invalid-name
    self.subtest_success.append(params)


class MockTestResultWithoutSubTest(xml_reporter._TextAndXMLTestResult):
  # hasattr(MockTestResultWithoutSubTest, addSubTest) return False

  def __init__(self):
    super(MockTestResultWithoutSubTest, self).__init__(six.StringIO(),
                                                       six.StringIO(),
                                                       'description',
                                                       False)

  @property
  def addSubTest(self):  # pylint: disable=invalid-name
    raise AttributeError


class Unittest3BackportTest(absltest.TestCase):

  def test_subtest_pass(self):

    class Foo(absltest.TestCase):

      def runTest(self):
        for i in [1, 2]:
          with self.subTest(i=i):
            for j in [2, 3]:
              with self.subTest(j=j):
                pass

    result = MockTestResult()
    Foo().run(result)
    expected_success = [{'i': 1, 'j': 2}, {'i': 1, 'j': 3}, {'i': 1},
                        {'i': 2, 'j': 2}, {'i': 2, 'j': 3}, {'i': 2}]
    self.assertListEqual(result.subtest_success, expected_success)

  def test_subtest_fail(self):
    class Foo(absltest.TestCase):

      def runTest(self):
        for i in [1, 2]:
          with self.subTest(i=i):
            for j in [2, 3]:
              with self.subTest(j=j):
                if j == 2:
                  self.fail('failure')

    result = MockTestResult()
    Foo().run(result)

    # The first layer subtest result is only added to the output when it is a
    # success
    expected_success = [{'i': 1, 'j': 3}, {'i': 2, 'j': 3}]
    expected_failure = [{'i': 1, 'j': 2}, {'i': 2, 'j': 2}]
    self.assertListEqual(expected_success, result.subtest_success)
    self.assertListEqual(expected_failure, result.subtest_failure)

  def test_subtest_expected_failure(self):
    class Foo(absltest.TestCase):

      @unittest.expectedFailure
      def runTest(self):
        for i in [1, 2, 3]:
          with self.subTest(i=i):
            self.assertEqual(i, 2)

    foo = Foo()
    with mock.patch.object(foo, '_addExpectedFailure',
                           autospec=True) as mock_subtest_expected_failure:
      result = MockTestResult()
      foo.run(result)
      self.assertEqual(mock_subtest_expected_failure.call_count, 1)

  def test_subtest_unexpected_success(self):
    class Foo(absltest.TestCase):

      @unittest.expectedFailure
      def runTest(self):
        for i in [1, 2, 3]:
          with self.subTest(i=i):
            self.assertEqual(i, i)

    foo = Foo()
    with mock.patch.object(foo, '_addUnexpectedSuccess',
                           autospec=True) as mock_subtest_unexpected_success:
      result = MockTestResult()
      foo.run(result)
      self.assertEqual(mock_subtest_unexpected_success.call_count, 1)

  def test_subtest_fail_fast(self):
    # Ensure failfast works with subtest

    class Foo(absltest.TestCase):

      def runTest(self):
        with self.subTest(i=1):
          self.fail('failure')
        with self.subTest(i=2):
          self.fail('failure')
        self.fail('failure')

    result = MockTestResult()
    result.failfast = True
    Foo().run(result)
    expected_failure = [{'i': 1}]
    self.assertListEqual(expected_failure, result.subtest_failure)

  def test_subtest_skip(self):
    # When a test case is skipped, addSubTest should not be called

    class Foo(absltest.TestCase):

      @unittest.skip('no reason')
      def runTest(self):
        for i in [1, 2, 3]:
          with self.subTest(i=i):
            self.assertEqual(i, i)

    foo = Foo()
    result = MockTestResult()

    with mock.patch.object(foo, '_addSkip', autospec=True) as mock_test_skip:
      with mock.patch.object(result, 'addSubTestSuccess',
                             autospec=True) as mock_subtest_success:
        foo.run(result)
        self.assertEqual(mock_test_skip.call_count, 1)
        self.assertEqual(mock_subtest_success.call_count, 0)

  @mock.patch.object(MockTestResultWithoutSubTest, 'addFailure', autospec=True)
  def test_subtest_legacy(self, mock_test_fail):
    # When the result object does not have addSubTest method,
    # text execution stops after the first subtest failure.

    class Foo(absltest.TestCase):

      def runTest(self):
        for i in [1, 2, 3]:
          with self.subTest(i=i):
            if i == 1:
              self.fail('failure')
            for j in [2, 3]:
              with self.subTest(j=j):
                if i * j == 6:
                  raise RuntimeError('raised by Foo.test')

    result = MockTestResultWithoutSubTest()

    Foo().run(result)
    self.assertEqual(mock_test_fail.call_count, 1)

if __name__ == '__main__':
  absltest.main()
