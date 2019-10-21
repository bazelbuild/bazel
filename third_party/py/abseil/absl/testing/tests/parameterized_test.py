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

"""Tests for absl.testing.parameterized."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

from absl._collections_abc import abc
from absl.testing import absltest
from absl.testing import parameterized
import six
from six.moves import range  # pylint: disable=redefined-builtin


class MyOwnClass(object):
  pass


def dummy_decorator(method):

  def decorated(*args, **kwargs):
    return method(*args, **kwargs)

  return decorated


def dict_decorator(key, value):
  """Sample implementation of a chained decorator.

  Sets a single field in a dict on a test with a dict parameter.
  Uses the exposed '_ParameterizedTestIter.testcases' field to
  modify arguments from previous decorators to allow decorator chains.

  Args:
    key: key to map to
    value: value to set

  Returns:
    The test decorator
  """
  def decorator(test_method):
    # If decorating result of another dict_decorator
    if isinstance(test_method, abc.Iterable):
      actual_tests = []
      for old_test in test_method.testcases:
        # each test is a ('test_suffix', dict) tuple
        new_dict = old_test[1].copy()
        new_dict[key] = value
        test_suffix = '%s_%s_%s' % (old_test[0], key, value)
        actual_tests.append((test_suffix, new_dict))

      test_method.testcases = actual_tests
      return test_method
    else:
      test_suffix = ('_%s_%s') % (key, value)
      tests_to_make = ((test_suffix, {key: value}),)
      # 'test_method' here is the original test method
      return parameterized.named_parameters(*tests_to_make)(test_method)
  return decorator


class ParameterizedTestsTest(absltest.TestCase):
  # The test testcases are nested so they're not
  # picked up by the normal test case loader code.

  class GoodAdditionParams(parameterized.TestCase):

    @parameterized.parameters(
        (1, 2, 3),
        (4, 5, 9))
    def test_addition(self, op1, op2, result):
      self.arguments = (op1, op2, result)
      self.assertEqual(result, op1 + op2)

  # This class does not inherit from TestCase.
  class BadAdditionParams(absltest.TestCase):

    @parameterized.parameters(
        (1, 2, 3),
        (4, 5, 9))
    def test_addition(self, op1, op2, result):
      pass  # Always passes, but not called w/out TestCase.

  class MixedAdditionParams(parameterized.TestCase):

    @parameterized.parameters(
        (1, 2, 1),
        (4, 5, 9))
    def test_addition(self, op1, op2, result):
      self.arguments = (op1, op2, result)
      self.assertEqual(result, op1 + op2)

  class DictionaryArguments(parameterized.TestCase):

    @parameterized.parameters(
        {'op1': 1, 'op2': 2, 'result': 3},
        {'op1': 4, 'op2': 5, 'result': 9})
    def test_addition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)

  class NoParameterizedTests(parameterized.TestCase):
    # iterable member with non-matching name
    a = 'BCD'
    # member with matching name, but not a generator
    testInstanceMember = None  # pylint: disable=invalid-name
    test_instance_member = None

    # member with a matching name and iterator, but not a generator
    testString = 'foo'  # pylint: disable=invalid-name
    test_string = 'foo'

    # generator, but no matching name
    def someGenerator(self):  # pylint: disable=invalid-name
      yield
      yield
      yield

    def some_generator(self):
      yield
      yield
      yield

    # Generator function, but not a generator instance.
    def testGenerator(self):
      yield
      yield
      yield

    def test_generator(self):
      yield
      yield
      yield

    def testNormal(self):
      self.assertEqual(3, 1 + 2)

    def test_normal(self):
      self.assertEqual(3, 1 + 2)

  class ArgumentsWithAddresses(parameterized.TestCase):

    @parameterized.parameters(
        (object(),),
        (MyOwnClass(),),
    )
    def test_something(self, case):
      pass

  class CamelCaseNamedTests(parameterized.TestCase):

    @parameterized.named_parameters(
        ('Interesting', 0),
    )
    def testSingle(self, case):
      pass

    @parameterized.named_parameters(
        {'testcase_name': 'Interesting', 'case': 0},
    )
    def testDictSingle(self, case):
      pass

    @parameterized.named_parameters(
        ('Interesting', 0),
        ('Boring', 1),
    )
    def testSomething(self, case):
      pass

    @parameterized.named_parameters(
        {'testcase_name': 'Interesting', 'case': 0},
        {'testcase_name': 'Boring', 'case': 1},
    )
    def testDictSomething(self, case):
      pass

    @parameterized.named_parameters(
        {'testcase_name': 'Interesting', 'case': 0},
        ('Boring', 1),
    )
    def testMixedSomething(self, case):
      pass

    def testWithoutParameters(self):
      pass

  class NamedTests(parameterized.TestCase):
    """Example tests using PEP-8 style names instead of camel-case."""

    @parameterized.named_parameters(
        ('interesting', 0),
    )
    def test_single(self, case):
      pass

    @parameterized.named_parameters(
        {'testcase_name': 'interesting', 'case': 0},
    )
    def test_dict_single(self, case):
      pass

    @parameterized.named_parameters(
        ('interesting', 0),
        ('boring', 1),
    )
    def test_something(self, case):
      pass

    @parameterized.named_parameters(
        {'testcase_name': 'interesting', 'case': 0},
        {'testcase_name': 'boring', 'case': 1},
    )
    def test_dict_something(self, case):
      pass

    @parameterized.named_parameters(
        {'testcase_name': 'interesting', 'case': 0},
        ('boring', 1),
    )
    def test_mixed_something(self, case):
      pass

    def test_without_parameters(self):
      pass

  class ChainedTests(parameterized.TestCase):

    @dict_decorator('cone', 'waffle')
    @dict_decorator('flavor', 'strawberry')
    def test_chained(self, dictionary):
      self.assertDictEqual(dictionary, {'cone': 'waffle',
                                        'flavor': 'strawberry'})

  class SingletonListExtraction(parameterized.TestCase):

    @parameterized.parameters(
        (i, i * 2) for i in range(10))
    def test_something(self, unused_1, unused_2):
      pass

  class SingletonArgumentExtraction(parameterized.TestCase):

    @parameterized.parameters(1, 2, 3, 4, 5, 6)
    def test_numbers(self, unused_1):
      pass

    @parameterized.parameters('foo', 'bar', 'baz')
    def test_strings(self, unused_1):
      pass

  @parameterized.parameters(
      (1, 2, 3),
      (4, 5, 9))
  class DecoratedClass(parameterized.TestCase):

    def test_add(self, arg1, arg2, arg3):
      self.assertEqual(arg1 + arg2, arg3)

    def test_subtract_fail(self, arg1, arg2, arg3):
      self.assertEqual(arg3 + arg2, arg1)

  @parameterized.parameters(
      (a, b, a+b) for a in range(1, 5) for b in range(1, 5))
  class GeneratorDecoratedClass(parameterized.TestCase):

    def test_add(self, arg1, arg2, arg3):
      self.assertEqual(arg1 + arg2, arg3)

    def test_subtract_fail(self, arg1, arg2, arg3):
      self.assertEqual(arg3 + arg2, arg1)

  @parameterized.parameters(
      (1, 2, 3),
      (4, 5, 9),
  )
  class DecoratedBareClass(absltest.TestCase):

    def test_add(self, arg1, arg2, arg3):
      self.assertEqual(arg1 + arg2, arg3)

  class OtherDecoratorUnnamed(parameterized.TestCase):

    @dummy_decorator
    @parameterized.parameters((1), (2))
    def test_other_then_parameterized(self, arg1):
      pass

    @parameterized.parameters((1), (2))
    @dummy_decorator
    def test_parameterized_then_other(self, arg1):
      pass

  class OtherDecoratorNamed(parameterized.TestCase):

    @dummy_decorator
    @parameterized.named_parameters(('a', 1), ('b', 2))
    def test_other_then_parameterized(self, arg1):
      pass

    @parameterized.named_parameters(('a', 1), ('b', 2))
    @dummy_decorator
    def test_parameterized_then_other(self, arg1):
      pass

  class OtherDecoratorNamedWithDict(parameterized.TestCase):

    @dummy_decorator
    @parameterized.named_parameters(
        {'testcase_name': 'a', 'arg1': 1},
        {'testcase_name': 'b', 'arg1': 2})
    def test_other_then_parameterized(self, arg1):
      pass

    @parameterized.named_parameters(
        {'testcase_name': 'a', 'arg1': 1},
        {'testcase_name': 'b', 'arg1': 2})
    @dummy_decorator
    def test_parameterized_then_other(self, arg1):
      pass

  class UniqueDescriptiveNamesTest(parameterized.TestCase):

    class JustBeingMean(object):

      def __repr__(self):
        return '13) (2'

    @parameterized.parameters(13, 13)
    def test_normal(self, number):
      del number

    @parameterized.parameters(13, 13, JustBeingMean())
    def test_double_conflict(self, number):
      del number

    @parameterized.parameters(13, JustBeingMean(), 13, 13)
    def test_triple_conflict(self, number):
      del number

  class MultiGeneratorsTestCase(parameterized.TestCase):

    @parameterized.parameters((i for i in (1, 2, 3)), (i for i in (3, 2, 1)))
    def test_sum(self, a, b, c):
      self.assertEqual(6, sum([a, b, c]))

  class NamedParametersReusableTestCase(parameterized.TestCase):
    named_params_a = (
        {'testcase_name': 'dict_a', 'unused_obj': 0},
        ('list_a', 1),
    )
    named_params_b = (
        {'testcase_name': 'dict_b', 'unused_obj': 2},
        ('list_b', 3),
    )
    named_params_c = (
        {'testcase_name': 'dict_c', 'unused_obj': 4},
        ('list_b', 5),
    )

    @parameterized.named_parameters(*(named_params_a + named_params_b))
    def testSomething(self, unused_obj):
      pass

    @parameterized.named_parameters(*(named_params_a + named_params_c))
    def testSomethingElse(self, unused_obj):
      pass

  class SuperclassTestCase(parameterized.TestCase):

    @parameterized.parameters('foo', 'bar')
    def test_name(self, name):
      del name

  class SubclassTestCase(SuperclassTestCase):
    pass

  @unittest.skipIf(
      (sys.version_info[:2] == (3, 7) and sys.version_info[2] in {0, 1, 2}),
      'Python 3.7.0 to 3.7.2 have a bug that breaks this test, see '
      'https://bugs.python.org/issue35767')
  def test_missing_inheritance(self):
    ts = unittest.makeSuite(self.BadAdditionParams)
    self.assertEqual(1, ts.countTestCases())

    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(1, res.testsRun)
    self.assertFalse(res.wasSuccessful())
    self.assertIn('without having inherited', str(res.errors[0]))

  def test_correct_extraction_numbers(self):
    ts = unittest.makeSuite(self.GoodAdditionParams)
    self.assertEqual(2, ts.countTestCases())

  def test_successful_execution(self):
    ts = unittest.makeSuite(self.GoodAdditionParams)

    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def test_correct_arguments(self):
    ts = unittest.makeSuite(self.GoodAdditionParams)
    res = unittest.TestResult()

    params = set([
        (1, 2, 3),
        (4, 5, 9)])
    for test in ts:
      test(res)
      self.assertIn(test.arguments, params)
      params.remove(test.arguments)
    self.assertEqual(0, len(params))

  def test_recorded_failures(self):
    ts = unittest.makeSuite(self.MixedAdditionParams)
    self.assertEqual(2, ts.countTestCases())

    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertFalse(res.wasSuccessful())
    self.assertEqual(1, len(res.failures))
    self.assertEqual(0, len(res.errors))

  def test_short_description(self):
    ts = unittest.makeSuite(self.GoodAdditionParams)
    short_desc = list(ts)[0].shortDescription().split('\n')
    self.assertEqual(
        'test_addition(1, 2, 3)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_addition(1, 2, 3)'))

  def test_short_description_addresses_removed(self):
    ts = unittest.makeSuite(self.ArgumentsWithAddresses)
    short_desc = list(ts)[0].shortDescription().split('\n')
    self.assertEqual(
        'test_something(<object>)', short_desc[1])
    short_desc = list(ts)[1].shortDescription().split('\n')
    self.assertEqual(
        'test_something(<__main__.MyOwnClass>)', short_desc[1])

  def test_id(self):
    ts = unittest.makeSuite(self.ArgumentsWithAddresses)
    self.assertEqual(
        (unittest.util.strclass(self.ArgumentsWithAddresses) +
         '.test_something(<object>)'),
        list(ts)[0].id())
    ts = unittest.makeSuite(self.GoodAdditionParams)
    self.assertEqual(
        (unittest.util.strclass(self.GoodAdditionParams) +
         '.test_addition(1, 2, 3)'),
        list(ts)[0].id())

  def test_dict_parameters(self):
    ts = unittest.makeSuite(self.DictionaryArguments)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def test_no_parameterized_tests(self):
    ts = unittest.makeSuite(self.NoParameterizedTests)
    self.assertEqual(4, ts.countTestCases())
    short_descs = [x.shortDescription() for x in list(ts)]
    full_class_name = unittest.util.strclass(self.NoParameterizedTests)
    self.assertSameElements(
        [
            'testGenerator (%s)' % (full_class_name,),
            'test_generator (%s)' % (full_class_name,),
            'testNormal (%s)' % (full_class_name,),
            'test_normal (%s)' % (full_class_name,),
        ],
        short_descs)

  def test_generator_tests(self):
    with self.assertRaises(AssertionError):

      # This fails because the generated test methods share the same test id.
      class GeneratorTests(parameterized.TestCase):
        test_generator_method = (lambda self: None for _ in range(10))

      del GeneratorTests

  def test_named_parameters_run(self):
    ts = unittest.makeSuite(self.NamedTests)
    self.assertEqual(9, ts.countTestCases())
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(9, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def test_named_parameters_id(self):
    ts = sorted(unittest.makeSuite(self.CamelCaseNamedTests),
                key=lambda t: t.id())
    self.assertLen(ts, 9)
    full_class_name = unittest.util.strclass(self.CamelCaseNamedTests)
    self.assertEqual(
        full_class_name + '.testDictSingleInteresting',
        ts[0].id())
    self.assertEqual(
        full_class_name + '.testDictSomethingBoring',
        ts[1].id())
    self.assertEqual(
        full_class_name + '.testDictSomethingInteresting',
        ts[2].id())
    self.assertEqual(
        full_class_name + '.testMixedSomethingBoring',
        ts[3].id())
    self.assertEqual(
        full_class_name + '.testMixedSomethingInteresting',
        ts[4].id())
    self.assertEqual(
        full_class_name + '.testSingleInteresting',
        ts[5].id())
    self.assertEqual(
        full_class_name + '.testSomethingBoring',
        ts[6].id())
    self.assertEqual(
        full_class_name + '.testSomethingInteresting',
        ts[7].id())
    self.assertEqual(
        full_class_name + '.testWithoutParameters',
        ts[8].id())

  def test_named_parameters_id_with_underscore_case(self):
    ts = sorted(unittest.makeSuite(self.NamedTests),
                key=lambda t: t.id())
    self.assertLen(ts, 9)
    full_class_name = unittest.util.strclass(self.NamedTests)
    self.assertEqual(
        full_class_name + '.test_dict_single_interesting',
        ts[0].id())
    self.assertEqual(
        full_class_name + '.test_dict_something_boring',
        ts[1].id())
    self.assertEqual(
        full_class_name + '.test_dict_something_interesting',
        ts[2].id())
    self.assertEqual(
        full_class_name + '.test_mixed_something_boring',
        ts[3].id())
    self.assertEqual(
        full_class_name + '.test_mixed_something_interesting',
        ts[4].id())
    self.assertEqual(
        full_class_name + '.test_single_interesting',
        ts[5].id())
    self.assertEqual(
        full_class_name + '.test_something_boring',
        ts[6].id())
    self.assertEqual(
        full_class_name + '.test_something_interesting',
        ts[7].id())
    self.assertEqual(
        full_class_name + '.test_without_parameters',
        ts[8].id())

  def test_named_parameters_short_description(self):
    ts = sorted(unittest.makeSuite(self.NamedTests),
                key=lambda t: t.id())
    short_desc = ts[0].shortDescription().split('\n')
    self.assertEqual(
        'test_dict_single_interesting(case=0)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_dict_single_interesting'))

    short_desc = ts[1].shortDescription().split('\n')
    self.assertEqual(
        'test_dict_something_boring(case=1)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_dict_something_boring'))

    short_desc = ts[2].shortDescription().split('\n')
    self.assertEqual(
        'test_dict_something_interesting(case=0)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_dict_something_interesting'))

    short_desc = ts[3].shortDescription().split('\n')
    self.assertEqual(
        'test_mixed_something_boring(1)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_mixed_something_boring'))

    short_desc = ts[4].shortDescription().split('\n')
    self.assertEqual(
        'test_mixed_something_interesting(case=0)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_mixed_something_interesting'))

    short_desc = ts[6].shortDescription().split('\n')
    self.assertEqual(
        'test_something_boring(1)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_something_boring'))

    short_desc = ts[7].shortDescription().split('\n')
    self.assertEqual(
        'test_something_interesting(0)', short_desc[1])
    self.assertTrue(
        short_desc[0].startswith('test_something_interesting'))

  def test_load_tuple_named_test(self):
    loader = unittest.TestLoader()
    ts = list(loader.loadTestsFromName('NamedTests.test_something_interesting',
                                       module=self))
    self.assertEqual(1, len(ts))
    self.assertEndsWith(ts[0].id(), '.test_something_interesting')

  def test_load_dict_named_test(self):
    loader = unittest.TestLoader()
    ts = list(
        loader.loadTestsFromName(
            'NamedTests.test_dict_something_interesting', module=self))
    self.assertEqual(1, len(ts))
    self.assertEndsWith(ts[0].id(), '.test_dict_something_interesting')

  def test_load_mixed_named_test(self):
    loader = unittest.TestLoader()
    ts = list(
        loader.loadTestsFromName(
            'NamedTests.test_mixed_something_interesting', module=self))
    self.assertEqual(1, len(ts))
    self.assertEndsWith(ts[0].id(), '.test_mixed_something_interesting')

  def test_duplicate_named_test_fails(self):
    with self.assertRaises(parameterized.DuplicateTestNameError):

      class _(parameterized.TestCase):

        @parameterized.named_parameters(
            ('Interesting', 0),
            ('Interesting', 1),
        )
        def test_something(self, unused_obj):
          pass

  def test_duplicate_dict_named_test_fails(self):
    with self.assertRaises(parameterized.DuplicateTestNameError):

      class _(parameterized.TestCase):

        @parameterized.named_parameters(
            {'testcase_name': 'Interesting', 'unused_obj': 0},
            {'testcase_name': 'Interesting', 'unused_obj': 1},
        )
        def test_dict_something(self, unused_obj):
          pass

  def test_duplicate_mixed_named_test_fails(self):
    with self.assertRaises(parameterized.DuplicateTestNameError):

      class _(parameterized.TestCase):

        @parameterized.named_parameters(
            {'testcase_name': 'Interesting', 'unused_obj': 0},
            ('Interesting', 1),
        )
        def test_mixed_something(self, unused_obj):
          pass

  def test_parameterized_test_iter_has_testcases_property(self):
    @parameterized.parameters(1, 2, 3, 4, 5, 6)
    def test_something(unused_self, unused_obj):  # pylint: disable=invalid-name
      pass

    expected_testcases = [1, 2, 3, 4, 5, 6]
    self.assertTrue(hasattr(test_something, 'testcases'))
    self.assertItemsEqual(expected_testcases, test_something.testcases)

  def test_chained_decorator(self):
    ts = unittest.makeSuite(self.ChainedTests)
    self.assertEqual(1, ts.countTestCases())
    test = next(t for t in ts)
    self.assertTrue(hasattr(test, 'test_chained_flavor_strawberry_cone_waffle'))
    res = unittest.TestResult()

    ts.run(res)
    self.assertEqual(1, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def test_singleton_list_extraction(self):
    ts = unittest.makeSuite(self.SingletonListExtraction)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(10, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def test_singleton_argument_extraction(self):
    ts = unittest.makeSuite(self.SingletonArgumentExtraction)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(9, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def test_decorated_bare_class(self):
    ts = unittest.makeSuite(self.DecoratedBareClass)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful(), msg=str(res.failures))

  def test_decorated_class(self):
    ts = unittest.makeSuite(self.DecoratedClass)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(4, res.testsRun)
    self.assertEqual(2, len(res.failures))

  def test_generator_decorated_class(self):
    ts = unittest.makeSuite(self.GeneratorDecoratedClass)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(32, res.testsRun)
    self.assertEqual(16, len(res.failures))

  def test_no_duplicate_decorations(self):
    with self.assertRaises(AssertionError):

      @parameterized.parameters(1, 2, 3, 4)
      class _(parameterized.TestCase):

        @parameterized.parameters(5, 6, 7, 8)
        def test_something(self, unused_obj):
          pass

  def tes_double_class_decorations_not_supported(self):

    @parameterized.parameters('foo', 'bar')
    class SuperclassWithClassDecorator(parameterized.TestCase):

      def test_name(self, name):
        del name

    with self.assertRaises(AssertionError):

      @parameterized.parameters('foo', 'bar')
      class SubclassWithClassDecorator(SuperclassWithClassDecorator):
        pass

      del SubclassWithClassDecorator

  def test_other_decorator_ordering_unnamed(self):
    ts = unittest.makeSuite(self.OtherDecoratorUnnamed)
    res = unittest.TestResult()
    ts.run(res)
    # Two for when the parameterized tests call the skip wrapper.
    # One for when the skip wrapper is called first and doesn't iterate.
    self.assertEqual(3, res.testsRun)
    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.failures, 0)
    # One error from test_other_then_parameterized.
    self.assertLen(res.errors, 1)

  def test_other_decorator_ordering_named(self):
    ts = unittest.makeSuite(self.OtherDecoratorNamed)
    # Verify it generates the test method names from the original test method.
    for test in ts:  # There is only one test.
      ts_attributes = dir(test)
      self.assertIn('test_parameterized_then_other_a', ts_attributes)
      self.assertIn('test_parameterized_then_other_b', ts_attributes)

    res = unittest.TestResult()
    ts.run(res)
    # Two for when the parameterized tests call the skip wrapper.
    # One for when the skip wrapper is called first and doesn't iterate.
    self.assertEqual(3, res.testsRun)
    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.failures, 0)
    # One error from test_other_then_parameterized.
    self.assertLen(res.errors, 1)

  def test_other_decorator_ordering_named_with_dict(self):
    ts = unittest.makeSuite(self.OtherDecoratorNamedWithDict)
    # Verify it generates the test method names from the original test method.
    for test in ts:  # There is only one test.
      ts_attributes = dir(test)
      self.assertIn('test_parameterized_then_other_a', ts_attributes)
      self.assertIn('test_parameterized_then_other_b', ts_attributes)

    res = unittest.TestResult()
    ts.run(res)
    # Two for when the parameterized tests call the skip wrapper.
    # One for when the skip wrapper is called first and doesn't iterate.
    self.assertEqual(3, res.testsRun)
    self.assertFalse(res.wasSuccessful())
    self.assertLen(res.failures, 0)
    # One error from test_other_then_parameterized.
    self.assertLen(res.errors, 1)

  def test_no_test_error_empty_parameters(self):
    with self.assertRaises(parameterized.NoTestsError):

      @parameterized.parameters()
      def test_something():
        pass

      del test_something

  def test_no_test_error_empty_generator(self):
    with self.assertRaises(parameterized.NoTestsError):

      @parameterized.parameters((i for i in []))
      def test_something():
        pass

      del test_something

  def test_unique_descriptive_names(self):

    class RecordSuccessTestsResult(unittest.TestResult):

      def __init__(self, *args, **kwargs):
        super(RecordSuccessTestsResult, self).__init__(*args, **kwargs)
        self.successful_tests = []

      def addSuccess(self, test):
        self.successful_tests.append(test)

    ts = unittest.makeSuite(self.UniqueDescriptiveNamesTest)
    res = RecordSuccessTestsResult()
    ts.run(res)
    self.assertTrue(res.wasSuccessful())
    self.assertEqual(9, res.testsRun)
    test_ids = [test.id() for test in res.successful_tests]
    full_class_name = unittest.util.strclass(self.UniqueDescriptiveNamesTest)
    expected_test_ids = [
        full_class_name + '.test_normal(13)',
        full_class_name + '.test_normal(13) (2)',
        full_class_name + '.test_double_conflict(13)',
        full_class_name + '.test_double_conflict(13) (2)',
        full_class_name + '.test_double_conflict(13) (2) (2)',
        full_class_name + '.test_triple_conflict(13)',
        full_class_name + '.test_triple_conflict(13) (2)',
        full_class_name + '.test_triple_conflict(13) (2) (2)',
        full_class_name + '.test_triple_conflict(13) (3)',
    ]
    self.assertTrue(test_ids)
    self.assertItemsEqual(expected_test_ids, test_ids)

  def test_multi_generators(self):
    ts = unittest.makeSuite(self.MultiGeneratorsTestCase)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful(), msg=str(res.failures))

  def test_named_parameters_reusable(self):
    ts = unittest.makeSuite(self.NamedParametersReusableTestCase)
    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(8, res.testsRun)
    self.assertTrue(res.wasSuccessful(), msg=str(res.failures))

  def test_subclass_inherits_superclass_test_method_ids(self):
    self.assertEqual(
        {'test_name0': "test_name('foo')", 'test_name1': "test_name('bar')"},
        self.SuperclassTestCase._test_method_ids)
    self.assertEqual(
        {'test_name0': "test_name('foo')", 'test_name1': "test_name('bar')"},
        self.SubclassTestCase._test_method_ids)


def _decorate_with_side_effects(func, self):
  self.sideeffect = True
  func(self)


class CoopMetaclassCreationTest(absltest.TestCase):

  class TestBase(absltest.TestCase):

    # This test simulates a metaclass that sets some attribute ('sideeffect')
    # on each member of the class that starts with 'test'. The test code then
    # checks that this attribute exists when the custom metaclass and
    # TestGeneratorMetaclass are combined with cooperative inheritance.

    # The attribute has to be set in the __init__ method of the metaclass,
    # since the TestGeneratorMetaclass already overrides __new__. Only one
    # base metaclass can override __new__, but all can provide custom __init__
    # methods.

    class __metaclass__(type):  # pylint: disable=g-bad-name

      def __init__(cls, name, bases, dct):
        type.__init__(cls, name, bases, dct)
        for member_name, obj in six.iteritems(dct):
          if member_name.startswith('test'):
            setattr(cls, member_name,
                    lambda self, f=obj: _decorate_with_side_effects(f, self))

  class MyParams(parameterized.CoopTestCase(TestBase)):

    @parameterized.parameters(
        (1, 2, 3),
        (4, 5, 9))
    def test_addition(self, op1, op2, result):
      self.assertEqual(result, op1 + op2)

  class MySuite(unittest.TestSuite):
    # Under Python 3.4 the TestCases in the suite's list of tests to run are
    # destroyed and replaced with None after successful execution by default.
    # This disables that behavior.
    _cleanup = False

  def test_successful_execution(self):
    ts = unittest.makeSuite(self.MyParams)

    res = unittest.TestResult()
    ts.run(res)
    self.assertEqual(2, res.testsRun)
    self.assertTrue(res.wasSuccessful())

  def test_metaclass_side_effects(self):
    ts = unittest.makeSuite(self.MyParams, suiteClass=self.MySuite)

    res = unittest.TestResult()
    ts.run(res)
    self.assertTrue(list(ts)[0].sideeffect)


if __name__ == '__main__':
  absltest.main()
