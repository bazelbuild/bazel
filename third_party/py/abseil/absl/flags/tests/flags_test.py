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

"""Tests for absl.flags used as a package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import io
import os
import shutil
import sys
import tempfile
import unittest

from absl import flags
from absl._enum_module import enum
from absl.flags import _exceptions
from absl.flags import _helpers
from absl.flags.tests import module_bar
from absl.flags.tests import module_baz
from absl.flags.tests import module_foo
from absl.testing import absltest
import six


FLAGS = flags.FLAGS


@contextlib.contextmanager
def _use_gnu_getopt(flag_values, use_gnu_get_opt):
  old_use_gnu_get_opt = flag_values.is_gnu_getopt()
  flag_values.set_gnu_getopt(use_gnu_get_opt)
  yield
  flag_values.set_gnu_getopt(old_use_gnu_get_opt)


class FlagDictToArgsTest(absltest.TestCase):

  def test_flatten_google_flag_map(self):
    arg_dict = {
        'week-end': None,
        'estudia': False,
        'trabaja': False,
        'party': True,
        'monday': 'party',
        'score': 42,
        'loadthatstuff': [42, 'hello', 'goodbye'],
    }
    self.assertSameElements(
        (
            '--week-end', '--noestudia', '--notrabaja',
            '--party', '--monday=party', '--score=42',
            '--loadthatstuff=42,hello,goodbye'),
        flags.flag_dict_to_args(arg_dict))


class Fruit(enum.Enum):
  apple = 1
  orange = 2
  APPLE = 3


class EmptyEnum(enum.Enum):
  pass


class FlagsUnitTest(absltest.TestCase):
  """Flags Unit Test."""

  maxDiff = None

  def test_flags(self):
    """Test normal usage with no (expected) errors."""
    # Define flags
    number_test_framework_flags = len(FLAGS)
    repeat_help = 'how many times to repeat (0-5)'
    flags.DEFINE_integer('repeat', 4, repeat_help,
                         lower_bound=0, short_name='r')
    flags.DEFINE_string('name', 'Bob', 'namehelp')
    flags.DEFINE_boolean('debug', 0, 'debughelp')
    flags.DEFINE_boolean('q', 1, 'quiet mode')
    flags.DEFINE_boolean('quack', 0, "superstring of 'q'")
    flags.DEFINE_boolean('noexec', 1, 'boolean flag with no as prefix')
    flags.DEFINE_float('float', 3.14, 'using floats')
    flags.DEFINE_integer('octal', '0o666', 'using octals')
    flags.DEFINE_integer('decimal', '666', 'using decimals')
    flags.DEFINE_integer('hexadecimal', '0x666', 'using hexadecimals')
    flags.DEFINE_integer('x', 3, 'how eXtreme to be')
    flags.DEFINE_integer('l', 0x7fffffff00000000, 'how long to be')
    flags.DEFINE_list('args', 'v=1,"vmodule=a=0,b=2"', 'a list of arguments')
    flags.DEFINE_list('letters', 'a,b,c', 'a list of letters')
    flags.DEFINE_list('numbers', [1, 2, 3], 'a list of numbers')
    flags.DEFINE_enum('kwery', None, ['who', 'what', 'Why', 'where', 'when'],
                      '?')
    flags.DEFINE_enum('sense', None, ['Case', 'case', 'CASE'],
                      '?', case_sensitive=True)
    flags.DEFINE_enum('cases', None, ['UPPER', 'lower', 'Initial', 'Ot_HeR'],
                      '?', case_sensitive=False)
    flags.DEFINE_enum('funny', None, ['Joke', 'ha', 'ha', 'ha', 'ha'],
                      '?', case_sensitive=True)
    flags.DEFINE_enum('blah', None, ['bla', 'Blah', 'BLAH', 'blah'],
                      '?', case_sensitive=False)
    flags.DEFINE_string('only_once', None, 'test only sets this once',
                        allow_overwrite=False)
    flags.DEFINE_string('universe', None, 'test tries to set this three times',
                        allow_overwrite=False)

    # Specify number of flags defined above.  The short_name defined
    # for 'repeat' counts as an extra flag.
    number_defined_flags = 22 + 1
    self.assertEqual(len(FLAGS),
                     number_defined_flags + number_test_framework_flags)

    self.assertEqual(FLAGS.repeat, 4)
    self.assertEqual(FLAGS.name, 'Bob')
    self.assertEqual(FLAGS.debug, 0)
    self.assertEqual(FLAGS.q, 1)
    self.assertEqual(FLAGS.octal, 0o666)
    self.assertEqual(FLAGS.decimal, 666)
    self.assertEqual(FLAGS.hexadecimal, 0x666)
    self.assertEqual(FLAGS.x, 3)
    self.assertEqual(FLAGS.l, 0x7fffffff00000000)
    self.assertEqual(FLAGS.args, ['v=1', 'vmodule=a=0,b=2'])
    self.assertEqual(FLAGS.letters, ['a', 'b', 'c'])
    self.assertEqual(FLAGS.numbers, [1, 2, 3])
    self.assertIsNone(FLAGS.kwery)
    self.assertIsNone(FLAGS.sense)
    self.assertIsNone(FLAGS.cases)
    self.assertIsNone(FLAGS.funny)
    self.assertIsNone(FLAGS.blah)

    flag_values = FLAGS.flag_values_dict()
    self.assertEqual(flag_values['repeat'], 4)
    self.assertEqual(flag_values['name'], 'Bob')
    self.assertEqual(flag_values['debug'], 0)
    self.assertEqual(flag_values['r'], 4)  # Short for repeat.
    self.assertEqual(flag_values['q'], 1)
    self.assertEqual(flag_values['quack'], 0)
    self.assertEqual(flag_values['x'], 3)
    self.assertEqual(flag_values['l'], 0x7fffffff00000000)
    self.assertEqual(flag_values['args'], ['v=1', 'vmodule=a=0,b=2'])
    self.assertEqual(flag_values['letters'], ['a', 'b', 'c'])
    self.assertEqual(flag_values['numbers'], [1, 2, 3])
    self.assertIsNone(flag_values['kwery'])
    self.assertIsNone(flag_values['sense'])
    self.assertIsNone(flag_values['cases'])
    self.assertIsNone(flag_values['funny'])
    self.assertIsNone(flag_values['blah'])

    # Verify string form of defaults
    self.assertEqual(FLAGS['repeat'].default_as_str, "'4'")
    self.assertEqual(FLAGS['name'].default_as_str, "'Bob'")
    self.assertEqual(FLAGS['debug'].default_as_str, "'false'")
    self.assertEqual(FLAGS['q'].default_as_str, "'true'")
    self.assertEqual(FLAGS['quack'].default_as_str, "'false'")
    self.assertEqual(FLAGS['noexec'].default_as_str, "'true'")
    self.assertEqual(FLAGS['x'].default_as_str, "'3'")
    self.assertEqual(FLAGS['l'].default_as_str, "'9223372032559808512'")
    self.assertEqual(FLAGS['args'].default_as_str, '\'v=1,"vmodule=a=0,b=2"\'')
    self.assertEqual(FLAGS['letters'].default_as_str, "'a,b,c'")
    self.assertEqual(FLAGS['numbers'].default_as_str, "'1,2,3'")

    # Verify that the iterator for flags yields all the keys
    keys = list(FLAGS)
    keys.sort()
    reg_flags = list(FLAGS._flags())
    reg_flags.sort()
    self.assertEqual(keys, reg_flags)

    # Parse flags
    # .. empty command line
    argv = ('./program',)
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')

    # .. non-empty command line
    argv = ('./program', '--debug', '--name=Bob', '-q', '--x=8')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['debug'].present, 1)
    FLAGS['debug'].present = 0  # Reset
    self.assertEqual(FLAGS['name'].present, 1)
    FLAGS['name'].present = 0  # Reset
    self.assertEqual(FLAGS['q'].present, 1)
    FLAGS['q'].present = 0  # Reset
    self.assertEqual(FLAGS['x'].present, 1)
    FLAGS['x'].present = 0  # Reset

    # Flags list.
    self.assertEqual(len(FLAGS),
                     number_defined_flags + number_test_framework_flags)
    self.assertIn('name', FLAGS)
    self.assertIn('debug', FLAGS)
    self.assertIn('repeat', FLAGS)
    self.assertIn('r', FLAGS)
    self.assertIn('q', FLAGS)
    self.assertIn('quack', FLAGS)
    self.assertIn('x', FLAGS)
    self.assertIn('l', FLAGS)
    self.assertIn('args', FLAGS)
    self.assertIn('letters', FLAGS)
    self.assertIn('numbers', FLAGS)

    # __contains__
    self.assertIn('name', FLAGS)
    self.assertNotIn('name2', FLAGS)

    # try deleting a flag
    del FLAGS.r
    self.assertEqual(len(FLAGS),
                     number_defined_flags - 1 + number_test_framework_flags)
    self.assertNotIn('r', FLAGS)

    # .. command line with extra stuff
    argv = ('./program', '--debug', '--name=Bob', 'extra')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')
    self.assertEqual(FLAGS['debug'].present, 1)
    FLAGS['debug'].present = 0  # Reset
    self.assertEqual(FLAGS['name'].present, 1)
    FLAGS['name'].present = 0  # Reset

    # Test reset
    argv = ('./program', '--debug')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['debug'].present, 1)
    self.assertTrue(FLAGS['debug'].value)
    FLAGS.unparse_flags()
    self.assertEqual(FLAGS['debug'].present, 0)
    self.assertFalse(FLAGS['debug'].value)

    # Test that reset restores default value when default value is None.
    argv = ('./program', '--kwery=who')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['kwery'].present, 1)
    self.assertEqual(FLAGS['kwery'].value, 'who')
    FLAGS.unparse_flags()
    argv = ('./program', '--kwery=Why')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['kwery'].present, 1)
    self.assertEqual(FLAGS['kwery'].value, 'Why')
    FLAGS.unparse_flags()
    self.assertEqual(FLAGS['kwery'].present, 0)
    self.assertIsNone(FLAGS['kwery'].value)

    # Test case sensitive enum.
    argv = ('./program', '--sense=CASE')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['sense'].present, 1)
    self.assertEqual(FLAGS['sense'].value, 'CASE')
    FLAGS.unparse_flags()
    argv = ('./program', '--sense=Case')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['sense'].present, 1)
    self.assertEqual(FLAGS['sense'].value, 'Case')
    FLAGS.unparse_flags()

    # Test case insensitive enum.
    argv = ('./program', '--cases=upper')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['cases'].present, 1)
    self.assertEqual(FLAGS['cases'].value, 'UPPER')
    FLAGS.unparse_flags()

    # Test case sensitive enum with duplicates.
    argv = ('./program', '--funny=ha')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['funny'].present, 1)
    self.assertEqual(FLAGS['funny'].value, 'ha')
    FLAGS.unparse_flags()

    # Test case insensitive enum with duplicates.
    argv = ('./program', '--blah=bLah')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['blah'].present, 1)
    self.assertEqual(FLAGS['blah'].value, 'Blah')
    FLAGS.unparse_flags()
    argv = ('./program', '--blah=BLAH')
    argv = FLAGS(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(FLAGS['blah'].present, 1)
    self.assertEqual(FLAGS['blah'].value, 'Blah')
    FLAGS.unparse_flags()

    # Test integer argument passing
    argv = ('./program', '--x', '0x12345')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.x, 0x12345)
    self.assertEqual(type(FLAGS.x), int)

    argv = ('./program', '--x', '0x1234567890ABCDEF1234567890ABCDEF')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.x, 0x1234567890ABCDEF1234567890ABCDEF)
    self.assertIsInstance(FLAGS.x, six.integer_types)

    argv = ('./program', '--x', '0o12345')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.x, 0o12345)
    self.assertEqual(type(FLAGS.x), int)

    # Treat 0-prefixed parameters as base-10, not base-8
    argv = ('./program', '--x', '012345')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.x, 12345)
    self.assertEqual(type(FLAGS.x), int)

    argv = ('./program', '--x', '0123459')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.x, 123459)
    self.assertEqual(type(FLAGS.x), int)

    argv = ('./program', '--x', '0x123efg')
    with self.assertRaises(flags.IllegalFlagValueError):
      argv = FLAGS(argv)

    # Test boolean argument parsing
    flags.DEFINE_boolean('test0', None, 'test boolean parsing')
    argv = ('./program', '--notest0')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.test0, 0)

    flags.DEFINE_boolean('test1', None, 'test boolean parsing')
    argv = ('./program', '--test1')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.test1, 1)

    FLAGS.test0 = None
    argv = ('./program', '--test0=false')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.test0, 0)

    FLAGS.test1 = None
    argv = ('./program', '--test1=true')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.test1, 1)

    FLAGS.test0 = None
    argv = ('./program', '--test0=0')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.test0, 0)

    FLAGS.test1 = None
    argv = ('./program', '--test1=1')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.test1, 1)

    # Test booleans that already have 'no' as a prefix
    FLAGS.noexec = None
    argv = ('./program', '--nonoexec', '--name', 'Bob')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.noexec, 0)

    FLAGS.noexec = None
    argv = ('./program', '--name', 'Bob', '--noexec')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.noexec, 1)

    # Test unassigned booleans
    flags.DEFINE_boolean('testnone', None, 'test boolean parsing')
    argv = ('./program',)
    argv = FLAGS(argv)
    self.assertIsNone(FLAGS.testnone)

    # Test get with default
    flags.DEFINE_boolean('testget1', None, 'test parsing with defaults')
    flags.DEFINE_boolean('testget2', None, 'test parsing with defaults')
    flags.DEFINE_boolean('testget3', None, 'test parsing with defaults')
    flags.DEFINE_integer('testget4', None, 'test parsing with defaults')
    argv = ('./program', '--testget1', '--notestget2')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('testget1', 'foo'), 1)
    self.assertEqual(FLAGS.get_flag_value('testget2', 'foo'), 0)
    self.assertEqual(FLAGS.get_flag_value('testget3', 'foo'), 'foo')
    self.assertEqual(FLAGS.get_flag_value('testget4', 'foo'), 'foo')

    # test list code
    lists = [['hello', 'moo', 'boo', '1'],
             []]

    flags.DEFINE_list('testcomma_list', '', 'test comma list parsing')
    flags.DEFINE_spaceseplist('testspace_list', '', 'tests space list parsing')
    flags.DEFINE_spaceseplist(
        'testspace_or_comma_list', '',
        'tests space list parsing with comma compatibility', comma_compat=True)

    for name, sep in (
        ('testcomma_list', ','),
        ('testspace_list', ' '),
        ('testspace_list', '\n'),
        ('testspace_or_comma_list', ' '),
        ('testspace_or_comma_list', '\n'),
        ('testspace_or_comma_list', ',')):
      for lst in lists:
        argv = ('./program', '--%s=%s' % (name, sep.join(lst)))
        argv = FLAGS(argv)
        self.assertEqual(getattr(FLAGS, name), lst)

    # Test help text
    flags_help = str(FLAGS)
    self.assertNotEqual(flags_help.find('repeat'), -1,
                        'cannot find flag in help')
    self.assertNotEqual(flags_help.find(repeat_help), -1,
                        'cannot find help string in help')

    # Test flag specified twice
    argv = ('./program', '--repeat=4', '--repeat=2', '--debug', '--nodebug')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('repeat', None), 2)
    self.assertEqual(FLAGS.get_flag_value('debug', None), 0)

    # Test MultiFlag with single default value
    flags.DEFINE_multi_string('s_str', 'sing1',
                              'string option that can occur multiple times',
                              short_name='s')
    self.assertEqual(FLAGS.get_flag_value('s_str', None), ['sing1'])

    # Test MultiFlag with list of default values
    multi_string_defs = ['def1', 'def2']
    flags.DEFINE_multi_string('m_str', multi_string_defs,
                              'string option that can occur multiple times',
                              short_name='m')
    self.assertEqual(FLAGS.get_flag_value('m_str', None), multi_string_defs)

    # Test flag specified multiple times with a MultiFlag
    argv = ('./program', '--m_str=str1', '-m', 'str2')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('m_str', None), ['str1', 'str2'])

    # A flag with allow_overwrite set to False should behave normally when it
    # is only specified once
    argv = ('./program', '--only_once=singlevalue')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('only_once', None), 'singlevalue')

    # A flag with allow_overwrite set to False should complain when it is
    # specified more than once
    argv = ('./program', '--universe=ptolemaic',
            '--universe=copernicean', '--universe=euclidean')
    self.assertRaisesWithLiteralMatch(
        flags.IllegalFlagValueError,
        'flag --universe=copernicean: already defined as ptolemaic',
        FLAGS,
        argv)

    # Test single-letter flags; should support both single and double dash
    argv = ('./program', '-q')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('q', None), 1)

    argv = ('./program', '--q', '--x', '9', '--noquack')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('q', None), 1)
    self.assertEqual(FLAGS.get_flag_value('x', None), 9)
    self.assertEqual(FLAGS.get_flag_value('quack', None), 0)

    argv = ('./program', '--noq', '--x=10', '--quack')
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('q', None), 0)
    self.assertEqual(FLAGS.get_flag_value('x', None), 10)
    self.assertEqual(FLAGS.get_flag_value('quack', None), 1)

    ####################################
    # Test flag serialization code:

    old_testcomma_list = FLAGS.testcomma_list
    old_testspace_list = FLAGS.testspace_list
    old_testspace_or_comma_list = FLAGS.testspace_or_comma_list

    argv = ('./program',
            FLAGS['test0'].serialize(),
            FLAGS['test1'].serialize(),
            FLAGS['s_str'].serialize())

    argv = FLAGS(argv)
    self.assertEqual(FLAGS['test0'].serialize(), '--notest0')
    self.assertEqual(FLAGS['test1'].serialize(), '--test1')
    self.assertEqual(FLAGS['s_str'].serialize(), '--s_str=sing1')

    self.assertEqual(FLAGS['testnone'].serialize(), '')

    testcomma_list1 = ['aa', 'bb']
    testspace_list1 = ['aa', 'bb', 'cc']
    testspace_or_comma_list1 = ['aa', 'bb', 'cc', 'dd']
    FLAGS.testcomma_list = list(testcomma_list1)
    FLAGS.testspace_list = list(testspace_list1)
    FLAGS.testspace_or_comma_list = list(testspace_or_comma_list1)
    argv = ('./program',
            FLAGS['testcomma_list'].serialize(),
            FLAGS['testspace_list'].serialize(),
            FLAGS['testspace_or_comma_list'].serialize())
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.testcomma_list, testcomma_list1)
    self.assertEqual(FLAGS.testspace_list, testspace_list1)
    self.assertEqual(FLAGS.testspace_or_comma_list, testspace_or_comma_list1)

    testcomma_list1 = ['aa some spaces', 'bb']
    testspace_list1 = ['aa', 'bb,some,commas,', 'cc']
    testspace_or_comma_list1 = ['aa', 'bb,some,commas,', 'cc']
    FLAGS.testcomma_list = list(testcomma_list1)
    FLAGS.testspace_list = list(testspace_list1)
    FLAGS.testspace_or_comma_list = list(testspace_or_comma_list1)
    argv = ('./program',
            FLAGS['testcomma_list'].serialize(),
            FLAGS['testspace_list'].serialize(),
            FLAGS['testspace_or_comma_list'].serialize())
    argv = FLAGS(argv)
    self.assertEqual(FLAGS.testcomma_list, testcomma_list1)
    self.assertEqual(FLAGS.testspace_list, testspace_list1)
    # We don't expect idempotency when commas are placed in an item value and
    # comma_compat is enabled.
    self.assertEqual(FLAGS.testspace_or_comma_list,
                     ['aa', 'bb', 'some', 'commas', 'cc'])

    FLAGS.testcomma_list = old_testcomma_list
    FLAGS.testspace_list = old_testspace_list
    FLAGS.testspace_or_comma_list = old_testspace_or_comma_list

    ####################################
    # Test flag-update:

    def args_list():
      # Exclude flags that have different default values based on the
      # environment.
      flags_to_exclude = {'log_dir', 'test_srcdir', 'test_tmpdir'}
      flagnames = set(FLAGS) - flags_to_exclude

      nonbool_flags = ['--%s %s' % (name, FLAGS.get_flag_value(name, None))
                       for name in flagnames
                       if not isinstance(FLAGS[name], flags.BooleanFlag)]

      truebool_flags = ['--%s' % (name)
                        for name in flagnames
                        if isinstance(FLAGS[name], flags.BooleanFlag) and
                        FLAGS.get_flag_value(name, None)]
      falsebool_flags = ['--no%s' % (name)
                         for name in flagnames
                         if isinstance(FLAGS[name], flags.BooleanFlag) and
                         not FLAGS.get_flag_value(name, None)]
      all_flags = nonbool_flags + truebool_flags + falsebool_flags
      all_flags.sort()
      return all_flags

    argv = ('./program', '--repeat=3', '--name=giants', '--nodebug')

    FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('repeat', None), 3)
    self.assertEqual(FLAGS.get_flag_value('name', None), 'giants')
    self.assertEqual(FLAGS.get_flag_value('debug', None), 0)
    self.assertListEqual([
        '--alsologtostderr',
        "--args ['v=1', 'vmodule=a=0,b=2']",
        '--blah None',
        '--cases None',
        '--decimal 666',
        '--float 3.14',
        '--funny None',
        '--hexadecimal 1638',
        '--kwery None',
        '--l 9223372032559808512',
        "--letters ['a', 'b', 'c']",
        "--m ['str1', 'str2']",
        "--m_str ['str1', 'str2']",
        '--name giants',
        '--no?',
        '--nodebug',
        '--noexec',
        '--nohelp',
        '--nohelpfull',
        '--nohelpshort',
        '--nohelpxml',
        '--nologtostderr',
        '--noonly_check_args',
        '--nopdb_post_mortem',
        '--noq',
        '--norun_with_pdb',
        '--norun_with_profiling',
        '--notest0',
        '--notestget2',
        '--notestget3',
        '--notestnone',
        '--numbers [1, 2, 3]',
        '--octal 438',
        '--only_once singlevalue',
        '--profile_file None',
        '--quack',
        '--repeat 3',
        "--s ['sing1']",
        "--s_str ['sing1']",
        '--sense None',
        '--showprefixforinfo',
        '--stderrthreshold fatal',
        '--test1',
        '--test_random_seed 301',
        '--test_randomize_ordering_seed None',
        '--testcomma_list []',
        '--testget1',
        '--testget4 None',
        '--testspace_list []',
        '--testspace_or_comma_list []',
        '--tmod_baz_x',
        '--universe ptolemaic',
        '--use_cprofile_for_profiling',
        '--v -1',
        '--verbosity -1',
        '--x 10',
        '--xml_output_file ',
    ], args_list())

    argv = ('./program', '--debug', '--m_str=upd1', '-s', 'upd2')
    FLAGS(argv)
    self.assertEqual(FLAGS.get_flag_value('repeat', None), 3)
    self.assertEqual(FLAGS.get_flag_value('name', None), 'giants')
    self.assertEqual(FLAGS.get_flag_value('debug', None), 1)

    # items appended to existing non-default value lists for --m/--m_str
    # new value overwrites default value (not appended to it) for --s/--s_str
    self.assertListEqual([
        '--alsologtostderr',
        "--args ['v=1', 'vmodule=a=0,b=2']",
        '--blah None',
        '--cases None',
        '--debug',
        '--decimal 666',
        '--float 3.14',
        '--funny None',
        '--hexadecimal 1638',
        '--kwery None',
        '--l 9223372032559808512',
        "--letters ['a', 'b', 'c']",
        "--m ['str1', 'str2', 'upd1']",
        "--m_str ['str1', 'str2', 'upd1']",
        '--name giants',
        '--no?',
        '--noexec',
        '--nohelp',
        '--nohelpfull',
        '--nohelpshort',
        '--nohelpxml',
        '--nologtostderr',
        '--noonly_check_args',
        '--nopdb_post_mortem',
        '--noq',
        '--norun_with_pdb',
        '--norun_with_profiling',
        '--notest0',
        '--notestget2',
        '--notestget3',
        '--notestnone',
        '--numbers [1, 2, 3]',
        '--octal 438',
        '--only_once singlevalue',
        '--profile_file None',
        '--quack',
        '--repeat 3',
        "--s ['sing1', 'upd2']",
        "--s_str ['sing1', 'upd2']",
        '--sense None',
        '--showprefixforinfo',
        '--stderrthreshold fatal',
        '--test1',
        '--test_random_seed 301',
        '--test_randomize_ordering_seed None',
        '--testcomma_list []',
        '--testget1',
        '--testget4 None',
        '--testspace_list []',
        '--testspace_or_comma_list []',
        '--tmod_baz_x',
        '--universe ptolemaic',
        '--use_cprofile_for_profiling',
        '--v -1',
        '--verbosity -1',
        '--x 10',
        '--xml_output_file ',
    ], args_list())

    ####################################
    # Test all kind of error conditions.

    # Argument not in enum exception
    argv = ('./program', '--kwery=WHEN')
    self.assertRaises(flags.IllegalFlagValueError, FLAGS, argv)
    argv = ('./program', '--kwery=why')
    self.assertRaises(flags.IllegalFlagValueError, FLAGS, argv)

    # Duplicate flag detection
    with self.assertRaises(flags.DuplicateFlagError):
      flags.DEFINE_boolean('run', 0, 'runhelp', short_name='q')

    # Duplicate short flag detection
    with self.assertRaisesRegex(
        flags.DuplicateFlagError,
        r"The flag 'z' is defined twice\. .*First from.*, Second from"):
      flags.DEFINE_boolean('zoom1', 0, 'runhelp z1', short_name='z')
      flags.DEFINE_boolean('zoom2', 0, 'runhelp z2', short_name='z')
      raise AssertionError('duplicate short flag detection failed')

    # Duplicate mixed flag detection
    with self.assertRaisesRegex(
        flags.DuplicateFlagError,
        r"The flag 's' is defined twice\. .*First from.*, Second from"):
      flags.DEFINE_boolean('short1', 0, 'runhelp s1', short_name='s')
      flags.DEFINE_boolean('s', 0, 'runhelp s2')

    # Check that duplicate flag detection detects definition sites
    # correctly.
    flagnames = ['repeated']
    original_flags = flags.FlagValues()
    flags.DEFINE_boolean(flagnames[0], False, 'Flag about to be repeated.',
                         flag_values=original_flags)
    duplicate_flags = module_foo.duplicate_flags(flagnames)
    with self.assertRaisesRegex(flags.DuplicateFlagError,
                                'flags_test.*module_foo'):
      original_flags.append_flag_values(duplicate_flags)

    # Make sure allow_override works
    try:
      flags.DEFINE_boolean('dup1', 0, 'runhelp d11', short_name='u',
                           allow_override=0)
      flag = FLAGS._flags()['dup1']
      self.assertEqual(flag.default, 0)

      flags.DEFINE_boolean('dup1', 1, 'runhelp d12', short_name='u',
                           allow_override=1)
      flag = FLAGS._flags()['dup1']
      self.assertEqual(flag.default, 1)
    except flags.DuplicateFlagError:
      raise AssertionError('allow_override did not permit a flag duplication')

    # Make sure allow_override works
    try:
      flags.DEFINE_boolean('dup2', 0, 'runhelp d21', short_name='u',
                           allow_override=1)
      flag = FLAGS._flags()['dup2']
      self.assertEqual(flag.default, 0)

      flags.DEFINE_boolean('dup2', 1, 'runhelp d22', short_name='u',
                           allow_override=0)
      flag = FLAGS._flags()['dup2']
      self.assertEqual(flag.default, 1)
    except flags.DuplicateFlagError:
      raise AssertionError('allow_override did not permit a flag duplication')

    # Make sure that re-importing a module does not cause a DuplicateFlagError
    # to be raised.
    try:
      sys.modules.pop(
          'absl.flags.tests.module_baz')
      import absl.flags.tests.module_baz
      del absl
    except flags.DuplicateFlagError:
      raise AssertionError('Module reimport caused flag duplication error')

    # Make sure that when we override, the help string gets updated correctly
    flags.DEFINE_boolean('dup3', 0, 'runhelp d31', short_name='u',
                         allow_override=1)
    flags.DEFINE_boolean('dup3', 1, 'runhelp d32', short_name='u',
                         allow_override=1)
    self.assertEqual(str(FLAGS).find('runhelp d31'), -1)
    self.assertNotEqual(str(FLAGS).find('runhelp d32'), -1)

    # Make sure append_flag_values works
    new_flags = flags.FlagValues()
    flags.DEFINE_boolean('new1', 0, 'runhelp n1', flag_values=new_flags)
    flags.DEFINE_boolean('new2', 0, 'runhelp n2', flag_values=new_flags)
    self.assertEqual(len(new_flags._flags()), 2)
    old_len = len(FLAGS._flags())
    FLAGS.append_flag_values(new_flags)
    self.assertEqual(len(FLAGS._flags()) - old_len, 2)
    self.assertEqual('new1' in FLAGS._flags(), True)
    self.assertEqual('new2' in FLAGS._flags(), True)

    # Then test that removing those flags works
    FLAGS.remove_flag_values(new_flags)
    self.assertEqual(len(FLAGS._flags()), old_len)
    self.assertFalse('new1' in FLAGS._flags())
    self.assertFalse('new2' in FLAGS._flags())

    # Make sure append_flag_values works with flags with shortnames.
    new_flags = flags.FlagValues()
    flags.DEFINE_boolean('new3', 0, 'runhelp n3', flag_values=new_flags)
    flags.DEFINE_boolean('new4', 0, 'runhelp n4', flag_values=new_flags,
                         short_name='n4')
    self.assertEqual(len(new_flags._flags()), 3)
    old_len = len(FLAGS._flags())
    FLAGS.append_flag_values(new_flags)
    self.assertEqual(len(FLAGS._flags()) - old_len, 3)
    self.assertIn('new3', FLAGS._flags())
    self.assertIn('new4', FLAGS._flags())
    self.assertIn('n4', FLAGS._flags())
    self.assertEqual(FLAGS._flags()['n4'], FLAGS._flags()['new4'])

    # Then test removing them
    FLAGS.remove_flag_values(new_flags)
    self.assertEqual(len(FLAGS._flags()), old_len)
    self.assertFalse('new3' in FLAGS._flags())
    self.assertFalse('new4' in FLAGS._flags())
    self.assertFalse('n4' in FLAGS._flags())

    # Make sure append_flag_values fails on duplicates
    flags.DEFINE_boolean('dup4', 0, 'runhelp d41')
    new_flags = flags.FlagValues()
    flags.DEFINE_boolean('dup4', 0, 'runhelp d42', flag_values=new_flags)
    with self.assertRaises(flags.DuplicateFlagError):
      FLAGS.append_flag_values(new_flags)

    # Integer out of bounds
    with self.assertRaises(flags.IllegalFlagValueError):
      argv = ('./program', '--repeat=-4')
      FLAGS(argv)

    # Non-integer
    with self.assertRaises(flags.IllegalFlagValueError):
      argv = ('./program', '--repeat=2.5')
      FLAGS(argv)

    # Missing required arugment
    with self.assertRaises(flags.Error):
      argv = ('./program', '--name')
      FLAGS(argv)

    # Non-boolean arguments for boolean
    with self.assertRaises(flags.IllegalFlagValueError):
      argv = ('./program', '--debug=goofup')
      FLAGS(argv)

    with self.assertRaises(flags.IllegalFlagValueError):
      argv = ('./program', '--debug=42')
      FLAGS(argv)

    # Non-numeric argument for integer flag --repeat
    with self.assertRaises(flags.IllegalFlagValueError):
      argv = ('./program', '--repeat', 'Bob', 'extra')
      FLAGS(argv)

    # Aliases of existing flags
    with self.assertRaises(flags.UnrecognizedFlagError):
      flags.DEFINE_alias('alias_not_a_flag', 'not_a_flag')

    # Programmtically modify alias and aliased flag
    flags.DEFINE_alias('alias_octal', 'octal')
    FLAGS.octal = 0o2222
    self.assertEqual(0o2222, FLAGS.octal)
    self.assertEqual(0o2222, FLAGS.alias_octal)
    FLAGS.alias_octal = 0o4444
    self.assertEqual(0o4444, FLAGS.octal)
    self.assertEqual(0o4444, FLAGS.alias_octal)

    # Setting alias preserves the default of the original
    flags.DEFINE_alias('alias_name', 'name')
    flags.DEFINE_alias('alias_debug', 'debug')
    flags.DEFINE_alias('alias_decimal', 'decimal')
    flags.DEFINE_alias('alias_float', 'float')
    flags.DEFINE_alias('alias_letters', 'letters')
    self.assertEqual(FLAGS['name'].default, FLAGS.alias_name)
    self.assertEqual(FLAGS['debug'].default, FLAGS.alias_debug)
    self.assertEqual(
        int(FLAGS['decimal'].default), FLAGS.alias_decimal)
    self.assertEqual(
        float(FLAGS['float'].default), FLAGS.alias_float)
    self.assertSameElements(
        FLAGS['letters'].default, FLAGS.alias_letters)

    # Original flags set on comand line
    argv = ('./program',
            '--name=Martin',
            '--debug=True',
            '--decimal=777',
            '--letters=x,y,z')
    FLAGS(argv)
    self.assertEqual('Martin', FLAGS.name)
    self.assertEqual('Martin', FLAGS.alias_name)
    self.assertTrue(FLAGS.debug)
    self.assertTrue(FLAGS.alias_debug)
    self.assertEqual(777, FLAGS.decimal)
    self.assertEqual(777, FLAGS.alias_decimal)
    self.assertSameElements(['x', 'y', 'z'], FLAGS.letters)
    self.assertSameElements(['x', 'y', 'z'], FLAGS.alias_letters)

    # Alias flags set on command line
    argv = ('./program',
            '--alias_name=Auston',
            '--alias_debug=False',
            '--alias_decimal=888',
            '--alias_letters=l,m,n')
    FLAGS(argv)
    self.assertEqual('Auston', FLAGS.name)
    self.assertEqual('Auston', FLAGS.alias_name)
    self.assertFalse(FLAGS.debug)
    self.assertFalse(FLAGS.alias_debug)
    self.assertEqual(888, FLAGS.decimal)
    self.assertEqual(888, FLAGS.alias_decimal)
    self.assertSameElements(['l', 'm', 'n'], FLAGS.letters)
    self.assertSameElements(['l', 'm', 'n'], FLAGS.alias_letters)

    # Make sure importing a module does not change flag value parsed
    # from commandline.
    flags.DEFINE_integer('dup5', 1, 'runhelp d51', short_name='u5',
                         allow_override=0)
    self.assertEqual(FLAGS.dup5, 1)
    self.assertEqual(FLAGS.dup5, 1)
    argv = ('./program', '--dup5=3')
    FLAGS(argv)
    self.assertEqual(FLAGS.dup5, 3)
    flags.DEFINE_integer('dup5', 2, 'runhelp d52', short_name='u5',
                         allow_override=1)
    self.assertEqual(FLAGS.dup5, 3)

    # Make sure importing a module does not change user defined flag value.
    flags.DEFINE_integer('dup6', 1, 'runhelp d61', short_name='u6',
                         allow_override=0)
    self.assertEqual(FLAGS.dup6, 1)
    FLAGS.dup6 = 3
    self.assertEqual(FLAGS.dup6, 3)
    flags.DEFINE_integer('dup6', 2, 'runhelp d62', short_name='u6',
                         allow_override=1)
    self.assertEqual(FLAGS.dup6, 3)

    # Make sure importing a module does not change user defined flag value
    # even if it is the 'default' value.
    flags.DEFINE_integer('dup7', 1, 'runhelp d71', short_name='u7',
                         allow_override=0)
    self.assertEqual(FLAGS.dup7, 1)
    FLAGS.dup7 = 1
    self.assertEqual(FLAGS.dup7, 1)
    flags.DEFINE_integer('dup7', 2, 'runhelp d72', short_name='u7',
                         allow_override=1)
    self.assertEqual(FLAGS.dup7, 1)

    # Test module_help().
    helpstr = FLAGS.module_help(module_baz)

    expected_help = '\n' + module_baz.__name__ + ':' + """
  --[no]tmod_baz_x: Boolean flag.
    (default: 'true')"""

    self.assertMultiLineEqual(expected_help, helpstr)

    # Test main_module_help().  This must be part of test_flags because
    # it depends on dup1/2/3/etc being introduced first.
    helpstr = FLAGS.main_module_help()

    expected_help = '\n' + sys.argv[0] + ':' + """
  --[no]alias_debug: Alias for --debug.
    (default: 'false')
  --alias_decimal: Alias for --decimal.
    (default: '666')
  --alias_float: Alias for --float.
    (default: '3.14')
  --alias_letters: Alias for --letters.
    (default: 'a,b,c')
  --alias_name: Alias for --name.
    (default: 'Bob')
  --alias_octal: Alias for --octal.
    (default: '438')
  --args: a list of arguments
    (default: 'v=1,"vmodule=a=0,b=2"')
    (a comma separated list)
  --blah: <bla|Blah|BLAH|blah>: ?
  --cases: <UPPER|lower|Initial|Ot_HeR>: ?
  --[no]debug: debughelp
    (default: 'false')
  --decimal: using decimals
    (default: '666')
    (an integer)
  -u,--[no]dup1: runhelp d12
    (default: 'true')
  -u,--[no]dup2: runhelp d22
    (default: 'true')
  -u,--[no]dup3: runhelp d32
    (default: 'true')
  --[no]dup4: runhelp d41
    (default: 'false')
  -u5,--dup5: runhelp d51
    (default: '1')
    (an integer)
  -u6,--dup6: runhelp d61
    (default: '1')
    (an integer)
  -u7,--dup7: runhelp d71
    (default: '1')
    (an integer)
  --float: using floats
    (default: '3.14')
    (a number)
  --funny: <Joke|ha|ha|ha|ha>: ?
  --hexadecimal: using hexadecimals
    (default: '1638')
    (an integer)
  --kwery: <who|what|Why|where|when>: ?
  --l: how long to be
    (default: '9223372032559808512')
    (an integer)
  --letters: a list of letters
    (default: 'a,b,c')
    (a comma separated list)
  -m,--m_str: string option that can occur multiple times;
    repeat this option to specify a list of values
    (default: "['def1', 'def2']")
  --name: namehelp
    (default: 'Bob')
  --[no]noexec: boolean flag with no as prefix
    (default: 'true')
  --numbers: a list of numbers
    (default: '1,2,3')
    (a comma separated list)
  --octal: using octals
    (default: '438')
    (an integer)
  --only_once: test only sets this once
  --[no]q: quiet mode
    (default: 'true')
  --[no]quack: superstring of 'q'
    (default: 'false')
  -r,--repeat: how many times to repeat (0-5)
    (default: '4')
    (a non-negative integer)
  -s,--s_str: string option that can occur multiple times;
    repeat this option to specify a list of values
    (default: "['sing1']")
  --sense: <Case|case|CASE>: ?
  --[no]test0: test boolean parsing
  --[no]test1: test boolean parsing
  --testcomma_list: test comma list parsing
    (default: '')
    (a comma separated list)
  --[no]testget1: test parsing with defaults
  --[no]testget2: test parsing with defaults
  --[no]testget3: test parsing with defaults
  --testget4: test parsing with defaults
    (an integer)
  --[no]testnone: test boolean parsing
  --testspace_list: tests space list parsing
    (default: '')
    (a whitespace separated list)
  --testspace_or_comma_list: tests space list parsing with comma compatibility
    (default: '')
    (a whitespace or comma separated list)
  --universe: test tries to set this three times
  --x: how eXtreme to be
    (default: '3')
    (an integer)
  -z,--[no]zoom1: runhelp z1
    (default: 'false')"""

    self.assertMultiLineEqual(expected_help, helpstr)

  def test_string_flag_with_wrong_type(self):
    fv = flags.FlagValues()
    with self.assertRaises(flags.IllegalFlagValueError):
      flags.DEFINE_string('name', False, 'help', flag_values=fv)
    with self.assertRaises(flags.IllegalFlagValueError):
      flags.DEFINE_string('name2', 0, 'help', flag_values=fv)

  def test_integer_flag_with_wrong_type(self):
    fv = flags.FlagValues()
    with self.assertRaises(flags.IllegalFlagValueError):
      flags.DEFINE_integer('name', 1e2, 'help', flag_values=fv)
    with self.assertRaises(flags.IllegalFlagValueError):
      flags.DEFINE_integer('name', [], 'help', flag_values=fv)
    with self.assertRaises(flags.IllegalFlagValueError):
      flags.DEFINE_integer('name', False, 'help', flag_values=fv)

  def test_float_flag_with_wrong_type(self):
    fv = flags.FlagValues()
    with self.assertRaises(flags.IllegalFlagValueError):
      flags.DEFINE_float('name', False, 'help', flag_values=fv)

  def test_enum_flag_with_empty_values(self):
    fv = flags.FlagValues()
    with self.assertRaises(ValueError):
      flags.DEFINE_enum('fruit', None, [], 'help', flag_values=fv)

  def test_define_enum_class_flag(self):
    fv = flags.FlagValues()
    flags.DEFINE_enum_class('fruit', None, Fruit, '?', flag_values=fv)
    fv.mark_as_parsed()

    self.assertIsNone(fv.fruit)

  def test_parse_enum_class_flag(self):
    fv = flags.FlagValues()
    flags.DEFINE_enum_class('fruit', None, Fruit, '?', flag_values=fv)

    argv = ('./program', '--fruit=apple')
    argv = fv(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(fv['fruit'].present, 1)
    self.assertEqual(fv['fruit'].value, Fruit.apple)
    fv.unparse_flags()
    argv = ('./program', '--fruit=APPLE')
    argv = fv(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(fv['fruit'].present, 1)
    self.assertEqual(fv['fruit'].value, Fruit.APPLE)
    fv.unparse_flags()

  def test_enum_class_flag_help_message(self):
    fv = flags.FlagValues()
    flags.DEFINE_enum_class('fruit', None, Fruit, '?', flag_values=fv)

    helpstr = fv.main_module_help()
    expected_help = '\n%s:\n  --fruit: <apple|orange|APPLE>: ?' % sys.argv[0]

    self.assertEqual(helpstr, expected_help)

  def test_enum_class_flag_with_wrong_default_value_type(self):
    fv = flags.FlagValues()
    with self.assertRaises(_exceptions.IllegalFlagValueError):
      flags.DEFINE_enum_class('fruit', 1, Fruit, 'help',
                              flag_values=fv)

  def test_enum_class_flag_requires_enum_class(self):
    fv = flags.FlagValues()
    with self.assertRaises(TypeError):
      flags.DEFINE_enum_class('fruit', None, ['apple', 'orange'], 'help',
                              flag_values=fv)

  def test_enum_class_flag_requires_non_empty_enum_class(self):
    fv = flags.FlagValues()
    with self.assertRaises(ValueError):
      flags.DEFINE_enum_class('empty', None, EmptyEnum, 'help',
                              flag_values=fv)


class MultiNumericalFlagsTest(absltest.TestCase):

  def test_multi_numerical_flags(self):
    """Test multi_int and multi_float flags."""

    int_defaults = [77, 88]
    flags.DEFINE_multi_integer('m_int', int_defaults,
                               'integer option that can occur multiple times',
                               short_name='mi')
    self.assertListEqual(FLAGS.get_flag_value('m_int', None), int_defaults)
    argv = ('./program', '--m_int=-99', '--mi=101')
    FLAGS(argv)
    self.assertListEqual(FLAGS.get_flag_value('m_int', None), [-99, 101])

    float_defaults = [2.2, 3]
    flags.DEFINE_multi_float('m_float', float_defaults,
                             'float option that can occur multiple times',
                             short_name='mf')
    for (expected, actual) in zip(
        float_defaults, FLAGS.get_flag_value('m_float', None)):
      self.assertAlmostEqual(expected, actual)
    argv = ('./program', '--m_float=-17', '--mf=2.78e9')
    FLAGS(argv)
    expected_floats = [-17.0, 2.78e9]
    for (expected, actual) in zip(
        expected_floats, FLAGS.get_flag_value('m_float', None)):
      self.assertAlmostEqual(expected, actual)

  def test_multi_numerical_with_tuples(self):
    """Verify multi_int/float accept tuples as default values."""
    flags.DEFINE_multi_integer(
        'm_int_tuple',
        (77, 88),
        'integer option that can occur multiple times',
        short_name='mi_tuple')
    self.assertListEqual(FLAGS.get_flag_value('m_int_tuple', None), [77, 88])

    dict_with_float_keys = {2.2: 'hello', 3: 'happy'}
    float_defaults = dict_with_float_keys.keys()
    flags.DEFINE_multi_float(
        'm_float_tuple',
        float_defaults,
        'float option that can occur multiple times',
        short_name='mf_tuple')
    for (expected, actual) in zip(float_defaults,
                                  FLAGS.get_flag_value('m_float_tuple', None)):
      self.assertAlmostEqual(expected, actual)

  def test_single_value_default(self):
    """Test multi_int and multi_float flags with a single default value."""
    int_default = 77
    flags.DEFINE_multi_integer('m_int1', int_default,
                               'integer option that can occur multiple times')
    self.assertListEqual(FLAGS.get_flag_value('m_int1', None), [int_default])

    float_default = 2.2
    flags.DEFINE_multi_float('m_float1', float_default,
                             'float option that can occur multiple times')
    actual = FLAGS.get_flag_value('m_float1', None)
    self.assertEqual(1, len(actual))
    self.assertAlmostEqual(actual[0], float_default)

  def test_bad_multi_numerical_flags(self):
    """Test multi_int and multi_float flags with non-parseable values."""

    # Test non-parseable defaults.
    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r"flag --m_int2=abc: invalid literal for int\(\) with base 10: 'abc'",
        flags.DEFINE_multi_integer, 'm_int2', ['abc'], 'desc')

    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r'flag --m_float2=abc: '
        r'(invalid literal for float\(\)|could not convert string to float): '
        r"'?abc'?",
        flags.DEFINE_multi_float, 'm_float2', ['abc'], 'desc')

    # Test non-parseable command line values.
    flags.DEFINE_multi_integer('m_int2', '77',
                               'integer option that can occur multiple times')
    argv = ('./program', '--m_int2=def')
    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r"flag --m_int2=def: invalid literal for int\(\) with base 10: 'def'",
        FLAGS, argv)

    flags.DEFINE_multi_float('m_float2', 2.2,
                             'float option that can occur multiple times')
    argv = ('./program', '--m_float2=def')
    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r'flag --m_float2=def: '
        r'(invalid literal for float\(\)|could not convert string to float): '
        r"'?def'?",
        FLAGS, argv)


class MultiEnumFlagsTest(absltest.TestCase):

  def test_multi_enum_flags(self):
    """Test multi_enum flags."""

    enum_defaults = ['FOO', 'BAZ']
    flags.DEFINE_multi_enum('m_enum', enum_defaults,
                            ['FOO', 'BAR', 'BAZ', 'WHOOSH'],
                            'Enum option that can occur multiple times',
                            short_name='me')
    self.assertListEqual(FLAGS.get_flag_value('m_enum', None), enum_defaults)
    argv = ('./program', '--m_enum=WHOOSH', '--me=FOO')
    FLAGS(argv)
    self.assertListEqual(
        FLAGS.get_flag_value('m_enum', None), ['WHOOSH', 'FOO'])

  def test_single_value_default(self):
    """Test multi_enum flags with a single default value."""
    enum_default = 'FOO'
    flags.DEFINE_multi_enum('m_enum1', enum_default,
                            ['FOO', 'BAR', 'BAZ', 'WHOOSH'],
                            'enum option that can occur multiple times')
    self.assertListEqual(FLAGS.get_flag_value('m_enum1', None), [enum_default])

  def test_case_sensitivity(self):
    """Test case sensitivity of multi_enum flag."""
    # Test case insensitive enum.
    flags.DEFINE_multi_enum('m_enum2', ['whoosh'],
                            ['FOO', 'BAR', 'BAZ', 'WHOOSH'],
                            'Enum option that can occur multiple times',
                            short_name='me2',
                            case_sensitive=False)
    argv = ('./program', '--m_enum2=bar', '--me2=fOo')
    FLAGS(argv)
    self.assertListEqual(FLAGS.get_flag_value('m_enum2', None), ['BAR', 'FOO'])

    # Test case sensitive enum.
    flags.DEFINE_multi_enum('m_enum3', ['BAR'],
                            ['FOO', 'BAR', 'BAZ', 'WHOOSH'],
                            'Enum option that can occur multiple times',
                            short_name='me3',
                            case_sensitive=True)
    argv = ('./program', '--m_enum3=bar', '--me3=fOo')
    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r'flag --m_enum3=invalid: value should be one of <FOO|BAR|BAZ|WHOOSH>',
        FLAGS, argv)

  def test_bad_multi_enum_flags(self):
    """Test multi_enum with invalid values."""

    # Test defaults that are not in the permitted list of enums.
    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r'flag --m_enum=INVALID: value should be one of <FOO|BAR|BAZ>',
        flags.DEFINE_multi_enum, 'm_enum', ['INVALID'],
        ['FOO', 'BAR', 'BAZ'], 'desc')

    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r'flag --m_enum=1234: value should be one of <FOO|BAR|BAZ>',
        flags.DEFINE_multi_enum, 'm_enum2', [1234],
        ['FOO', 'BAR', 'BAZ'], 'desc')

    # Test command-line values that are not in the permitted list of enums.
    flags.DEFINE_multi_enum('m_enum4', 'FOO',
                            ['FOO', 'BAR', 'BAZ'],
                            'enum option that can occur multiple times')
    argv = ('./program', '--m_enum4=INVALID')
    self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        r'flag --m_enum4=invalid: value should be one of <FOO|BAR|BAZ>',
        FLAGS, argv)


class MultiEnumClassFlagsTest(absltest.TestCase):

  def test_define_results_in_registered_flag_with_none(self):
    fv = flags.FlagValues()
    enum_defaults = None
    flags.DEFINE_multi_enum_class('fruit',
                                  enum_defaults, Fruit,
                                  'Enum option that can occur multiple times',
                                  flag_values=fv)
    fv.mark_as_parsed()

    self.assertIsNone(fv.fruit)

  def test_define_results_in_registered_flag_with_string(self):
    fv = flags.FlagValues()
    enum_defaults = 'apple'
    flags.DEFINE_multi_enum_class('fruit',
                                  enum_defaults, Fruit,
                                  'Enum option that can occur multiple times',
                                  flag_values=fv)
    fv.mark_as_parsed()

    self.assertListEqual(fv.fruit, [Fruit.apple])

  def test_define_results_in_registered_flag_with_enum(self):
    fv = flags.FlagValues()
    enum_defaults = Fruit.APPLE
    flags.DEFINE_multi_enum_class('fruit',
                                  enum_defaults, Fruit,
                                  'Enum option that can occur multiple times',
                                  flag_values=fv)
    fv.mark_as_parsed()

    self.assertListEqual(fv.fruit, [Fruit.APPLE])

  def test_define_results_in_registered_flag_with_string_list(self):
    fv = flags.FlagValues()
    enum_defaults = ['apple', 'APPLE']
    flags.DEFINE_multi_enum_class('fruit',
                                  enum_defaults, Fruit,
                                  'Enum option that can occur multiple times',
                                  flag_values=fv)
    fv.mark_as_parsed()

    self.assertListEqual(fv.fruit, [Fruit.apple, Fruit.APPLE])

  def test_define_results_in_registered_flag_with_enum_list(self):
    fv = flags.FlagValues()
    enum_defaults = [Fruit.APPLE, Fruit.orange]
    flags.DEFINE_multi_enum_class('fruit',
                                  enum_defaults, Fruit,
                                  'Enum option that can occur multiple times',
                                  flag_values=fv)
    fv.mark_as_parsed()

    self.assertListEqual(fv.fruit, [Fruit.APPLE, Fruit.orange])

  def test_from_command_line_returns_multiple(self):
    fv = flags.FlagValues()
    enum_defaults = [Fruit.APPLE]
    flags.DEFINE_multi_enum_class('fruit',
                                  enum_defaults, Fruit,
                                  'Enum option that can occur multiple times',
                                  flag_values=fv)
    argv = ('./program', '--fruit=apple', '--fruit=orange')
    fv(argv)
    self.assertListEqual(fv.fruit, [Fruit.apple, Fruit.orange])

  def test_bad_multi_enum_class_flags_from_definition(self):
    with self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        'flag --fruit=INVALID: value should be one of <apple|orange|APPLE>'):
      flags.DEFINE_multi_enum_class('fruit', ['INVALID'], Fruit, 'desc')

  def test_bad_multi_enum_class_flags_from_commandline(self):
    fv = flags.FlagValues()
    enum_defaults = [Fruit.APPLE]
    flags.DEFINE_multi_enum_class('fruit', enum_defaults, Fruit, 'desc',
                                  flag_values=fv)
    argv = ('./program', '--fruit=INVALID')
    with self.assertRaisesRegex(
        flags.IllegalFlagValueError,
        'flag --fruit=INVALID: value should be one of <apple|orange|APPLE>'):
      fv(argv)


class UnicodeFlagsTest(absltest.TestCase):
  """Testing proper unicode support for flags."""

  def test_unicode_default_and_helpstring(self):
    flags.DEFINE_string('unicode_str', b'\xC3\x80\xC3\xBD'.decode('utf-8'),
                        b'help:\xC3\xAA'.decode('utf-8'))
    argv = ('./program',)
    FLAGS(argv)   # should not raise any exceptions

    argv = ('./program', '--unicode_str=foo')
    FLAGS(argv)   # should not raise any exceptions

  def test_unicode_in_list(self):
    flags.DEFINE_list('unicode_list', ['abc', b'\xC3\x80'.decode('utf-8'),
                                       b'\xC3\xBD'.decode('utf-8')],
                      b'help:\xC3\xAB'.decode('utf-8'))
    argv = ('./program',)
    FLAGS(argv)   # should not raise any exceptions

    argv = ('./program', '--unicode_list=hello,there')
    FLAGS(argv)   # should not raise any exceptions

  def test_xmloutput(self):
    flags.DEFINE_string('unicode1', b'\xC3\x80\xC3\xBD'.decode('utf-8'),
                        b'help:\xC3\xAC'.decode('utf-8'))
    flags.DEFINE_list('unicode2', ['abc', b'\xC3\x80'.decode('utf-8'),
                                   b'\xC3\xBD'.decode('utf-8')],
                      b'help:\xC3\xAD'.decode('utf-8'))
    flags.DEFINE_list('non_unicode', ['abc', 'def', 'ghi'],
                      b'help:\xC3\xAD'.decode('utf-8'))

    outfile = io.StringIO() if six.PY3 else io.BytesIO()
    FLAGS.write_help_in_xml_format(outfile)
    actual_output = outfile.getvalue()
    if six.PY2:
      actual_output = actual_output.decode('utf-8')

    # The xml output is large, so we just check parts of it.
    self.assertIn(b'<name>unicode1</name>\n'
                  b'    <meaning>help:\xc3\xac</meaning>\n'
                  b'    <default>\xc3\x80\xc3\xbd</default>\n'
                  b'    <current>\xc3\x80\xc3\xbd</current>'.decode('utf-8'),
                  actual_output)
    if six.PY2:
      self.assertIn(b'<name>unicode2</name>\n'
                    b'    <meaning>help:\xc3\xad</meaning>\n'
                    b'    <default>abc,\xc3\x80,\xc3\xbd</default>\n'
                    b"    <current>['abc', u'\\xc0', u'\\xfd']"
                    b'</current>'.decode('utf-8'),
                    actual_output)
    else:
      self.assertIn(b'<name>unicode2</name>\n'
                    b'    <meaning>help:\xc3\xad</meaning>\n'
                    b'    <default>abc,\xc3\x80,\xc3\xbd</default>\n'
                    b"    <current>['abc', '\xc3\x80', '\xc3\xbd']"
                    b'</current>'.decode('utf-8'),
                    actual_output)
    self.assertIn(b'<name>non_unicode</name>\n'
                  b'    <meaning>help:\xc3\xad</meaning>\n'
                  b'    <default>abc,def,ghi</default>\n'
                  b"    <current>['abc', 'def', 'ghi']"
                  b'</current>'.decode('utf-8'),
                  actual_output)


class LoadFromFlagFileTest(absltest.TestCase):
  """Testing loading flags from a file and parsing them."""

  def setUp(self):
    self.flag_values = flags.FlagValues()
    flags.DEFINE_string('unittest_message1', 'Foo!', 'You Add Here.',
                        flag_values=self.flag_values)
    flags.DEFINE_string('unittest_message2', 'Bar!', 'Hello, Sailor!',
                        flag_values=self.flag_values)
    flags.DEFINE_boolean('unittest_boolflag', 0, 'Some Boolean thing',
                         flag_values=self.flag_values)
    flags.DEFINE_integer('unittest_number', 12345, 'Some integer',
                         lower_bound=0, flag_values=self.flag_values)
    flags.DEFINE_list('UnitTestList', '1,2,3', 'Some list',
                      flag_values=self.flag_values)
    self.tmp_path = None
    self.flag_values.mark_as_parsed()

  def tearDown(self):
    self._remove_test_files()

  def _setup_test_files(self):
    """Creates and sets up some dummy flagfile files with bogus flags."""

    # Figure out where to create temporary files
    self.assertFalse(self.tmp_path)
    self.tmp_path = tempfile.mkdtemp()

    tmp_flag_file_1 = open(self.tmp_path + '/UnitTestFile1.tst', 'w')
    tmp_flag_file_2 = open(self.tmp_path + '/UnitTestFile2.tst', 'w')
    tmp_flag_file_3 = open(self.tmp_path + '/UnitTestFile3.tst', 'w')
    tmp_flag_file_4 = open(self.tmp_path + '/UnitTestFile4.tst', 'w')

    # put some dummy flags in our test files
    tmp_flag_file_1.write('#A Fake Comment\n')
    tmp_flag_file_1.write('--unittest_message1=tempFile1!\n')
    tmp_flag_file_1.write('\n')
    tmp_flag_file_1.write('--unittest_number=54321\n')
    tmp_flag_file_1.write('--nounittest_boolflag\n')
    file_list = [tmp_flag_file_1.name]
    # this one includes test file 1
    tmp_flag_file_2.write('//A Different Fake Comment\n')
    tmp_flag_file_2.write('--flagfile=%s\n' % tmp_flag_file_1.name)
    tmp_flag_file_2.write('--unittest_message2=setFromTempFile2\n')
    tmp_flag_file_2.write('\t\t\n')
    tmp_flag_file_2.write('--unittest_number=6789a\n')
    file_list.append(tmp_flag_file_2.name)
    # this file points to itself
    tmp_flag_file_3.write('--flagfile=%s\n' % tmp_flag_file_3.name)
    tmp_flag_file_3.write('--unittest_message1=setFromTempFile3\n')
    tmp_flag_file_3.write('#YAFC\n')
    tmp_flag_file_3.write('--unittest_boolflag\n')
    file_list.append(tmp_flag_file_3.name)
    # this file is unreadable
    tmp_flag_file_4.write('--flagfile=%s\n' % tmp_flag_file_3.name)
    tmp_flag_file_4.write('--unittest_message1=setFromTempFile4\n')
    tmp_flag_file_4.write('--unittest_message1=setFromTempFile4\n')
    os.chmod(self.tmp_path + '/UnitTestFile4.tst', 0)
    file_list.append(tmp_flag_file_4.name)

    tmp_flag_file_1.close()
    tmp_flag_file_2.close()
    tmp_flag_file_3.close()
    tmp_flag_file_4.close()

    return file_list  # these are just the file names

  def _remove_test_files(self):
    """Removes the files we just created."""
    if self.tmp_path:
      shutil.rmtree(self.tmp_path, ignore_errors=True)
      self.tmp_path = None

  def _read_flags_from_files(self, argv, force_gnu):
    return argv[:1] + self.flag_values.read_flags_from_files(
        argv[1:], force_gnu=force_gnu)

  #### Flagfile Unit Tests ####
  def test_method_flagfiles_1(self):
    """Test trivial case with no flagfile based options."""
    fake_cmd_line = 'fooScript --unittest_boolflag'
    fake_argv = fake_cmd_line.split(' ')
    self.flag_values(fake_argv)
    self.assertEqual(self.flag_values.unittest_boolflag, 1)
    self.assertListEqual(
        fake_argv, self._read_flags_from_files(fake_argv, False))

  def test_method_flagfiles_2(self):
    """Tests parsing one file + arguments off simulated argv."""
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = 'fooScript --q --flagfile=%s' % tmp_files[0]
    fake_argv = fake_cmd_line.split(' ')

    # We should see the original cmd line with the file's contents spliced in.
    # Flags from the file will appear in the order order they are sepcified
    # in the file, in the same position as the flagfile argument.
    expected_results = ['fooScript',
                        '--q',
                        '--unittest_message1=tempFile1!',
                        '--unittest_number=54321',
                        '--nounittest_boolflag']
    test_results = self._read_flags_from_files(fake_argv, False)
    self.assertListEqual(expected_results, test_results)
  # end testTwo def

  def test_method_flagfiles_3(self):
    """Tests parsing nested files + arguments of simulated argv."""
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --unittest_number=77 --flagfile=%s'
                     % tmp_files[1])
    fake_argv = fake_cmd_line.split(' ')

    expected_results = ['fooScript',
                        '--unittest_number=77',
                        '--unittest_message1=tempFile1!',
                        '--unittest_number=54321',
                        '--nounittest_boolflag',
                        '--unittest_message2=setFromTempFile2',
                        '--unittest_number=6789a']
    test_results = self._read_flags_from_files(fake_argv, False)
    self.assertListEqual(expected_results, test_results)
  # end testThree def

  def test_method_flagfiles_3_spaces(self):
    """Tests parsing nested files + arguments of simulated argv.

    The arguments include a pair that is actually an arg with a value, so it
    doesn't stop processing.
    """
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --unittest_number 77 --flagfile=%s'
                     % tmp_files[1])
    fake_argv = fake_cmd_line.split(' ')

    expected_results = ['fooScript',
                        '--unittest_number',
                        '77',
                        '--unittest_message1=tempFile1!',
                        '--unittest_number=54321',
                        '--nounittest_boolflag',
                        '--unittest_message2=setFromTempFile2',
                        '--unittest_number=6789a']
    test_results = self._read_flags_from_files(fake_argv, False)
    self.assertListEqual(expected_results, test_results)

  def test_method_flagfiles_3_spaces_boolean(self):
    """Tests parsing nested files + arguments of simulated argv.

    The arguments include a pair that looks like a --x y arg with value, but
    since the flag is a boolean it's actually not.
    """
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --unittest_boolflag 77 --flagfile=%s'
                     % tmp_files[1])
    fake_argv = fake_cmd_line.split(' ')

    expected_results = ['fooScript',
                        '--unittest_boolflag',
                        '77',
                        '--flagfile=%s' % tmp_files[1]]
    with _use_gnu_getopt(self.flag_values, False):
      test_results = self._read_flags_from_files(fake_argv, False)
      self.assertListEqual(expected_results, test_results)

  def test_method_flagfiles_4(self):
    """Tests parsing self-referential files + arguments of simulated argv.

    This test should print a warning to stderr of some sort.
    """
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --flagfile=%s --nounittest_boolflag'
                     % tmp_files[2])
    fake_argv = fake_cmd_line.split(' ')
    expected_results = ['fooScript',
                        '--unittest_message1=setFromTempFile3',
                        '--unittest_boolflag',
                        '--nounittest_boolflag']

    test_results = self._read_flags_from_files(fake_argv, False)
    self.assertListEqual(expected_results, test_results)

  def test_method_flagfiles_5(self):
    """Test that --flagfile parsing respects the '--' end-of-options marker."""
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = 'fooScript --some_flag -- --flagfile=%s' % tmp_files[0]
    fake_argv = fake_cmd_line.split(' ')
    expected_results = ['fooScript',
                        '--some_flag',
                        '--',
                        '--flagfile=%s' % tmp_files[0]]

    test_results = self._read_flags_from_files(fake_argv, False)
    self.assertListEqual(expected_results, test_results)

  def test_method_flagfiles_6(self):
    """Test that --flagfile parsing stops at non-options (non-GNU behavior)."""
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --some_flag some_arg --flagfile=%s'
                     % tmp_files[0])
    fake_argv = fake_cmd_line.split(' ')
    expected_results = ['fooScript',
                        '--some_flag',
                        'some_arg',
                        '--flagfile=%s' % tmp_files[0]]

    with _use_gnu_getopt(self.flag_values, False):
      test_results = self._read_flags_from_files(fake_argv, False)
      self.assertListEqual(expected_results, test_results)

  def test_method_flagfiles_7(self):
    """Test that --flagfile parsing skips over a non-option (GNU behavior)."""
    self.flag_values.set_gnu_getopt()
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --some_flag some_arg --flagfile=%s'
                     % tmp_files[0])
    fake_argv = fake_cmd_line.split(' ')
    expected_results = ['fooScript',
                        '--some_flag',
                        'some_arg',
                        '--unittest_message1=tempFile1!',
                        '--unittest_number=54321',
                        '--nounittest_boolflag']

    test_results = self._read_flags_from_files(fake_argv, False)
    self.assertListEqual(expected_results, test_results)

  def test_method_flagfiles_8(self):
    """Test that --flagfile parsing respects force_gnu=True."""
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --some_flag some_arg --flagfile=%s'
                     % tmp_files[0])
    fake_argv = fake_cmd_line.split(' ')
    expected_results = ['fooScript',
                        '--some_flag',
                        'some_arg',
                        '--unittest_message1=tempFile1!',
                        '--unittest_number=54321',
                        '--nounittest_boolflag']

    test_results = self._read_flags_from_files(fake_argv, True)
    self.assertListEqual(expected_results, test_results)

  def test_method_flagfiles_repeated_non_circular(self):
    """Tests that parsing repeated non-circular flagfiles works."""
    tmp_files = self._setup_test_files()
    # specify our temp files on the fake cmd line
    fake_cmd_line = ('fooScript --flagfile=%s --flagfile=%s'
                     % (tmp_files[1], tmp_files[0]))
    fake_argv = fake_cmd_line.split(' ')
    expected_results = ['fooScript',
                        '--unittest_message1=tempFile1!',
                        '--unittest_number=54321',
                        '--nounittest_boolflag',
                        '--unittest_message2=setFromTempFile2',
                        '--unittest_number=6789a',
                        '--unittest_message1=tempFile1!',
                        '--unittest_number=54321',
                        '--nounittest_boolflag']

    test_results = self._read_flags_from_files(fake_argv, False)
    self.assertListEqual(expected_results, test_results)

  @unittest.skipIf(
      os.name == 'nt',
      'There is no good way to create an unreadable file on Windows.')
  def test_method_flagfiles_no_permissions(self):
    """Test that --flagfile raises except on file that is unreadable."""
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --some_flag some_arg --flagfile=%s'
                     % tmp_files[3])
    fake_argv = fake_cmd_line.split(' ')
    self.assertRaises(flags.CantOpenFlagFileError,
                      self._read_flags_from_files, fake_argv, True)

  def test_method_flagfiles_not_found(self):
    """Test that --flagfile raises except on file that does not exist."""
    tmp_files = self._setup_test_files()
    # specify our temp file on the fake cmd line
    fake_cmd_line = ('fooScript --some_flag some_arg --flagfile=%sNOTEXIST'
                     % tmp_files[3])
    fake_argv = fake_cmd_line.split(' ')
    self.assertRaises(flags.CantOpenFlagFileError,
                      self._read_flags_from_files, fake_argv, True)

  def test_flagfiles_user_path_expansion(self):
    """Test that user directory referenced paths are correctly expanded.

    Test paths like ~/foo. This test depends on whatever account's running
    the unit test to have read/write access to their own home directory,
    otherwise it'll FAIL.
    """
    fake_flagfile_item_style_1 = '--flagfile=~/foo.file'
    fake_flagfile_item_style_2 = '-flagfile=~/foo.file'

    expected_results = os.path.expanduser('~/foo.file')

    test_results = self.flag_values._extract_filename(
        fake_flagfile_item_style_1)
    self.assertEqual(expected_results, test_results)

    test_results = self.flag_values._extract_filename(
        fake_flagfile_item_style_2)
    self.assertEqual(expected_results, test_results)

  def test_no_touchy_non_flags(self):
    """Test that the flags parser does not mutilate arguments.

    The argumants are not supposed to be flags
    """
    fake_argv = ['fooScript', '--unittest_boolflag',
                 'command', '--command_arg1', '--UnitTestBoom', '--UnitTestB']
    with _use_gnu_getopt(self.flag_values, False):
      argv = self.flag_values(fake_argv)
      self.assertListEqual(argv, fake_argv[:1] + fake_argv[2:])

  def test_parse_flags_after_args_if_using_gnugetopt(self):
    """Test that flags given after arguments are parsed if using gnu_getopt."""
    self.flag_values.set_gnu_getopt()
    fake_argv = ['fooScript', '--unittest_boolflag',
                 'command', '--unittest_number=54321']
    argv = self.flag_values(fake_argv)
    self.assertListEqual(argv, ['fooScript', 'command'])

  def test_set_default(self):
    """Test changing flag defaults."""
    # Test that set_default changes both the default and the value,
    # and that the value is changed when one is given as an option.
    self.flag_values.set_default('unittest_message1', 'New value')
    self.assertEqual(self.flag_values.unittest_message1, 'New value')
    self.assertEqual(self.flag_values['unittest_message1'].default_as_str,
                     "'New value'")
    self.flag_values(['dummyscript', '--unittest_message1=Newer value'])
    self.assertEqual(self.flag_values.unittest_message1, 'Newer value')

    # Test that setting the default to None works correctly.
    self.flag_values.set_default('unittest_number', None)
    self.assertEqual(self.flag_values.unittest_number, None)
    self.assertEqual(self.flag_values['unittest_number'].default_as_str, None)
    self.flag_values(['dummyscript', '--unittest_number=56'])
    self.assertEqual(self.flag_values.unittest_number, 56)

    # Test that setting the default to zero works correctly.
    self.flag_values.set_default('unittest_number', 0)
    self.assertEqual(self.flag_values['unittest_number'].default, 0)
    self.assertEqual(self.flag_values.unittest_number, 56)
    self.assertEqual(
        self.flag_values['unittest_number'].default_as_str, "'0'")
    self.flag_values(['dummyscript', '--unittest_number=56'])
    self.assertEqual(self.flag_values.unittest_number, 56)

    # Test that setting the default to '' works correctly.
    self.flag_values.set_default('unittest_message1', '')
    self.assertEqual(self.flag_values['unittest_message1'].default, '')
    self.assertEqual(self.flag_values.unittest_message1, 'Newer value')
    self.assertEqual(self.flag_values['unittest_message1'].default_as_str, "''")
    self.flag_values(['dummyscript', '--unittest_message1=fifty-six'])
    self.assertEqual(self.flag_values.unittest_message1, 'fifty-six')

    # Test that setting the default to false works correctly.
    self.flag_values.set_default('unittest_boolflag', False)
    self.assertEqual(self.flag_values.unittest_boolflag, False)
    self.assertEqual(self.flag_values['unittest_boolflag'].default_as_str,
                     "'false'")
    self.flag_values(['dummyscript', '--unittest_boolflag=true'])
    self.assertEqual(self.flag_values.unittest_boolflag, True)

    # Test that setting a list default works correctly.
    self.flag_values.set_default('UnitTestList', '4,5,6')
    self.assertListEqual(self.flag_values.UnitTestList, ['4', '5', '6'])
    self.assertEqual(self.flag_values['UnitTestList'].default_as_str,
                     "'4,5,6'")
    self.flag_values(['dummyscript', '--UnitTestList=7,8,9'])
    self.assertListEqual(self.flag_values.UnitTestList, ['7', '8', '9'])

    # Test that setting invalid defaults raises exceptions
    with self.assertRaises(flags.IllegalFlagValueError):
      self.flag_values.set_default('unittest_number', 'oops')
    with self.assertRaises(flags.IllegalFlagValueError):
      self.flag_values.set_default('unittest_number', -1)


class FlagsParsingTest(absltest.TestCase):
  """Testing different aspects of parsing: '-f' vs '--flag', etc."""

  def setUp(self):
    self.flag_values = flags.FlagValues()

  def test_two_dash_arg_first(self):
    flags.DEFINE_string('twodash_name', 'Bob', 'namehelp',
                        flag_values=self.flag_values)
    flags.DEFINE_string('twodash_blame', 'Rob', 'blamehelp',
                        flag_values=self.flag_values)
    argv = ('./program',
            '--',
            '--twodash_name=Harry')
    argv = self.flag_values(argv)
    self.assertEqual('Bob', self.flag_values.twodash_name)
    self.assertEqual(argv[1], '--twodash_name=Harry')

  def test_two_dash_arg_middle(self):
    flags.DEFINE_string('twodash2_name', 'Bob', 'namehelp',
                        flag_values=self.flag_values)
    flags.DEFINE_string('twodash2_blame', 'Rob', 'blamehelp',
                        flag_values=self.flag_values)
    argv = ('./program',
            '--twodash2_blame=Larry',
            '--',
            '--twodash2_name=Harry')
    argv = self.flag_values(argv)
    self.assertEqual('Bob', self.flag_values.twodash2_name)
    self.assertEqual('Larry', self.flag_values.twodash2_blame)
    self.assertEqual(argv[1], '--twodash2_name=Harry')

  def test_one_dash_arg_first(self):
    flags.DEFINE_string('onedash_name', 'Bob', 'namehelp',
                        flag_values=self.flag_values)
    flags.DEFINE_string('onedash_blame', 'Rob', 'blamehelp',
                        flag_values=self.flag_values)
    argv = ('./program',
            '-',
            '--onedash_name=Harry')
    with _use_gnu_getopt(self.flag_values, False):
      argv = self.flag_values(argv)
      self.assertEqual(len(argv), 3)
      self.assertEqual(argv[1], '-')
      self.assertEqual(argv[2], '--onedash_name=Harry')

  def test_unrecognized_flags(self):
    flags.DEFINE_string('name', 'Bob', 'namehelp', flag_values=self.flag_values)
    # Unknown flag --nosuchflag
    try:
      argv = ('./program', '--nosuchflag', '--name=Bob', 'extra')
      self.flag_values(argv)
      raise AssertionError('Unknown flag exception not raised')
    except flags.UnrecognizedFlagError as e:
      self.assertEqual(e.flagname, 'nosuchflag')
      self.assertEqual(e.flagvalue, '--nosuchflag')

    # Unknown flag -w (short option)
    try:
      argv = ('./program', '-w', '--name=Bob', 'extra')
      self.flag_values(argv)
      raise AssertionError('Unknown flag exception not raised')
    except flags.UnrecognizedFlagError as e:
      self.assertEqual(e.flagname, 'w')
      self.assertEqual(e.flagvalue, '-w')

    # Unknown flag --nosuchflagwithparam=foo
    try:
      argv = ('./program', '--nosuchflagwithparam=foo', '--name=Bob', 'extra')
      self.flag_values(argv)
      raise AssertionError('Unknown flag exception not raised')
    except flags.UnrecognizedFlagError as e:
      self.assertEqual(e.flagname, 'nosuchflagwithparam')
      self.assertEqual(e.flagvalue, '--nosuchflagwithparam=foo')

    # Allow unknown flag --nosuchflag if specified with undefok
    argv = ('./program', '--nosuchflag', '--name=Bob',
            '--undefok=nosuchflag', 'extra')
    argv = self.flag_values(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')

    # Allow unknown flag --noboolflag if undefok=boolflag is specified
    argv = ('./program', '--noboolflag', '--name=Bob',
            '--undefok=boolflag', 'extra')
    argv = self.flag_values(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')

    # But not if the flagname is misspelled:
    try:
      argv = ('./program', '--nosuchflag', '--name=Bob',
              '--undefok=nosuchfla', 'extra')
      self.flag_values(argv)
      raise AssertionError('Unknown flag exception not raised')
    except flags.UnrecognizedFlagError as e:
      self.assertEqual(e.flagname, 'nosuchflag')

    try:
      argv = ('./program', '--nosuchflag', '--name=Bob',
              '--undefok=nosuchflagg', 'extra')
      self.flag_values(argv)
      raise AssertionError('Unknown flag exception not raised')
    except flags.UnrecognizedFlagError as e:
      self.assertEqual(e.flagname, 'nosuchflag')

    # Allow unknown short flag -w if specified with undefok
    argv = ('./program', '-w', '--name=Bob', '--undefok=w', 'extra')
    argv = self.flag_values(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')

    # Allow unknown flag --nosuchflagwithparam=foo if specified
    # with undefok
    argv = ('./program', '--nosuchflagwithparam=foo', '--name=Bob',
            '--undefok=nosuchflagwithparam', 'extra')
    argv = self.flag_values(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')

    # Even if undefok specifies multiple flags
    argv = ('./program', '--nosuchflag', '-w', '--nosuchflagwithparam=foo',
            '--name=Bob',
            '--undefok=nosuchflag,w,nosuchflagwithparam',
            'extra')
    argv = self.flag_values(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')

    # However, not if undefok doesn't specify the flag
    try:
      argv = ('./program', '--nosuchflag', '--name=Bob',
              '--undefok=another_such', 'extra')
      self.flag_values(argv)
      raise AssertionError('Unknown flag exception not raised')
    except flags.UnrecognizedFlagError as e:
      self.assertEqual(e.flagname, 'nosuchflag')

    # Make sure --undefok doesn't mask other option errors.
    try:
      # Provide an option requiring a parameter but not giving it one.
      argv = ('./program', '--undefok=name', '--name')
      self.flag_values(argv)
      raise AssertionError('Missing option parameter exception not raised')
    except flags.UnrecognizedFlagError:
      raise AssertionError('Wrong kind of error exception raised')
    except flags.Error:
      pass

    # Test --undefok <list>
    argv = ('./program', '--nosuchflag', '-w', '--nosuchflagwithparam=foo',
            '--name=Bob',
            '--undefok',
            'nosuchflag,w,nosuchflagwithparam',
            'extra')
    argv = self.flag_values(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')

    # Test incorrect --undefok with no value.
    argv = ('./program', '--name=Bob', '--undefok')
    with self.assertRaises(flags.Error):
      self.flag_values(argv)


class NonGlobalFlagsTest(absltest.TestCase):

  def test_nonglobal_flags(self):
    """Test use of non-global FlagValues."""
    nonglobal_flags = flags.FlagValues()
    flags.DEFINE_string('nonglobal_flag', 'Bob', 'flaghelp', nonglobal_flags)
    argv = ('./program',
            '--nonglobal_flag=Mary',
            'extra')
    argv = nonglobal_flags(argv)
    self.assertEqual(len(argv), 2, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')
    self.assertEqual(argv[1], 'extra', 'extra argument not preserved')
    self.assertEqual(nonglobal_flags['nonglobal_flag'].value, 'Mary')

  def test_unrecognized_nonglobal_flags(self):
    """Test unrecognized non-global flags."""
    nonglobal_flags = flags.FlagValues()
    argv = ('./program',
            '--nosuchflag')
    try:
      argv = nonglobal_flags(argv)
      raise AssertionError('Unknown flag exception not raised')
    except flags.UnrecognizedFlagError as e:
      self.assertEqual(e.flagname, 'nosuchflag')

    argv = ('./program',
            '--nosuchflag',
            '--undefok=nosuchflag')

    argv = nonglobal_flags(argv)
    self.assertEqual(len(argv), 1, 'wrong number of arguments pulled')
    self.assertEqual(argv[0], './program', 'program name not preserved')

  def test_create_flag_errors(self):
    # Since the exception classes are exposed, nothing stops users
    # from creating their own instances. This test makes sure that
    # people modifying the flags module understand that the external
    # mechanisms for creating the exceptions should continue to work.
    _ = flags.Error()
    _ = flags.Error('message')
    _ = flags.DuplicateFlagError()
    _ = flags.DuplicateFlagError('message')
    _ = flags.IllegalFlagValueError()
    _ = flags.IllegalFlagValueError('message')

  def test_flag_values_del_attr(self):
    """Checks that del self.flag_values.flag_id works."""
    default_value = 'default value for test_flag_values_del_attr'
    # 1. Declare and delete a flag with no short name.
    flag_values = flags.FlagValues()
    flags.DEFINE_string('delattr_foo', default_value, 'A simple flag.',
                        flag_values=flag_values)

    flag_values.mark_as_parsed()
    self.assertEqual(flag_values.delattr_foo, default_value)
    flag_obj = flag_values['delattr_foo']
    # We also check that _FlagIsRegistered works as expected :)
    self.assertTrue(flag_values._flag_is_registered(flag_obj))
    del flag_values.delattr_foo
    self.assertFalse('delattr_foo' in flag_values._flags())
    self.assertFalse(flag_values._flag_is_registered(flag_obj))
    # If the previous del FLAGS.delattr_foo did not work properly, the
    # next definition will trigger a redefinition error.
    flags.DEFINE_integer('delattr_foo', 3, 'A simple flag.',
                         flag_values=flag_values)
    del flag_values.delattr_foo

    self.assertFalse('delattr_foo' in flag_values)

    # 2. Declare and delete a flag with a short name.
    flags.DEFINE_string('delattr_bar', default_value, 'flag with short name',
                        short_name='x5', flag_values=flag_values)
    flag_obj = flag_values['delattr_bar']
    self.assertTrue(flag_values._flag_is_registered(flag_obj))
    del flag_values.x5
    self.assertTrue(flag_values._flag_is_registered(flag_obj))
    del flag_values.delattr_bar
    self.assertFalse(flag_values._flag_is_registered(flag_obj))

    # 3. Just like 2, but del flag_values.name last
    flags.DEFINE_string('delattr_bar', default_value, 'flag with short name',
                        short_name='x5', flag_values=flag_values)
    flag_obj = flag_values['delattr_bar']
    self.assertTrue(flag_values._flag_is_registered(flag_obj))
    del flag_values.delattr_bar
    self.assertTrue(flag_values._flag_is_registered(flag_obj))
    del flag_values.x5
    self.assertFalse(flag_values._flag_is_registered(flag_obj))

    self.assertFalse('delattr_bar' in flag_values)
    self.assertFalse('x5' in flag_values)

  def test_list_flag_format(self):
    """Tests for correctly-formatted list flags."""
    flags.DEFINE_list('listflag', '', 'A list of arguments')

    def _check_parsing(listval):
      """Parse a particular value for our test flag, --listflag."""
      argv = FLAGS(['./program', '--listflag=' + listval, 'plain-arg'])
      self.assertEqual(['./program', 'plain-arg'], argv)
      return FLAGS.listflag

    # Basic success case
    self.assertEqual(_check_parsing('foo,bar'), ['foo', 'bar'])
    # Success case: newline in argument is quoted.
    self.assertEqual(_check_parsing('"foo","bar\nbar"'), ['foo', 'bar\nbar'])
    # Failure case: newline in argument is unquoted.
    self.assertRaises(
        flags.IllegalFlagValueError, _check_parsing, '"foo",bar\nbar')
    # Failure case: unmatched ".
    self.assertRaises(
        flags.IllegalFlagValueError, _check_parsing, '"foo,barbar')

  def test_flag_definition_via_setitem(self):
    with self.assertRaises(flags.IllegalFlagValueError):
      flag_values = flags.FlagValues()
      flag_values['flag_name'] = 'flag_value'


class KeyFlagsTest(absltest.TestCase):

  def setUp(self):
    self.flag_values = flags.FlagValues()

  def _get_names_of_defined_flags(self, module, flag_values):
    """Returns the list of names of flags defined by a module.

    Auxiliary for the test_key_flags* methods.

    Args:
      module: A module object or a string module name.
      flag_values: A FlagValues object.

    Returns:
      A list of strings.
    """
    return [f.name for f in flag_values._get_flags_defined_by_module(module)]

  def _get_names_of_key_flags(self, module, flag_values):
    """Returns the list of names of key flags for a module.

    Auxiliary for the test_key_flags* methods.

    Args:
      module: A module object or a string module name.
      flag_values: A FlagValues object.

    Returns:
      A list of strings.
    """
    return [f.name for f in flag_values.get_key_flags_for_module(module)]

  def _assert_lists_have_same_elements(self, list_1, list_2):
    # Checks that two lists have the same elements with the same
    # multiplicity, in possibly different order.
    list_1 = list(list_1)
    list_1.sort()
    list_2 = list(list_2)
    list_2.sort()
    self.assertListEqual(list_1, list_2)

  def test_key_flags(self):
    flag_values = flags.FlagValues()
    # Before starting any testing, make sure no flags are already
    # defined for module_foo and module_bar.
    self.assertListEqual(self._get_names_of_key_flags(module_foo, flag_values),
                         [])
    self.assertListEqual(self._get_names_of_key_flags(module_bar, flag_values),
                         [])
    self.assertListEqual(self._get_names_of_defined_flags(module_foo,
                                                          flag_values),
                         [])
    self.assertListEqual(self._get_names_of_defined_flags(module_bar,
                                                          flag_values),
                         [])

    # Defines a few flags in module_foo and module_bar.
    module_foo.define_flags(flag_values=flag_values)

    try:
      # Part 1. Check that all flags defined by module_foo are key for
      # that module, and similarly for module_bar.
      for module in [module_foo, module_bar]:
        self._assert_lists_have_same_elements(
            flag_values._get_flags_defined_by_module(module),
            flag_values.get_key_flags_for_module(module))
        # Also check that each module defined the expected flags.
        self._assert_lists_have_same_elements(
            self._get_names_of_defined_flags(module, flag_values),
            module.names_of_defined_flags())

      # Part 2. Check that flags.declare_key_flag works fine.
      # Declare that some flags from module_bar are key for
      # module_foo.
      module_foo.declare_key_flags(flag_values=flag_values)

      # Check that module_foo has the expected list of defined flags.
      self._assert_lists_have_same_elements(
          self._get_names_of_defined_flags(module_foo, flag_values),
          module_foo.names_of_defined_flags())

      # Check that module_foo has the expected list of key flags.
      self._assert_lists_have_same_elements(
          self._get_names_of_key_flags(module_foo, flag_values),
          module_foo.names_of_declared_key_flags())

      # Part 3. Check that flags.adopt_module_key_flags works fine.
      # Trigger a call to flags.adopt_module_key_flags(module_bar)
      # inside module_foo.  This should declare a few more key
      # flags in module_foo.
      module_foo.declare_extra_key_flags(flag_values=flag_values)

      # Check that module_foo has the expected list of key flags.
      self._assert_lists_have_same_elements(
          self._get_names_of_key_flags(module_foo, flag_values),
          module_foo.names_of_declared_key_flags() +
          module_foo.names_of_declared_extra_key_flags())
    finally:
      module_foo.remove_flags(flag_values=flag_values)

  def test_key_flags_with_non_default_flag_values_object(self):
    # Check that key flags work even when we use a FlagValues object
    # that is not the default flags.self.flag_values object.  Otherwise, this
    # test is similar to test_key_flags, but it uses only module_bar.
    # The other test module (module_foo) uses only the default values
    # for the flag_values keyword arguments.  This way, test_key_flags
    # and this method test both the default FlagValues, the explicitly
    # specified one, and a mixed usage of the two.

    # A brand-new FlagValues object, to use instead of flags.self.flag_values.
    fv = flags.FlagValues()

    # Before starting any testing, make sure no flags are already
    # defined for module_foo and module_bar.
    self.assertListEqual(
        self._get_names_of_key_flags(module_bar, fv),
        [])
    self.assertListEqual(
        self._get_names_of_defined_flags(module_bar, fv),
        [])

    module_bar.define_flags(flag_values=fv)

    # Check that all flags defined by module_bar are key for that
    # module, and that module_bar defined the expected flags.
    self._assert_lists_have_same_elements(
        fv._get_flags_defined_by_module(module_bar),
        fv.get_key_flags_for_module(module_bar))
    self._assert_lists_have_same_elements(
        self._get_names_of_defined_flags(module_bar, fv),
        module_bar.names_of_defined_flags())

    # Pick two flags from module_bar, declare them as key for the
    # current (i.e., main) module (via flags.declare_key_flag), and
    # check that we get the expected effect.  The important thing is
    # that we always use flags_values=fv (instead of the default
    # self.flag_values).
    main_module = sys.argv[0]
    names_of_flags_defined_by_bar = module_bar.names_of_defined_flags()
    flag_name_0 = names_of_flags_defined_by_bar[0]
    flag_name_2 = names_of_flags_defined_by_bar[2]

    flags.declare_key_flag(flag_name_0, flag_values=fv)
    self._assert_lists_have_same_elements(
        self._get_names_of_key_flags(main_module, fv),
        [flag_name_0])

    flags.declare_key_flag(flag_name_2, flag_values=fv)
    self._assert_lists_have_same_elements(
        self._get_names_of_key_flags(main_module, fv),
        [flag_name_0, flag_name_2])

    # Try with a special (not user-defined) flag too:
    flags.declare_key_flag('undefok', flag_values=fv)
    self._assert_lists_have_same_elements(
        self._get_names_of_key_flags(main_module, fv),
        [flag_name_0, flag_name_2, 'undefok'])

    flags.adopt_module_key_flags(module_bar, fv)
    self._assert_lists_have_same_elements(
        self._get_names_of_key_flags(main_module, fv),
        names_of_flags_defined_by_bar + ['undefok'])

    # Adopt key flags from the flags module itself.
    flags.adopt_module_key_flags(flags, flag_values=fv)
    self._assert_lists_have_same_elements(
        self._get_names_of_key_flags(main_module, fv),
        names_of_flags_defined_by_bar + ['flagfile', 'undefok'])

  def test_main_module_help_with_key_flags(self):
    # Similar to test_main_module_help, but this time we make sure to
    # declare some key flags.

    # Safety check that the main module does not declare any flags
    # at the beginning of this test.
    expected_help = ''
    self.assertMultiLineEqual(expected_help,
                              self.flag_values.main_module_help())

    # Define one flag in this main module and some flags in modules
    # a and b.  Also declare one flag from module a and one flag
    # from module b as key flags for the main module.
    flags.DEFINE_integer('main_module_int_fg', 1,
                         'Integer flag in the main module.',
                         flag_values=self.flag_values)

    try:
      main_module_int_fg_help = (
          '  --main_module_int_fg: Integer flag in the main module.\n'
          "    (default: '1')\n"
          '    (an integer)')

      expected_help += '\n%s:\n%s' % (sys.argv[0], main_module_int_fg_help)
      self.assertMultiLineEqual(expected_help,
                                self.flag_values.main_module_help())

      # The following call should be a no-op: any flag declared by a
      # module is automatically key for that module.
      flags.declare_key_flag('main_module_int_fg', flag_values=self.flag_values)
      self.assertMultiLineEqual(expected_help,
                                self.flag_values.main_module_help())

      # The definition of a few flags in an imported module should not
      # change the main module help.
      module_foo.define_flags(flag_values=self.flag_values)
      self.assertMultiLineEqual(expected_help,
                                self.flag_values.main_module_help())

      flags.declare_key_flag('tmod_foo_bool', flag_values=self.flag_values)
      tmod_foo_bool_help = (
          '  --[no]tmod_foo_bool: Boolean flag from module foo.\n'
          "    (default: 'true')")
      expected_help += '\n' + tmod_foo_bool_help
      self.assertMultiLineEqual(expected_help,
                                self.flag_values.main_module_help())

      flags.declare_key_flag('tmod_bar_z', flag_values=self.flag_values)
      tmod_bar_z_help = (
          '  --[no]tmod_bar_z: Another boolean flag from module bar.\n'
          "    (default: 'false')")
      # Unfortunately, there is some flag sorting inside
      # main_module_help, so we can't keep incrementally extending
      # the expected_help string ...
      expected_help = ('\n%s:\n%s\n%s\n%s' %
                       (sys.argv[0],
                        main_module_int_fg_help,
                        tmod_bar_z_help,
                        tmod_foo_bool_help))
      self.assertMultiLineEqual(self.flag_values.main_module_help(),
                                expected_help)

    finally:
      # At the end, delete all the flag information we created.
      self.flag_values.__delattr__('main_module_int_fg')
      module_foo.remove_flags(flag_values=self.flag_values)

  def test_adoptmodule_key_flags(self):
    # Check that adopt_module_key_flags raises an exception when
    # called with a module name (as opposed to a module object).
    self.assertRaises(flags.Error,
                      flags.adopt_module_key_flags,
                      'pyglib.app')

  def test_disclaimkey_flags(self):
    original_disclaim_module_ids = _helpers.disclaim_module_ids
    _helpers.disclaim_module_ids = set(_helpers.disclaim_module_ids)
    try:
      module_bar.disclaim_key_flags()
      module_foo.define_bar_flags(flag_values=self.flag_values)
      module_name = self.flag_values.find_module_defining_flag('tmod_bar_x')
      self.assertEqual(module_foo.__name__, module_name)
    finally:
      _helpers.disclaim_module_ids = original_disclaim_module_ids


class FindModuleTest(absltest.TestCase):
  """Testing methods that find a module that defines a given flag."""

  def test_find_module_defining_flag(self):
    self.assertEqual('default', FLAGS.find_module_defining_flag(
        '__NON_EXISTENT_FLAG__', 'default'))
    self.assertEqual(
        module_baz.__name__, FLAGS.find_module_defining_flag('tmod_baz_x'))

  def test_find_module_id_defining_flag(self):
    self.assertEqual('default', FLAGS.find_module_id_defining_flag(
        '__NON_EXISTENT_FLAG__', 'default'))
    self.assertEqual(
        id(module_baz), FLAGS.find_module_id_defining_flag('tmod_baz_x'))

  def test_find_module_defining_flag_passing_module_name(self):
    my_flags = flags.FlagValues()
    module_name = sys.__name__  # Must use an existing module.
    flags.DEFINE_boolean('flag_name', True,
                         'Flag with a different module name.',
                         flag_values=my_flags,
                         module_name=module_name)
    self.assertEqual(module_name,
                     my_flags.find_module_defining_flag('flag_name'))

  def test_find_module_id_defining_flag_passing_module_name(self):
    my_flags = flags.FlagValues()
    module_name = sys.__name__  # Must use an existing module.
    flags.DEFINE_boolean('flag_name', True,
                         'Flag with a different module name.',
                         flag_values=my_flags,
                         module_name=module_name)
    self.assertEqual(id(sys),
                     my_flags.find_module_id_defining_flag('flag_name'))


class FlagsErrorMessagesTest(absltest.TestCase):
  """Testing special cases for integer and float flags error messages."""

  def setUp(self):
    self.flag_values = flags.FlagValues()

  def test_integer_error_text(self):
    # Make sure we get proper error text
    flags.DEFINE_integer('positive', 4, 'non-negative flag', lower_bound=1,
                         flag_values=self.flag_values)
    flags.DEFINE_integer('non_negative', 4, 'positive flag', lower_bound=0,
                         flag_values=self.flag_values)
    flags.DEFINE_integer('negative', -4, 'negative flag', upper_bound=-1,
                         flag_values=self.flag_values)
    flags.DEFINE_integer('non_positive', -4, 'non-positive flag', upper_bound=0,
                         flag_values=self.flag_values)
    flags.DEFINE_integer('greater', 19, 'greater-than flag', lower_bound=4,
                         flag_values=self.flag_values)
    flags.DEFINE_integer('smaller', -19, 'smaller-than flag', upper_bound=4,
                         flag_values=self.flag_values)
    flags.DEFINE_integer('usual', 4, 'usual flag', lower_bound=0,
                         upper_bound=10000, flag_values=self.flag_values)
    flags.DEFINE_integer('another_usual', 0, 'usual flag', lower_bound=-1,
                         upper_bound=1, flag_values=self.flag_values)

    self._check_error_message('positive', -4, 'a positive integer')
    self._check_error_message('non_negative', -4, 'a non-negative integer')
    self._check_error_message('negative', 0, 'a negative integer')
    self._check_error_message('non_positive', 4, 'a non-positive integer')
    self._check_error_message('usual', -4, 'an integer in the range [0, 10000]')
    self._check_error_message('another_usual', 4,
                              'an integer in the range [-1, 1]')
    self._check_error_message('greater', -5, 'integer >= 4')
    self._check_error_message('smaller', 5, 'integer <= 4')

  def test_float_error_text(self):
    flags.DEFINE_float('positive', 4, 'non-negative flag', lower_bound=1,
                       flag_values=self.flag_values)
    flags.DEFINE_float('non_negative', 4, 'positive flag', lower_bound=0,
                       flag_values=self.flag_values)
    flags.DEFINE_float('negative', -4, 'negative flag', upper_bound=-1,
                       flag_values=self.flag_values)
    flags.DEFINE_float('non_positive', -4, 'non-positive flag', upper_bound=0,
                       flag_values=self.flag_values)
    flags.DEFINE_float('greater', 19, 'greater-than flag', lower_bound=4,
                       flag_values=self.flag_values)
    flags.DEFINE_float('smaller', -19, 'smaller-than flag', upper_bound=4,
                       flag_values=self.flag_values)
    flags.DEFINE_float('usual', 4, 'usual flag', lower_bound=0,
                       upper_bound=10000, flag_values=self.flag_values)
    flags.DEFINE_float('another_usual', 0, 'usual flag', lower_bound=-1,
                       upper_bound=1, flag_values=self.flag_values)

    self._check_error_message('positive', 0.5, 'number >= 1')
    self._check_error_message('non_negative', -4.0, 'a non-negative number')
    self._check_error_message('negative', 0.5, 'number <= -1')
    self._check_error_message('non_positive', 4.0, 'a non-positive number')
    self._check_error_message('usual', -4.0, 'a number in the range [0, 10000]')
    self._check_error_message('another_usual', 4.0,
                              'a number in the range [-1, 1]')
    self._check_error_message('smaller', 5.0, 'number <= 4')

  def _check_error_message(self, flag_name, flag_value,
                           expected_message_suffix):
    """Set a flag to a given value and make sure we get expected message."""

    try:
      self.flag_values.__setattr__(flag_name, flag_value)
      raise AssertionError('Bounds exception not raised!')
    except flags.IllegalFlagValueError as e:
      expected = ('flag --%(name)s=%(value)s: %(value)s is not %(suffix)s' %
                  {'name': flag_name, 'value': flag_value,
                   'suffix': expected_message_suffix})
      self.assertEqual(str(e), expected)


if __name__ == '__main__':
  absltest.main()
