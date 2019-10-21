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

"""Tests for flags.FlagValues class."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import pickle
import types
import unittest

from absl import logging
from absl.flags import _defines
from absl.flags import _exceptions
from absl.flags import _flagvalues
from absl.flags import _helpers
from absl.flags.tests import module_foo
from absl.testing import absltest
from absl.testing import parameterized
import mock
import six


class FlagValuesTest(absltest.TestCase):

  def test_bool_flags(self):
    for arg, expected in (('--nothing', True),
                          ('--nothing=true', True),
                          ('--nothing=false', False),
                          ('--nonothing', False)):
      fv = _flagvalues.FlagValues()
      _defines.DEFINE_boolean('nothing', None, '', flag_values=fv)
      fv(('./program', arg))
      self.assertIs(expected, fv.nothing)

    for arg in ('--nonothing=true', '--nonothing=false'):
      fv = _flagvalues.FlagValues()
      _defines.DEFINE_boolean('nothing', None, '', flag_values=fv)
      with self.assertRaises(ValueError):
        fv(('./program', arg))

  def test_boolean_flag_parser_gets_string_argument(self):
    for arg, expected in (('--nothing', 'true'),
                          ('--nothing=true', 'true'),
                          ('--nothing=false', 'false'),
                          ('--nonothing', 'false')):
      fv = _flagvalues.FlagValues()
      _defines.DEFINE_boolean('nothing', None, '', flag_values=fv)
      with mock.patch.object(fv['nothing'].parser, 'parse') as mock_parse:
        fv(('./program', arg))
        mock_parse.assert_called_once_with(expected)

  def test_unregistered_flags_are_cleaned_up(self):
    fv = _flagvalues.FlagValues()
    module, module_name = _helpers.get_calling_module_object_and_name()

    # Define first flag.
    _defines.DEFINE_integer('cores', 4, '', flag_values=fv, short_name='c')
    old_cores_flag = fv['cores']
    fv.register_key_flag_for_module(module_name, old_cores_flag)
    self.assertEqual(fv.flags_by_module_dict(),
                     {module_name: [old_cores_flag]})
    self.assertEqual(fv.flags_by_module_id_dict(),
                     {id(module): [old_cores_flag]})
    self.assertEqual(fv.key_flags_by_module_dict(),
                     {module_name: [old_cores_flag]})

    # Redefine the same flag.
    _defines.DEFINE_integer(
        'cores', 4, '', flag_values=fv, short_name='c', allow_override=True)
    new_cores_flag = fv['cores']
    self.assertNotEqual(old_cores_flag, new_cores_flag)
    self.assertEqual(fv.flags_by_module_dict(),
                     {module_name: [new_cores_flag]})
    self.assertEqual(fv.flags_by_module_id_dict(),
                     {id(module): [new_cores_flag]})
    # old_cores_flag is removed from key flags, and the new_cores_flag is
    # not automatically added because it must be registered explicitly.
    self.assertEqual(fv.key_flags_by_module_dict(), {module_name: []})

    # Define a new flag but with the same short_name.
    _defines.DEFINE_integer(
        'changelist',
        0,
        '',
        flag_values=fv,
        short_name='c',
        allow_override=True)
    old_changelist_flag = fv['changelist']
    fv.register_key_flag_for_module(module_name, old_changelist_flag)
    # The short named flag -c is overridden to be the old_changelist_flag.
    self.assertEqual(fv['c'], old_changelist_flag)
    self.assertNotEqual(fv['c'], new_cores_flag)
    self.assertEqual(fv.flags_by_module_dict(),
                     {module_name: [new_cores_flag, old_changelist_flag]})
    self.assertEqual(fv.flags_by_module_id_dict(),
                     {id(module): [new_cores_flag, old_changelist_flag]})
    self.assertEqual(fv.key_flags_by_module_dict(),
                     {module_name: [old_changelist_flag]})

    # Define a flag only with the same long name.
    _defines.DEFINE_integer(
        'changelist',
        0,
        '',
        flag_values=fv,
        short_name='l',
        allow_override=True)
    new_changelist_flag = fv['changelist']
    self.assertNotEqual(old_changelist_flag, new_changelist_flag)
    self.assertEqual(fv.flags_by_module_dict(),
                     {module_name: [new_cores_flag,
                                    old_changelist_flag,
                                    new_changelist_flag]})
    self.assertEqual(fv.flags_by_module_id_dict(),
                     {id(module): [new_cores_flag,
                                   old_changelist_flag,
                                   new_changelist_flag]})
    self.assertEqual(fv.key_flags_by_module_dict(),
                     {module_name: [old_changelist_flag]})

    # Delete the new changelist's long name, it should still be registered
    # because of its short name.
    del fv.changelist
    self.assertNotIn('changelist', fv)
    self.assertEqual(fv.flags_by_module_dict(),
                     {module_name: [new_cores_flag,
                                    old_changelist_flag,
                                    new_changelist_flag]})
    self.assertEqual(fv.flags_by_module_id_dict(),
                     {id(module): [new_cores_flag,
                                   old_changelist_flag,
                                   new_changelist_flag]})
    self.assertEqual(fv.key_flags_by_module_dict(),
                     {module_name: [old_changelist_flag]})

    # Delete the new changelist's short name, it should be removed.
    del fv.l
    self.assertNotIn('l', fv)
    self.assertEqual(fv.flags_by_module_dict(),
                     {module_name: [new_cores_flag,
                                    old_changelist_flag]})
    self.assertEqual(fv.flags_by_module_id_dict(),
                     {id(module): [new_cores_flag,
                                   old_changelist_flag]})
    self.assertEqual(fv.key_flags_by_module_dict(),
                     {module_name: [old_changelist_flag]})

  def _test_find_module_or_id_defining_flag(self, test_id):
    """Tests for find_module_defining_flag and find_module_id_defining_flag.

    Args:
      test_id: True to test find_module_id_defining_flag, False to test
          find_module_defining_flag.
    """
    fv = _flagvalues.FlagValues()
    current_module, current_module_name = (
        _helpers.get_calling_module_object_and_name())
    alt_module_name = _flagvalues.__name__

    if test_id:
      current_module_or_id = id(current_module)
      alt_module_or_id = id(_flagvalues)
      testing_fn = fv.find_module_id_defining_flag
    else:
      current_module_or_id = current_module_name
      alt_module_or_id = alt_module_name
      testing_fn = fv.find_module_defining_flag

    # Define first flag.
    _defines.DEFINE_integer('cores', 4, '', flag_values=fv, short_name='c')
    module_or_id_cores = testing_fn('cores')
    self.assertEqual(module_or_id_cores, current_module_or_id)
    module_or_id_c = testing_fn('c')
    self.assertEqual(module_or_id_c, current_module_or_id)

    # Redefine the same flag in another module.
    _defines.DEFINE_integer(
        'cores',
        4,
        '',
        flag_values=fv,
        module_name=alt_module_name,
        short_name='c',
        allow_override=True)
    module_or_id_cores = testing_fn('cores')
    self.assertEqual(module_or_id_cores, alt_module_or_id)
    module_or_id_c = testing_fn('c')
    self.assertEqual(module_or_id_c, alt_module_or_id)

    # Define a new flag but with the same short_name.
    _defines.DEFINE_integer(
        'changelist',
        0,
        '',
        flag_values=fv,
        short_name='c',
        allow_override=True)
    module_or_id_cores = testing_fn('cores')
    self.assertEqual(module_or_id_cores, alt_module_or_id)
    module_or_id_changelist = testing_fn('changelist')
    self.assertEqual(module_or_id_changelist, current_module_or_id)
    module_or_id_c = testing_fn('c')
    self.assertEqual(module_or_id_c, current_module_or_id)

    # Define a flag in another module only with the same long name.
    _defines.DEFINE_integer(
        'changelist',
        0,
        '',
        flag_values=fv,
        module_name=alt_module_name,
        short_name='l',
        allow_override=True)
    module_or_id_cores = testing_fn('cores')
    self.assertEqual(module_or_id_cores, alt_module_or_id)
    module_or_id_changelist = testing_fn('changelist')
    self.assertEqual(module_or_id_changelist, alt_module_or_id)
    module_or_id_c = testing_fn('c')
    self.assertEqual(module_or_id_c, current_module_or_id)
    module_or_id_l = testing_fn('l')
    self.assertEqual(module_or_id_l, alt_module_or_id)

    # Delete the changelist flag, its short name should still be registered.
    del fv.changelist
    module_or_id_changelist = testing_fn('changelist')
    self.assertEqual(module_or_id_changelist, None)
    module_or_id_c = testing_fn('c')
    self.assertEqual(module_or_id_c, current_module_or_id)
    module_or_id_l = testing_fn('l')
    self.assertEqual(module_or_id_l, alt_module_or_id)

  def test_find_module_defining_flag(self):
    self._test_find_module_or_id_defining_flag(test_id=False)

  def test_find_module_id_defining_flag(self):
    self._test_find_module_or_id_defining_flag(test_id=True)

  def test_set_default(self):
    fv = _flagvalues.FlagValues()
    fv.mark_as_parsed()
    with self.assertRaises(_exceptions.UnrecognizedFlagError):
      fv.set_default('changelist', 1)
    _defines.DEFINE_integer('changelist', 0, 'help', flag_values=fv)
    self.assertEqual(0, fv.changelist)
    fv.set_default('changelist', 2)
    self.assertEqual(2, fv.changelist)

  def test_default_gnu_getopt_value(self):
    self.assertTrue(_flagvalues.FlagValues().is_gnu_getopt())

  def test_known_only_flags_in_gnustyle(self):

    def run_test(argv, defined_py_flags, expected_argv):
      fv = _flagvalues.FlagValues()
      fv.set_gnu_getopt(True)
      for f in defined_py_flags:
        if f.startswith('b'):
          _defines.DEFINE_boolean(f, False, 'help', flag_values=fv)
        else:
          _defines.DEFINE_string(f, 'default', 'help', flag_values=fv)
      output_argv = fv(argv, known_only=True)
      self.assertEqual(expected_argv, output_argv)

    run_test(
        argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '),
        defined_py_flags=[],
        expected_argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '))
    run_test(
        argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '),
        defined_py_flags=['f1'],
        expected_argv='0 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '))
    run_test(
        argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '),
        defined_py_flags=['f2'],
        expected_argv='0 --f1=v1 cmd --b1 --f3 v3 --nob2'.split(' '))
    run_test(
        argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '),
        defined_py_flags=['b1'],
        expected_argv='0 --f1=v1 cmd --f2 v2 --f3 v3 --nob2'.split(' '))
    run_test(
        argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '),
        defined_py_flags=['f3'],
        expected_argv='0 --f1=v1 cmd --f2 v2 --b1 --nob2'.split(' '))
    run_test(
        argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3 --nob2'.split(' '),
        defined_py_flags=['b2'],
        expected_argv='0 --f1=v1 cmd --f2 v2 --b1 --f3 v3'.split(' '))
    run_test(
        argv=('0 --f1=v1 cmd --undefok=f1 --f2 v2 --b1 '
              '--f3 v3 --nob2').split(' '),
        defined_py_flags=['b2'],
        expected_argv='0 cmd --f2 v2 --b1 --f3 v3'.split(' '))
    run_test(
        argv=('0 --f1=v1 cmd --undefok f1,f2 --f2 v2 --b1 '
              '--f3 v3 --nob2').split(' '),
        defined_py_flags=['b2'],
        # Note v2 is preserved here, since undefok requires the flag being
        # specified in the form of --flag=value.
        expected_argv='0 cmd v2 --b1 --f3 v3'.split(' '))

  def test_invalid_flag_name(self):
    with self.assertRaises(_exceptions.Error):
      _defines.DEFINE_boolean('test ', 0, '')

    with self.assertRaises(_exceptions.Error):
      _defines.DEFINE_boolean(' test', 0, '')

    with self.assertRaises(_exceptions.Error):
      _defines.DEFINE_boolean('te st', 0, '')

    with self.assertRaises(_exceptions.Error):
      _defines.DEFINE_boolean('', 0, '')

    with self.assertRaises(_exceptions.Error):
      _defines.DEFINE_boolean(1, 0, '')

  def test_len(self):
    fv = _flagvalues.FlagValues()
    self.assertEqual(0, len(fv))
    self.assertFalse(fv)

    _defines.DEFINE_boolean('boolean', False, 'help', flag_values=fv)
    self.assertEqual(1, len(fv))
    self.assertTrue(fv)

    _defines.DEFINE_boolean(
        'bool', False, 'help', short_name='b', flag_values=fv)
    self.assertEqual(3, len(fv))
    self.assertTrue(fv)

  def test_pickle(self):
    fv = _flagvalues.FlagValues()
    with self.assertRaisesRegexp(TypeError, "can't pickle FlagValues"):
      pickle.dumps(fv)

  def test_copy(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_integer('answer', 0, 'help', flag_values=fv)
    fv(['', '--answer=1'])

    with self.assertRaisesRegexp(
        TypeError, 'FlagValues does not support shallow copies'):
      copy.copy(fv)

    fv2 = copy.deepcopy(fv)
    self.assertEqual(fv2.answer, 1)

    fv2.answer = 42
    self.assertEqual(fv2.answer, 42)
    self.assertEqual(fv.answer, 1)

  def test_conflicting_flags(self):
    fv = _flagvalues.FlagValues()
    with self.assertRaises(_exceptions.FlagNameConflictsWithMethodError):
      _defines.DEFINE_boolean('is_gnu_getopt', False, 'help', flag_values=fv)
    _defines.DEFINE_boolean(
        'is_gnu_getopt',
        False,
        'help',
        flag_values=fv,
        allow_using_method_names=True)
    self.assertFalse(fv['is_gnu_getopt'].value)
    self.assertIsInstance(fv.is_gnu_getopt, types.MethodType)

  def test_get_help(self):
    fv = _flagvalues.FlagValues()
    self.assertMultiLineEqual('''\
--flagfile: Insert flag definitions from the given file into the command line.
  (default: '')
--undefok: comma-separated list of flag names that it is okay to specify on the
  command line even if the program does not define a flag with that name.
  IMPORTANT: flags in this list that have arguments MUST use the --flag=value
  format.
  (default: '')''', fv.get_help())

    module_foo.define_flags(fv)
    self.assertMultiLineEqual('''
absl.flags.tests.module_bar:
  --tmod_bar_t: Sample int flag.
    (default: '4')
    (an integer)
  --tmod_bar_u: Sample int flag.
    (default: '5')
    (an integer)
  --tmod_bar_v: Sample int flag.
    (default: '6')
    (an integer)
  --[no]tmod_bar_x: Boolean flag.
    (default: 'true')
  --tmod_bar_y: String flag.
    (default: 'default')
  --[no]tmod_bar_z: Another boolean flag from module bar.
    (default: 'false')

absl.flags.tests.module_foo:
  --[no]tmod_foo_bool: Boolean flag from module foo.
    (default: 'true')
  --tmod_foo_int: Sample int flag.
    (default: '3')
    (an integer)
  --tmod_foo_str: String flag.
    (default: 'default')

absl.flags:
  --flagfile: Insert flag definitions from the given file into the command line.
    (default: '')
  --undefok: comma-separated list of flag names that it is okay to specify on
    the command line even if the program does not define a flag with that name.
    IMPORTANT: flags in this list that have arguments MUST use the --flag=value
    format.
    (default: '')''', fv.get_help())

    self.assertMultiLineEqual('''
xxxxabsl.flags.tests.module_bar:
xxxx  --tmod_bar_t: Sample int flag.
xxxx    (default: '4')
xxxx    (an integer)
xxxx  --tmod_bar_u: Sample int flag.
xxxx    (default: '5')
xxxx    (an integer)
xxxx  --tmod_bar_v: Sample int flag.
xxxx    (default: '6')
xxxx    (an integer)
xxxx  --[no]tmod_bar_x: Boolean flag.
xxxx    (default: 'true')
xxxx  --tmod_bar_y: String flag.
xxxx    (default: 'default')
xxxx  --[no]tmod_bar_z: Another boolean flag from module bar.
xxxx    (default: 'false')

xxxxabsl.flags.tests.module_foo:
xxxx  --[no]tmod_foo_bool: Boolean flag from module foo.
xxxx    (default: 'true')
xxxx  --tmod_foo_int: Sample int flag.
xxxx    (default: '3')
xxxx    (an integer)
xxxx  --tmod_foo_str: String flag.
xxxx    (default: 'default')

xxxxabsl.flags:
xxxx  --flagfile: Insert flag definitions from the given file into the command
xxxx    line.
xxxx    (default: '')
xxxx  --undefok: comma-separated list of flag names that it is okay to specify
xxxx    on the command line even if the program does not define a flag with that
xxxx    name.  IMPORTANT: flags in this list that have arguments MUST use the
xxxx    --flag=value format.
xxxx    (default: '')''', fv.get_help(prefix='xxxx'))

    self.assertMultiLineEqual('''
absl.flags.tests.module_bar:
  --tmod_bar_t: Sample int flag.
    (default: '4')
    (an integer)
  --tmod_bar_u: Sample int flag.
    (default: '5')
    (an integer)
  --tmod_bar_v: Sample int flag.
    (default: '6')
    (an integer)
  --[no]tmod_bar_x: Boolean flag.
    (default: 'true')
  --tmod_bar_y: String flag.
    (default: 'default')
  --[no]tmod_bar_z: Another boolean flag from module bar.
    (default: 'false')

absl.flags.tests.module_foo:
  --[no]tmod_foo_bool: Boolean flag from module foo.
    (default: 'true')
  --tmod_foo_int: Sample int flag.
    (default: '3')
    (an integer)
  --tmod_foo_str: String flag.
    (default: 'default')''', fv.get_help(include_special_flags=False))

  def test_str(self):
    fv = _flagvalues.FlagValues()
    self.assertEqual(str(fv), fv.get_help())
    module_foo.define_flags(fv)
    self.assertEqual(str(fv), fv.get_help())

  def test_empty_argv(self):
    fv = _flagvalues.FlagValues()
    with self.assertRaises(ValueError):
      fv([])

  def test_invalid_argv(self):
    fv = _flagvalues.FlagValues()
    with self.assertRaises(TypeError):
      fv('./program')
    with self.assertRaises(TypeError):
      fv(b'./program')
    with self.assertRaises(TypeError):
      fv(u'./program')

  def test_flags_dir(self):
    flag_values = _flagvalues.FlagValues()
    flag_name1 = 'bool_flag'
    flag_name2 = 'string_flag'
    flag_name3 = 'float_flag'
    description = 'Description'
    _defines.DEFINE_boolean(
        flag_name1, None, description, flag_values=flag_values)
    _defines.DEFINE_string(
        flag_name2, None, description, flag_values=flag_values)
    self.assertEqual(sorted([flag_name1, flag_name2]), dir(flag_values))

    _defines.DEFINE_float(
        flag_name3, None, description, flag_values=flag_values)
    self.assertEqual(
        sorted([flag_name1, flag_name2, flag_name3]), dir(flag_values))

  def test_flags_into_string_deterministic(self):
    flag_values = _flagvalues.FlagValues()
    _defines.DEFINE_string(
        'fa', 'x', '', flag_values=flag_values, module_name='mb')
    _defines.DEFINE_string(
        'fb', 'x', '', flag_values=flag_values, module_name='mb')
    _defines.DEFINE_string(
        'fc', 'x', '', flag_values=flag_values, module_name='ma')
    _defines.DEFINE_string(
        'fd', 'x', '', flag_values=flag_values, module_name='ma')

    expected = ('--fc=x\n'
                '--fd=x\n'
                '--fa=x\n'
                '--fb=x\n')

    flags_by_module_items = sorted(
        flag_values.flags_by_module_dict().items(), reverse=True)
    for _, module_flags in flags_by_module_items:
      module_flags.sort(reverse=True)

    flag_values.__dict__['__flags_by_module'] = collections.OrderedDict(
        flags_by_module_items)

    actual = flag_values.flags_into_string()
    self.assertEqual(expected, actual)


class FlagValuesLoggingTest(absltest.TestCase):
  """Test to make sure logging.* functions won't recurse.

  Logging may and does happen before flags initialization. We need to make
  sure that any warnings trown by flagvalues do not result in unlimited
  recursion.
  """

  def test_logging_do_not_recurse(self):
    logging.info('test info')
    try:
      raise ValueError('test exception')
    except ValueError:
      logging.exception('test message')


class FlagSubstrMatchingTests(parameterized.TestCase):
  """Tests related to flag substring matching."""

  def _get_test_flag_values(self):
    """Get a _flagvalues.FlagValues() instance, set up for tests."""
    flag_values = _flagvalues.FlagValues()

    _defines.DEFINE_string('strf', '', '', flag_values=flag_values)
    _defines.DEFINE_boolean('boolf', 0, '', flag_values=flag_values)

    return flag_values

  # Test cases that should always make parsing raise an error.
  # Tuples of strings with the argv to use.
  FAIL_TEST_CASES = [
      ('./program', '--boo', '0'),
      ('./program', '--boo=true', '0'),
      ('./program', '--boo=0'),
      ('./program', '--noboo'),
      ('./program', '--st=blah'),
      ('./program', '--st=de'),
      ('./program', '--st=blah', '--boo'),
      ('./program', '--st=blah', 'unused'),
      ('./program', '--st=--blah'),
      ('./program', '--st', '--blah'),
  ]

  @parameterized.parameters(FAIL_TEST_CASES)
  def test_raise(self, *argv):
    """Test that raising works."""
    fv = self._get_test_flag_values()
    with self.assertRaises(_exceptions.UnrecognizedFlagError):
      fv(argv)

  @parameterized.parameters(
      FAIL_TEST_CASES + [('./program', 'unused', '--st=blah')])
  def test_gnu_getopt_raise(self, *argv):
    """Test that raising works when combined with GNU-style getopt."""
    fv = self._get_test_flag_values()
    fv.set_gnu_getopt()
    with self.assertRaises(_exceptions.UnrecognizedFlagError):
      fv(argv)


class SettingUnknownFlagTest(absltest.TestCase):

  def setUp(self):
    self.setter_called = 0

  def set_undef(self, unused_name, unused_val):
    self.setter_called += 1

  def test_raise_on_undefined(self):
    new_flags = _flagvalues.FlagValues()
    with self.assertRaises(_exceptions.UnrecognizedFlagError):
      new_flags.undefined_flag = 0

  def test_not_raise(self):
    new_flags = _flagvalues.FlagValues()
    new_flags._register_unknown_flag_setter(self.set_undef)
    new_flags.undefined_flag = 0
    self.assertEqual(self.setter_called, 1)

  def test_not_raise_on_undefined_if_undefok(self):
    new_flags = _flagvalues.FlagValues()
    args = ['0', '--foo', '--bar=1', '--undefok=foo,bar']
    unparsed = new_flags(args, known_only=True)
    self.assertEqual(['0'], unparsed)

  def test_re_raise_undefined(self):
    def setter(unused_name, unused_val):
      raise NameError()
    new_flags = _flagvalues.FlagValues()
    new_flags._register_unknown_flag_setter(setter)
    with self.assertRaises(_exceptions.UnrecognizedFlagError):
      new_flags.undefined_flag = 0

  def test_re_raise_invalid(self):
    def setter(unused_name, unused_val):
      raise ValueError()
    new_flags = _flagvalues.FlagValues()
    new_flags._register_unknown_flag_setter(setter)
    with self.assertRaises(_exceptions.IllegalFlagValueError):
      new_flags.undefined_flag = 0


class FlagsDashSyntaxTest(absltest.TestCase):

  def setUp(self):
    self.fv = _flagvalues.FlagValues()
    _defines.DEFINE_string(
        'long_name', 'default', 'help', flag_values=self.fv, short_name='s')

  def test_long_name_one_dash(self):
    self.fv(['./program', '-long_name=new'])
    self.assertEqual('new', self.fv.long_name)

  def test_long_name_two_dashes(self):
    self.fv(['./program', '--long_name=new'])
    self.assertEqual('new', self.fv.long_name)

  def test_long_name_three_dashes(self):
    with self.assertRaises(_exceptions.UnrecognizedFlagError):
      self.fv(['./program', '---long_name=new'])

  def test_short_name_one_dash(self):
    self.fv(['./program', '-s=new'])
    self.assertEqual('new', self.fv.s)

  def test_short_name_two_dashes(self):
    self.fv(['./program', '--s=new'])
    self.assertEqual('new', self.fv.s)

  def test_short_name_three_dashes(self):
    with self.assertRaises(_exceptions.UnrecognizedFlagError):
      self.fv(['./program', '---s=new'])


class UnparseFlagsTest(absltest.TestCase):

  def test_using_default_value_none(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_string('default_none', None, 'help', flag_values=fv)
    self.assertTrue(fv['default_none'].using_default_value)
    fv(['', '--default_none=notNone'])
    self.assertFalse(fv['default_none'].using_default_value)
    fv.unparse_flags()
    self.assertTrue(fv['default_none'].using_default_value)
    fv(['', '--default_none=alsoNotNone'])
    self.assertFalse(fv['default_none'].using_default_value)
    fv.unparse_flags()
    self.assertTrue(fv['default_none'].using_default_value)

  def test_using_default_value_not_none(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_string('default_foo', 'foo', 'help', flag_values=fv)

    fv.mark_as_parsed()
    self.assertTrue(fv['default_foo'].using_default_value)

    fv(['', '--default_foo=foo'])
    self.assertFalse(fv['default_foo'].using_default_value)

    fv(['', '--default_foo=notFoo'])
    self.assertFalse(fv['default_foo'].using_default_value)

    fv.unparse_flags()
    self.assertTrue(fv['default_foo'].using_default_value)

    fv(['', '--default_foo=alsoNotFoo'])
    self.assertFalse(fv['default_foo'].using_default_value)

  def test_allow_overwrite_false(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_string(
        'default_none', None, 'help', allow_overwrite=False, flag_values=fv)
    _defines.DEFINE_string(
        'default_foo', 'foo', 'help', allow_overwrite=False, flag_values=fv)

    fv.mark_as_parsed()
    self.assertEqual('foo', fv.default_foo)
    self.assertEqual(None, fv.default_none)

    fv(['', '--default_foo=notFoo', '--default_none=notNone'])
    self.assertEqual('notFoo', fv.default_foo)
    self.assertEqual('notNone', fv.default_none)

    fv.unparse_flags()
    self.assertEqual('foo', fv['default_foo'].value)
    self.assertEqual(None, fv['default_none'].value)

    fv(['', '--default_foo=alsoNotFoo', '--default_none=alsoNotNone'])
    self.assertEqual('alsoNotFoo', fv.default_foo)
    self.assertEqual('alsoNotNone', fv.default_none)

  def test_multi_string_default_none(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_multi_string('foo', None, 'help', flag_values=fv)
    fv.mark_as_parsed()
    self.assertEqual(None, fv.foo)
    fv(['', '--foo=aa'])
    self.assertEqual(['aa'], fv.foo)
    fv.unparse_flags()
    self.assertEqual(None, fv['foo'].value)
    fv(['', '--foo=bb', '--foo=cc'])
    self.assertEqual(['bb', 'cc'], fv.foo)
    fv.unparse_flags()
    self.assertEqual(None, fv['foo'].value)

  def test_multi_string_default_string(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_multi_string('foo', 'xyz', 'help', flag_values=fv)
    expected_default = ['xyz']
    fv.mark_as_parsed()
    self.assertEqual(expected_default, fv.foo)
    fv(['', '--foo=aa'])
    self.assertEqual(['aa'], fv.foo)
    fv.unparse_flags()
    self.assertEqual(expected_default, fv['foo'].value)
    fv(['', '--foo=bb', '--foo=cc'])
    self.assertEqual(['bb', 'cc'], fv['foo'].value)
    fv.unparse_flags()
    self.assertEqual(expected_default, fv['foo'].value)

  def test_multi_string_default_list(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_multi_string(
        'foo', ['xx', 'yy', 'zz'], 'help', flag_values=fv)
    expected_default = ['xx', 'yy', 'zz']
    fv.mark_as_parsed()
    self.assertEqual(expected_default, fv.foo)
    fv(['', '--foo=aa'])
    self.assertEqual(['aa'], fv.foo)
    fv.unparse_flags()
    self.assertEqual(expected_default, fv['foo'].value)
    fv(['', '--foo=bb', '--foo=cc'])
    self.assertEqual(['bb', 'cc'], fv.foo)
    fv.unparse_flags()
    self.assertEqual(expected_default, fv['foo'].value)


class UnparsedFlagAccessTest(absltest.TestCase):

  def test_unparsed_flag_access(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_string('name', 'default', 'help', flag_values=fv)
    with self.assertRaises(_exceptions.UnparsedFlagAccessError):
      _ = fv.name

  @unittest.skipIf(six.PY3, 'Python 2 only test')
  def test_hasattr_logs_in_py2(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_string('name', 'default', 'help', flag_values=fv)
    with mock.patch.object(_flagvalues.logging, 'error') as mock_error:
      self.assertFalse(hasattr(fv, 'name'))
    mock_error.assert_called_once()

  @unittest.skipIf(six.PY2, 'Python 3 only test')
  def test_hasattr_raises_in_py3(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_string('name', 'default', 'help', flag_values=fv)
    with self.assertRaises(_exceptions.UnparsedFlagAccessError):
      _ = hasattr(fv, 'name')

  def test_unparsed_flags_access_raises_after_unparse_flags(self):
    fv = _flagvalues.FlagValues()
    _defines.DEFINE_string('a_str', 'default_value', 'help', flag_values=fv)
    fv.mark_as_parsed()
    self.assertEqual(fv.a_str, 'default_value')
    fv.unparse_flags()
    with self.assertRaises(_exceptions.UnparsedFlagAccessError):
      _ = fv.a_str


if __name__ == '__main__':
  absltest.main()
