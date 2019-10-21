# Copyright 2018 The Abseil Authors.
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

"""Tests for absl.flags.argparse_flags."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import sys
import tempfile

from absl import flags
from absl import logging
from absl.flags import argparse_flags
from absl.testing import _bazelize_command
from absl.testing import absltest
from absl.testing import parameterized
import mock
import six


FLAGS = flags.FLAGS


class ArgparseFlagsTest(parameterized.TestCase):

  def setUp(self):
    self._absl_flags = flags.FlagValues()
    flags.DEFINE_bool(
        'absl_bool', None, 'help for --absl_bool.',
        short_name='b', flag_values=self._absl_flags)
    # Add a boolean flag that starts with "no", to verify it can correctly
    # handle the "no" prefixes in boolean flags.
    flags.DEFINE_bool(
        'notice', None, 'help for --notice.',
        flag_values=self._absl_flags)
    flags.DEFINE_string(
        'absl_string', 'default', 'help for --absl_string=%.',
        short_name='s', flag_values=self._absl_flags)
    flags.DEFINE_integer(
        'absl_integer', 1, 'help for --absl_integer.',
        flag_values=self._absl_flags)
    flags.DEFINE_float(
        'absl_float', 1, 'help for --absl_integer.',
        flag_values=self._absl_flags)
    flags.DEFINE_enum(
        'absl_enum', 'apple', ['apple', 'orange'], 'help for --absl_enum.',
        flag_values=self._absl_flags)

  def test_dash_as_prefix_char_only(self):
    with self.assertRaises(ValueError):
      argparse_flags.ArgumentParser(prefix_chars='/')

  def test_default_inherited_absl_flags_value(self):
    parser = argparse_flags.ArgumentParser()
    self.assertIs(parser._inherited_absl_flags, flags.FLAGS)

  def test_parse_absl_flags(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    self.assertFalse(self._absl_flags.is_parsed())
    self.assertTrue(self._absl_flags['absl_string'].using_default_value)
    self.assertTrue(self._absl_flags['absl_integer'].using_default_value)
    self.assertTrue(self._absl_flags['absl_float'].using_default_value)
    self.assertTrue(self._absl_flags['absl_enum'].using_default_value)

    parser.parse_args(
        ['--absl_string=new_string', '--absl_integer', '2'])
    self.assertEqual(self._absl_flags.absl_string, 'new_string')
    self.assertEqual(self._absl_flags.absl_integer, 2)
    self.assertTrue(self._absl_flags.is_parsed())
    self.assertFalse(self._absl_flags['absl_string'].using_default_value)
    self.assertFalse(self._absl_flags['absl_integer'].using_default_value)
    self.assertTrue(self._absl_flags['absl_float'].using_default_value)
    self.assertTrue(self._absl_flags['absl_enum'].using_default_value)

  @parameterized.named_parameters(
      ('true', ['--absl_bool'], True),
      ('false', ['--noabsl_bool'], False),
      ('does_not_accept_equal_value', ['--absl_bool=true'], SystemExit),
      ('does_not_accept_space_value', ['--absl_bool', 'true'], SystemExit),
      ('long_name_single_dash', ['-absl_bool'], SystemExit),
      ('short_name', ['-b'], True),
      ('short_name_false', ['-nob'], SystemExit),
      ('short_name_double_dash', ['--b'], SystemExit),
      ('short_name_double_dash_false', ['--nob'], SystemExit),
  )
  def test_parse_boolean_flags(self, args, expected):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    self.assertIsNone(self._absl_flags['absl_bool'].value)
    self.assertIsNone(self._absl_flags['b'].value)
    if isinstance(expected, bool):
      parser.parse_args(args)
      self.assertEqual(expected, self._absl_flags.absl_bool)
      self.assertEqual(expected, self._absl_flags.b)
    else:
      with self.assertRaises(expected):
        parser.parse_args(args)

  @parameterized.named_parameters(
      ('true', ['--notice'], True),
      ('false', ['--nonotice'], False),
  )
  def test_parse_boolean_existing_no_prefix(self, args, expected):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    self.assertIsNone(self._absl_flags['notice'].value)
    parser.parse_args(args)
    self.assertEqual(expected, self._absl_flags.notice)

  def test_unrecognized_flag(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    with self.assertRaises(SystemExit):
      parser.parse_args(['--unknown_flag=what'])

  def test_absl_validators(self):

    @flags.validator('absl_integer', flag_values=self._absl_flags)
    def ensure_positive(value):
      return value > 0

    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    with self.assertRaises(SystemExit):
      parser.parse_args(['--absl_integer', '-2'])

    del ensure_positive

  @parameterized.named_parameters(
      ('regular_name_double_dash', '--absl_string=new_string', 'new_string'),
      ('regular_name_single_dash', '-absl_string=new_string', SystemExit),
      ('short_name_double_dash', '--s=new_string', SystemExit),
      ('short_name_single_dash', '-s=new_string', 'new_string'),
  )
  def test_dashes(self, argument, expected):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    if isinstance(expected, six.string_types):
      parser.parse_args([argument])
      self.assertEqual(self._absl_flags.absl_string, expected)
    else:
      with self.assertRaises(expected):
        parser.parse_args([argument])

  def test_absl_flags_not_added_to_namespace(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    args = parser.parse_args(['--absl_string=new_string'])
    self.assertIsNone(getattr(args, 'absl_string', None))

  def test_mixed_flags_and_positional(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    parser.add_argument('--header', help='Header message to print.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')

    args = parser.parse_args(
        ['--absl_string=new_string', '--header=HEADER', '--absl_integer',
         '2', '3', '4'])
    self.assertEqual(self._absl_flags.absl_string, 'new_string')
    self.assertEqual(self._absl_flags.absl_integer, 2)
    self.assertEqual(args.header, 'HEADER')
    self.assertListEqual(args.integers, [3, 4])

  def test_subparsers(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    parser.add_argument('--header', help='Header message to print.')
    subparsers = parser.add_subparsers(help='The command to execute.')

    sub_parser = subparsers.add_parser(
        'sub_cmd', help='Sub command.', inherited_absl_flags=self._absl_flags)
    sub_parser.add_argument('--sub_flag', help='Sub command flag.')

    def sub_command_func():
      pass

    sub_parser.set_defaults(command=sub_command_func)

    args = parser.parse_args([
        '--header=HEADER', '--absl_string=new_value', 'sub_cmd',
        '--absl_integer=2', '--sub_flag=new_sub_flag_value'])

    self.assertEqual(args.header, 'HEADER')
    self.assertEqual(self._absl_flags.absl_string, 'new_value')
    self.assertEqual(args.command, sub_command_func)
    self.assertEqual(self._absl_flags.absl_integer, 2)
    self.assertEqual(args.sub_flag, 'new_sub_flag_value')

  def test_subparsers_no_inherit_in_subparser(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    subparsers = parser.add_subparsers(help='The command to execute.')

    subparsers.add_parser(
        'sub_cmd', help='Sub command.',
        # Do not inherit absl flags in the subparser.
        # This is the behavior that this test exercises.
        inherited_absl_flags=None)

    with self.assertRaises(SystemExit):
      parser.parse_args(['sub_cmd', '--absl_string=new_value'])

  def test_help_main_module_flags(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    help_message = parser.format_help()

    # Only the short name is shown in the usage string.
    self.assertIn('[-s ABSL_STRING]', help_message)
    # Both names are included in the options section.
    self.assertIn('-s ABSL_STRING, --absl_string ABSL_STRING', help_message)
    # Verify help messages.
    self.assertIn('help for --absl_string=%.', help_message)
    self.assertIn('<apple|orange>: help for --absl_enum.', help_message)

  def test_help_non_main_module_flags(self):
    flags.DEFINE_string(
        'non_main_module_flag', 'default', 'help',
        module_name='other.module', flag_values=self._absl_flags)
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    help_message = parser.format_help()

    # Non main module key flags are not printed in the help message.
    self.assertNotIn('non_main_module_flag', help_message)

  def test_help_non_main_module_key_flags(self):
    flags.DEFINE_string(
        'non_main_module_flag', 'default', 'help',
        module_name='other.module', flag_values=self._absl_flags)
    flags.declare_key_flag('non_main_module_flag', flag_values=self._absl_flags)
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    help_message = parser.format_help()

    # Main module key fags are printed in the help message, even if the flag
    # is defined in another module.
    self.assertIn('non_main_module_flag', help_message)

  @parameterized.named_parameters(
      ('h', ['-h']),
      ('help', ['--help']),
      ('helpshort', ['--helpshort']),
      ('helpfull', ['--helpfull']),
  )
  def test_help_flags(self, args):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    with self.assertRaises(SystemExit):
      parser.parse_args(args)

  @parameterized.named_parameters(
      ('h', ['-h']),
      ('help', ['--help']),
      ('helpshort', ['--helpshort']),
      ('helpfull', ['--helpfull']),
  )
  def test_no_help_flags(self, args):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags, add_help=False)
    with mock.patch.object(parser, 'print_help'):
      with self.assertRaises(SystemExit):
        parser.parse_args(args)
      parser.print_help.assert_not_called()

  def test_helpfull_message(self):
    flags.DEFINE_string(
        'non_main_module_flag', 'default', 'help',
        module_name='other.module', flag_values=self._absl_flags)
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    with self.assertRaises(SystemExit),\
        mock.patch.object(sys, 'stdout', new=six.StringIO()) as mock_stdout:
      parser.parse_args(['--helpfull'])
    stdout_message = mock_stdout.getvalue()
    logging.info('captured stdout message:\n%s', stdout_message)
    self.assertIn('--non_main_module_flag', stdout_message)
    self.assertIn('other.module', stdout_message)
    # Make sure the main module is not included.
    self.assertNotIn(sys.argv[0], stdout_message)
    # Special flags defined in absl.flags.
    self.assertIn('absl.flags:', stdout_message)
    self.assertIn('--flagfile', stdout_message)
    self.assertIn('--undefok', stdout_message)

  @parameterized.named_parameters(
      ('at_end',
       ('1', '--absl_string=value_from_cmd', '--flagfile='),
       'value_from_file'),
      ('at_beginning',
       ('--flagfile=', '1', '--absl_string=value_from_cmd'),
       'value_from_cmd'),
  )
  def test_flagfile(self, cmd_args, expected_absl_string_value):
    # Set gnu_getopt to False, to verify it's ignored by argparse_flags.
    self._absl_flags.set_gnu_getopt(False)

    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    parser.add_argument('--header', help='Header message to print.')
    parser.add_argument('integers', metavar='N', type=int, nargs='+',
                        help='an integer for the accumulator')
    flagfile = tempfile.NamedTemporaryFile(dir=FLAGS.test_tmpdir, delete=False)
    self.addCleanup(os.unlink, flagfile.name)
    with flagfile:
      flagfile.write(b'''
# The flag file.
--absl_string=value_from_file
--absl_integer=1
--header=header_from_file
''')

    expand_flagfile = lambda x: x + flagfile.name if x == '--flagfile=' else x
    cmd_args = [expand_flagfile(x) for x in cmd_args]
    args = parser.parse_args(cmd_args)

    self.assertEqual([1], args.integers)
    self.assertEqual('header_from_file', args.header)
    self.assertEqual(expected_absl_string_value, self._absl_flags.absl_string)

  @parameterized.parameters(
      ('positional', {'positional'}, False),
      ('--not_existed', {'existed'}, False),
      ('--empty', set(), False),
      ('-single_dash', {'single_dash'}, True),
      ('--double_dash', {'double_dash'}, True),
      ('--with_value=value', {'with_value'}, True),
  )
  def test_is_undefok(self, arg, undefok_names, is_undefok):
    self.assertEqual(is_undefok, argparse_flags._is_undefok(arg, undefok_names))

  @parameterized.named_parameters(
      ('single', 'single', ['--single'], []),
      ('multiple', 'first,second', ['--first', '--second'], []),
      ('single_dash', 'dash', ['-dash'], []),
      ('mixed_dash', 'mixed', ['-mixed', '--mixed'], []),
      ('value', 'name', ['--name=value'], []),
      ('boolean_positive', 'bool', ['--bool'], []),
      ('boolean_negative', 'bool', ['--nobool'], []),
      ('left_over', 'strip', ['--first', '--strip', '--last'],
       ['--first', '--last']),
  )
  def test_strip_undefok_args(self, undefok, args, expected_args):
    actual_args = argparse_flags._strip_undefok_args(undefok, args)
    self.assertListEqual(expected_args, actual_args)

  @parameterized.named_parameters(
      ('at_end', ['--unknown', '--undefok=unknown']),
      ('at_beginning', ['--undefok=unknown', '--unknown']),
      ('multiple', ['--unknown', '--undefok=unknown,another_unknown']),
      ('with_value', ['--unknown=value', '--undefok=unknown']),
      ('maybe_boolean', ['--nounknown', '--undefok=unknown']),
      ('with_space', ['--unknown', '--undefok', 'unknown']),
  )
  def test_undefok_flag_correct_use(self, cmd_args):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    args = parser.parse_args(cmd_args)  # Make sure it doesn't raise.
    # Make sure `undefok` is not exposed in namespace.
    sentinel = object()
    self.assertIs(sentinel, getattr(args, 'undefok', sentinel))

  def test_undefok_flag_existing(self):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    parser.parse_args(
        ['--absl_string=new_value', '--undefok=absl_string'])
    self.assertEqual('new_value', self._absl_flags.absl_string)

  @parameterized.named_parameters(
      ('no_equal', ['--unknown', 'value', '--undefok=unknown']),
      ('single_dash', ['--unknown', '-undefok=unknown']),
  )
  def test_undefok_flag_incorrect_use(self, cmd_args):
    parser = argparse_flags.ArgumentParser(
        inherited_absl_flags=self._absl_flags)
    with self.assertRaises(SystemExit):
      parser.parse_args(cmd_args)


class ArgparseWithAppRunTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('simple',
       'main_simple', 'parse_flags_simple',
       ['--argparse_echo=I am argparse.', '--absl_echo=I am absl.'],
       ['I am argparse.', 'I am absl.']),
      ('subcommand_roll_dice',
       'main_subcommands', 'parse_flags_subcommands',
       ['--argparse_echo=I am argparse.', '--absl_echo=I am absl.',
        'roll_dice', '--num_faces=12'],
       ['I am argparse.', 'I am absl.', 'Rolled a dice: ']),
      ('subcommand_shuffle',
       'main_subcommands', 'parse_flags_subcommands',
       ['--argparse_echo=I am argparse.', '--absl_echo=I am absl.',
        'shuffle', 'a', 'b', 'c'],
       ['I am argparse.', 'I am absl.', 'Shuffled: ']),
  )
  def test_argparse_with_app_run(
      self, main_func_name, flags_parser_func_name, args, output_strings):
    env = os.environ.copy()
    env['MAIN_FUNC'] = main_func_name
    env['FLAGS_PARSER_FUNC'] = flags_parser_func_name
    helper = _bazelize_command.get_executable_path(
        'absl/flags/tests/argparse_flags_test_helper', add_version_suffix=False)
    try:
      stdout = subprocess.check_output(
          [helper] + args, env=env, universal_newlines=True)
    except subprocess.CalledProcessError as e:
      error_info = ('ERROR: argparse_helper failed\n'
                    'Command: {}\n'
                    'Exit code: {}\n'
                    '----- output -----\n{}'
                    '------------------')
      error_info = error_info.format(e.cmd, e.returncode,
                                     e.output + '\n' if e.output else '<empty>')
      print(error_info, file=sys.stderr)
      raise

    for output_string in output_strings:
      self.assertIn(output_string, stdout)


if __name__ == '__main__':
  absltest.main()
