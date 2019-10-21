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

"""Tests for app.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import contextlib
import os
import re
import subprocess
import sys
import tempfile
import unittest

from absl import app
from absl import flags
from absl._enum_module import enum
from absl.testing import _bazelize_command
from absl.testing import absltest
from absl.testing import flagsaver
from absl.tests import app_test_helper
import mock
import six


FLAGS = flags.FLAGS

# six.StringIO best reflects the normal behavior of stdout for both py2 and 3.
mock_stdio_type = six.StringIO

_newline_regex = re.compile('(\r\n)|\r')


@contextlib.contextmanager
def patch_main_module_docstring(docstring):
  old_doc = sys.modules['__main__'].__doc__
  sys.modules['__main__'].__doc__ = docstring
  yield
  sys.modules['__main__'].__doc__ = old_doc


def _normalize_newlines(s):
  return re.sub('(\r\n)|\r', '\n', s)


class UnitTests(absltest.TestCase):

  def test_install_exception_handler(self):
    with self.assertRaises(TypeError):
      app.install_exception_handler(1)

  def test_usage(self):
    with mock.patch.object(
        sys, 'stderr', new=mock_stdio_type()) as mock_stderr:
      app.usage()
    self.assertIn(__doc__, mock_stderr.getvalue())
    # Assert that flags are written to stderr.
    self.assertIn('\n  --[no]helpfull:', mock_stderr.getvalue())

  def test_usage_shorthelp(self):
    with mock.patch.object(
        sys, 'stderr', new=mock_stdio_type()) as mock_stderr:
      app.usage(shorthelp=True)
    # Assert that flags are NOT written to stderr.
    self.assertNotIn('  --', mock_stderr.getvalue())

  def test_usage_writeto_stderr(self):
    with mock.patch.object(
        sys, 'stdout', new=mock_stdio_type()) as mock_stdout:
      app.usage(writeto_stdout=True)
    self.assertIn(__doc__, mock_stdout.getvalue())

  def test_usage_detailed_error(self):
    with mock.patch.object(
        sys, 'stderr', new=mock_stdio_type()) as mock_stderr:
      app.usage(detailed_error='BAZBAZ')
    self.assertIn('BAZBAZ', mock_stderr.getvalue())

  def test_usage_exitcode(self):

    # The test environment may not have the correct output encoding,
    # and we can't really change it once we've started the test,
    # so we have to replace it with one that understands unicode.
    if six.PY2:
      stderr = codecs.getwriter('utf8')(sys.stderr)
    else:
      stderr = sys.stderr

    with mock.patch.object(sys, 'stderr', new=stderr):
      try:
        app.usage(exitcode=2)
        self.fail('app.usage(exitcode=1) should raise SystemExit')
      except SystemExit as e:
        self.assertEqual(2, e.code)

  def test_usage_expands_docstring(self):
    with patch_main_module_docstring('Name: %s, %%s'):
      with mock.patch.object(
          sys, 'stderr', new=mock_stdio_type()) as mock_stderr:
        app.usage()
    self.assertIn('Name: {}, %s'.format(sys.argv[0]),
                  mock_stderr.getvalue())

  def test_usage_does_not_expand_bad_docstring(self):
    with patch_main_module_docstring('Name: %s, %%s, %@'):
      with mock.patch.object(
          sys, 'stderr', new=mock_stdio_type()) as mock_stderr:
        app.usage()
    self.assertIn('Name: %s, %%s, %@', mock_stderr.getvalue())

  @flagsaver.flagsaver
  def test_register_and_parse_flags_with_usage_exits_on_only_check_args(self):
    done = app._register_and_parse_flags_with_usage.done
    try:
      app._register_and_parse_flags_with_usage.done = False
      with self.assertRaises(SystemExit):
        app._register_and_parse_flags_with_usage(
            argv=['./program', '--only_check_args'])
    finally:
      app._register_and_parse_flags_with_usage.done = done

  def test_register_and_parse_flags_with_usage_exits_on_second_run(self):
    with self.assertRaises(SystemError):
      app._register_and_parse_flags_with_usage()


class FunctionalTests(absltest.TestCase):
  """Functional tests that use runs app_test_helper."""

  helper_type = 'pure_python'

  def run_helper(self, expect_success,
                 expected_stdout_substring=None, expected_stderr_substring=None,
                 arguments=(),
                 env_overrides=None):
    env = os.environ.copy()
    env['APP_TEST_HELPER_TYPE'] = self.helper_type
    env['PYTHONIOENCODING'] = 'utf8'
    if env_overrides:
      env.update(env_overrides)

    helper = 'absl/tests/app_test_helper_{}'.format(self.helper_type)
    process = subprocess.Popen(
        [_bazelize_command.get_executable_path(helper)] + list(arguments),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=env, universal_newlines=False)
    stdout, stderr = process.communicate()
    # In Python 2, we can't control the encoding used by universal_newline
    # mode, which can cause UnicodeDecodeErrors when subprocess tries to
    # conver the bytes to unicode, so we have to decode it manually.
    stdout = _normalize_newlines(stdout.decode('utf8'))
    stderr = _normalize_newlines(stderr.decode('utf8'))

    message = (u'Command: {command}\n'
               'Exit Code: {exitcode}\n'
               '===== stdout =====\n{stdout}'
               '===== stderr =====\n{stderr}'
               '=================='.format(
                   command=' '.join([helper] + list(arguments)),
                   exitcode=process.returncode,
                   stdout=stdout or '<no output>\n',
                   stderr=stderr or '<no output>\n'))
    if expect_success:
      self.assertEqual(0, process.returncode, msg=message)
    else:
      self.assertNotEqual(0, process.returncode, msg=message)

    if expected_stdout_substring:
      self.assertIn(expected_stdout_substring, stdout, message)
    if expected_stderr_substring:
      self.assertIn(expected_stderr_substring, stderr, message)

    return process.returncode, stdout, stderr

  def test_help(self):
    _, _, stderr = self.run_helper(
        False,
        arguments=['--help'],
        expected_stdout_substring=app_test_helper.__doc__)
    self.assertNotIn('--', stderr)

  def test_helpfull_basic(self):
    self.run_helper(
        False,
        arguments=['--helpfull'],
        # --logtostderr is from absl.logging module.
        expected_stdout_substring='--[no]logtostderr')

  def test_helpfull_unicode_flag_help(self):
    _, stdout, _ = self.run_helper(
        False,
        arguments=['--helpfull'],
        expected_stdout_substring='str_flag_with_unicode_args')

    self.assertIn(u'smile:\U0001F604', stdout)

    if six.PY2:
      # Default values get repr'd, which causes unicode strings to incorrectly
      # render with their escaped values.
      self.assertIn(repr(u'thumb:\U0001F44D'), stdout)
    else:
      # In Python 3, the repr() of a unicode string isn't escaped.
      self.assertIn(u'thumb:\U0001F44D', stdout)

  def test_helpshort(self):
    _, _, stderr = self.run_helper(
        False,
        arguments=['--helpshort'],
        expected_stdout_substring=app_test_helper.__doc__)
    self.assertNotIn('--', stderr)

  def test_custom_main(self):
    self.run_helper(
        True,
        env_overrides={'APP_TEST_CUSTOM_MAIN_FUNC': 'custom_main'},
        expected_stdout_substring='Function called: custom_main.')

  def test_custom_argv(self):
    self.run_helper(
        True,
        expected_stdout_substring='argv: ./program pos_arg1',
        env_overrides={
            'APP_TEST_CUSTOM_ARGV': './program --noraise_exception pos_arg1',
            'APP_TEST_PRINT_ARGV': '1',
        })

  def test_gwq_status_file_on_exception(self):
    if self.helper_type == 'pure_python':
      # Pure python binary does not write to GWQ Status.
      return

    tmpdir = tempfile.mkdtemp(dir=FLAGS.test_tmpdir)
    self.run_helper(
        False,
        arguments=['--raise_exception'],
        env_overrides={'GOOGLE_STATUS_DIR': tmpdir})
    with open(os.path.join(tmpdir, 'STATUS')) as status_file:
      self.assertIn('MyException:', status_file.read())

  @unittest.skipIf(six.PY2,
                   'By default, faulthandler is only available in Python 3.')
  def test_faulthandler_dumps_stack_on_sigsegv(self):
    return_code, _, _ = self.run_helper(
        False,
        expected_stderr_substring='app_test_helper.py", line',
        arguments=['--faulthandler_sigsegv'])
    # sigsegv returns 3 on Windows, and -11 on LINUX/macOS.
    expected_return_code = 3 if os.name == 'nt' else -11
    self.assertEqual(expected_return_code, return_code)

  def test_top_level_exception(self):
    self.run_helper(
        False,
        arguments=['--raise_exception'],
        expected_stderr_substring='MyException')

  def test_only_check_args(self):
    self.run_helper(
        True,
        arguments=['--only_check_args', '--raise_exception'])

  def test_only_check_args_failure(self):
    self.run_helper(
        False,
        arguments=['--only_check_args', '--banana'],
        expected_stderr_substring='FATAL Flags parsing error')

  def test_usage_error(self):
    exitcode, _, _ = self.run_helper(
        False,
        arguments=['--raise_usage_error'],
        expected_stderr_substring=app_test_helper.__doc__)
    self.assertEqual(1, exitcode)

  def test_usage_error_exitcode(self):
    exitcode, _, _ = self.run_helper(
        False,
        arguments=['--raise_usage_error', '--usage_error_exitcode=88'],
        expected_stderr_substring=app_test_helper.__doc__)
    self.assertEqual(88, exitcode)

  def test_exception_handler(self):
    exception_handler_messages = (
        'MyExceptionHandler: first\nMyExceptionHandler: second\n')
    self.run_helper(
        False,
        arguments=['--raise_exception'],
        expected_stdout_substring=exception_handler_messages)

  def test_exception_handler_not_called(self):
    _, _, stdout = self.run_helper(True)
    self.assertNotIn('MyExceptionHandler', stdout)

  def test_print_init_callbacks(self):
    _, stdout, _ = self.run_helper(
        expect_success=True, arguments=['--print_init_callbacks'])
    self.assertIn('before app.run', stdout)
    self.assertIn('during real_main', stdout)


class FlagValuesExternalizationTest(absltest.TestCase):
  """Test to make sure FLAGS can be serialized out and parsed back in."""

  @flagsaver.flagsaver
  def test_nohelp_doesnt_show_help(self):
    with self.assertRaisesWithPredicateMatch(SystemExit,
                                             lambda e: e.code == 1):
      app.run(
          len,
          argv=[
              './program', '--nohelp', '--helpshort=false', '--helpfull=0',
              '--helpxml=f'
          ])

  @flagsaver.flagsaver
  def test_serialize_roundtrip(self):
    # Use the global 'FLAGS' as the source, to ensure all the framework defined
    # flags will go through the round trip process.
    flags.DEFINE_string('testflag', 'testval', 'help', flag_values=FLAGS)

    flags.DEFINE_multi_enum('test_multi_enum_flag',
                            ['x', 'y'], ['x', 'y', 'z'],
                            'Multi enum help.',
                            flag_values=FLAGS)

    class Fruit(enum.Enum):
      APPLE = 1
      ORANGE = 2
      TOMATO = 3
    flags.DEFINE_multi_enum_class('test_multi_enum_class_flag',
                                  ['APPLE', 'TOMATO'], Fruit,
                                  'Fruit help.',
                                  flag_values=FLAGS)

    new_flag_values = flags.FlagValues()
    new_flag_values.append_flag_values(FLAGS)

    FLAGS.testflag = 'roundtrip_me'
    FLAGS.test_multi_enum_flag = ['y', 'z']
    FLAGS.test_multi_enum_class_flag = [Fruit.ORANGE, Fruit.APPLE]
    argv = ['binary_name'] + FLAGS.flags_into_string().splitlines()

    self.assertNotEqual(new_flag_values['testflag'], FLAGS.testflag)
    self.assertNotEqual(new_flag_values['test_multi_enum_flag'],
                        FLAGS.test_multi_enum_flag)
    self.assertNotEqual(new_flag_values['test_multi_enum_class_flag'],
                        FLAGS.test_multi_enum_class_flag)
    new_flag_values(argv)
    self.assertEqual(new_flag_values.testflag, FLAGS.testflag)
    self.assertEqual(new_flag_values.test_multi_enum_flag,
                     FLAGS.test_multi_enum_flag)
    self.assertEqual(new_flag_values.test_multi_enum_class_flag,
                     FLAGS.test_multi_enum_class_flag)
    del FLAGS.testflag
    del FLAGS.test_multi_enum_flag
    del FLAGS.test_multi_enum_class_flag


if __name__ == '__main__':
  absltest.main()
