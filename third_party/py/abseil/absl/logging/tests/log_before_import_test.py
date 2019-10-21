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

"""Test of logging behavior before app.run(), aka flag and logging init()."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import io
import os
import re
import sys
import tempfile

from absl import logging
from absl.testing import absltest
import mock

logging.get_verbosity()  # Access --verbosity before flag parsing.
# Access --logtostderr before flag parsing.
logging.get_absl_handler().use_absl_log_file()


class Error(Exception):
  pass


@contextlib.contextmanager
def captured_stderr_filename():
  """Captures stderr and writes them to a temporary file.

  This uses os.dup/os.dup2 to redirect the stderr fd for capturing standard
  error of logging at import-time. We cannot mock sys.stderr because on the
  first log call, a default log handler writing to the mock sys.stderr is
  registered, and it will never be removed and subsequent logs go to the mock
  in addition to the real stder.

  Yields:
    The filename of captured stderr.
  """
  stderr_capture_file_fd, stderr_capture_file_name = tempfile.mkstemp()
  original_stderr_fd = os.dup(sys.stderr.fileno())
  os.dup2(stderr_capture_file_fd, sys.stderr.fileno())
  try:
    yield stderr_capture_file_name
  finally:
    os.close(stderr_capture_file_fd)
    os.dup2(original_stderr_fd, sys.stderr.fileno())


# Pre-initialization (aka "import" / __main__ time) test.
with captured_stderr_filename() as before_set_verbosity_filename:
  # Warnings and above go to stderr.
  logging.debug('Debug message at parse time.')
  logging.info('Info message at parse time.')
  logging.error('Error message at parse time.')
  logging.warning('Warning message at parse time.')
  try:
    raise Error('Exception reason.')
  except Error:
    logging.exception('Exception message at parse time.')


logging.set_verbosity(logging.ERROR)
with captured_stderr_filename() as after_set_verbosity_filename:
  # Verbosity is set to ERROR, errors and above go to stderr.
  logging.debug('Debug message at parse time.')
  logging.info('Info message at parse time.')
  logging.warning('Warning message at parse time.')
  logging.error('Error message at parse time.')


class LoggingInitWarningTest(absltest.TestCase):

  def test_captured_pre_init_warnings(self):
    with open(before_set_verbosity_filename) as stderr_capture_file:
      captured_stderr = stderr_capture_file.read()
    self.assertNotIn('Debug message at parse time.', captured_stderr)
    self.assertNotIn('Info message at parse time.', captured_stderr)

    traceback_re = re.compile(
        r'\nTraceback \(most recent call last\):.*?Error: Exception reason.',
        re.MULTILINE | re.DOTALL)
    if not traceback_re.search(captured_stderr):
      self.fail(
          'Cannot find traceback message from logging.exception '
          'in stderr:\n{}'.format(captured_stderr))
    # Remove the traceback so the rest of the stderr is deterministic.
    captured_stderr = traceback_re.sub('', captured_stderr)
    captured_stderr_lines = captured_stderr.splitlines()
    self.assertLen(captured_stderr_lines, 3)
    self.assertIn('Error message at parse time.', captured_stderr_lines[0])
    self.assertIn('Warning message at parse time.', captured_stderr_lines[1])
    self.assertIn('Exception message at parse time.', captured_stderr_lines[2])

  def test_set_verbosity_pre_init(self):
    with open(after_set_verbosity_filename) as stderr_capture_file:
      captured_stderr = stderr_capture_file.read()
    captured_stderr_lines = captured_stderr.splitlines()

    self.assertNotIn('Debug message at parse time.', captured_stderr)
    self.assertNotIn('Info message at parse time.', captured_stderr)
    self.assertNotIn('Warning message at parse time.', captured_stderr)
    self.assertLen(captured_stderr_lines, 1)
    self.assertIn('Error message at parse time.', captured_stderr_lines[0])

  def test_no_more_warnings(self):
    fake_stderr_type = io.BytesIO if bytes is str else io.StringIO
    with mock.patch('sys.stderr', new=fake_stderr_type()) as mock_stderr:
      self.assertMultiLineEqual('', mock_stderr.getvalue())
      logging.warning('Hello. hello. hello. Is there anybody out there?')
      self.assertNotIn('Logging before flag parsing goes to stderr',
                       mock_stderr.getvalue())
    logging.info('A major purpose of this executable is merely not to crash.')


if __name__ == '__main__':
  absltest.main()  # This calls the app.run() init equivalent.
