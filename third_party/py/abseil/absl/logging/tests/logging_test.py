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

"""Unit tests for absl.logging."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import functools
import getpass
import io
import logging as std_logging
import os
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
import unittest

from absl import flags
from absl import logging
from absl.testing import absltest
from absl.testing import flagsaver
from absl.testing import parameterized
import mock
import six
from six.moves import range  # pylint: disable=redefined-builtin

FLAGS = flags.FLAGS

_StreamIO = io.StringIO if six.PY3 else io.BytesIO  # pylint: disable=invalid-name


class ConfigurationTest(absltest.TestCase):
  """Tests the initial logging configuration."""

  def test_logger_and_handler(self):
    absl_logger = std_logging.getLogger('absl')
    self.assertIs(absl_logger, logging.get_absl_logger())
    self.assertTrue(isinstance(absl_logger, logging.ABSLLogger))
    self.assertTrue(
        isinstance(logging.get_absl_handler().python_handler.formatter,
                   logging.PythonFormatter))


class PythonHandlerTest(absltest.TestCase):
  """Tests the PythonHandler class."""

  def setUp(self):
    (year, month, day, hour, minute, sec,
     dunno, dayofyear, dst_flag) = (1979, 10, 21, 18, 17, 16, 3, 15, 0)
    self.now_tuple = (year, month, day, hour, minute, sec,
                      dunno, dayofyear, dst_flag)
    self.python_handler = logging.PythonHandler()

  def tearDown(self):
    mock.patch.stopall()

  @flagsaver.flagsaver(logtostderr=False)
  def test_set_google_log_file_no_log_to_stderr(self):
    with mock.patch.object(self.python_handler, 'start_logging_to_file'):
      self.python_handler.use_absl_log_file()
      self.python_handler.start_logging_to_file.assert_called_once_with(
          program_name=None, log_dir=None)

  @flagsaver.flagsaver(logtostderr=True)
  def test_set_google_log_file_with_log_to_stderr(self):
    self.python_handler.stream = None
    self.python_handler.use_absl_log_file()
    self.assertEqual(sys.stderr, self.python_handler.stream)

  @mock.patch.object(logging, 'find_log_dir_and_names')
  @mock.patch.object(logging.time, 'localtime')
  @mock.patch.object(logging.time, 'time')
  @mock.patch.object(os.path, 'islink')
  @mock.patch.object(os, 'unlink')
  @mock.patch.object(os, 'getpid')
  def test_start_logging_to_file(
      self, mock_getpid, mock_unlink, mock_islink, mock_time,
      mock_localtime, mock_find_log_dir_and_names):
    mock_find_log_dir_and_names.return_value = ('here', 'prog1', 'prog1')
    mock_time.return_value = '12345'
    mock_localtime.return_value = self.now_tuple
    mock_getpid.return_value = 4321
    symlink = os.path.join('here', 'prog1.INFO')
    mock_islink.return_value = True
    with mock.patch.object(
        logging, 'open', return_value=sys.stdout, create=True):
      if getattr(os, 'symlink', None):
        with mock.patch.object(os, 'symlink'):
          self.python_handler.start_logging_to_file()
          mock_unlink.assert_called_once_with(symlink)
          os.symlink.assert_called_once_with(
              'prog1.INFO.19791021-181716.4321', symlink)
      else:
        self.python_handler.start_logging_to_file()

  def test_log_file(self):
    handler = logging.PythonHandler()
    self.assertEqual(sys.stderr, handler.stream)

    stream = mock.Mock()
    handler = logging.PythonHandler(stream)
    self.assertEqual(stream, handler.stream)

  def test_flush(self):
    stream = mock.Mock()
    handler = logging.PythonHandler(stream)
    handler.flush()
    stream.flush.assert_called_once()

  def test_flush_with_value_error(self):
    stream = mock.Mock()
    stream.flush.side_effect = ValueError
    handler = logging.PythonHandler(stream)
    handler.flush()
    stream.flush.assert_called_once()

  def test_flush_with_environment_error(self):
    stream = mock.Mock()
    stream.flush.side_effect = EnvironmentError
    handler = logging.PythonHandler(stream)
    handler.flush()
    stream.flush.assert_called_once()

  def test_flush_with_assertion_error(self):
    stream = mock.Mock()
    stream.flush.side_effect = AssertionError
    handler = logging.PythonHandler(stream)
    with self.assertRaises(AssertionError):
      handler.flush()

  def test_log_to_std_err(self):
    record = std_logging.LogRecord(
        'name', std_logging.INFO, 'path', 12, 'logging_msg', [], False)
    with mock.patch.object(std_logging.StreamHandler, 'emit'):
      self.python_handler._log_to_stderr(record)
      std_logging.StreamHandler.emit.assert_called_once_with(record)

  @flagsaver.flagsaver(logtostderr=True)
  def test_emit_log_to_stderr(self):
    record = std_logging.LogRecord(
        'name', std_logging.INFO, 'path', 12, 'logging_msg', [], False)
    with mock.patch.object(self.python_handler, '_log_to_stderr'):
      self.python_handler.emit(record)
      self.python_handler._log_to_stderr.assert_called_once_with(record)

  def test_emit(self):
    stream = _StreamIO()
    handler = logging.PythonHandler(stream)
    handler.stderr_threshold = std_logging.FATAL
    record = std_logging.LogRecord(
        'name', std_logging.INFO, 'path', 12, 'logging_msg', [], False)
    handler.emit(record)
    self.assertEqual(1, stream.getvalue().count('logging_msg'))

  @flagsaver.flagsaver(stderrthreshold='debug')
  def test_emit_and_stderr_threshold(self):
    mock_stderr = _StreamIO()
    stream = _StreamIO()
    handler = logging.PythonHandler(stream)
    record = std_logging.LogRecord(
        'name', std_logging.INFO, 'path', 12, 'logging_msg', [], False)
    with mock.patch.object(sys, 'stderr', new=mock_stderr) as mock_stderr:
      handler.emit(record)
      self.assertEqual(1, stream.getvalue().count('logging_msg'))
      self.assertEqual(1, mock_stderr.getvalue().count('logging_msg'))

  @flagsaver.flagsaver(alsologtostderr=True)
  def test_emit_also_log_to_stderr(self):
    mock_stderr = _StreamIO()
    stream = _StreamIO()
    handler = logging.PythonHandler(stream)
    handler.stderr_threshold = std_logging.FATAL
    record = std_logging.LogRecord(
        'name', std_logging.INFO, 'path', 12, 'logging_msg', [], False)
    with mock.patch.object(sys, 'stderr', new=mock_stderr) as mock_stderr:
      handler.emit(record)
      self.assertEqual(1, stream.getvalue().count('logging_msg'))
      self.assertEqual(1, mock_stderr.getvalue().count('logging_msg'))

  def test_emit_on_stderr(self):
    mock_stderr = _StreamIO()
    with mock.patch.object(sys, 'stderr', new=mock_stderr) as mock_stderr:
      handler = logging.PythonHandler()
      handler.stderr_threshold = std_logging.INFO
      record = std_logging.LogRecord(
          'name', std_logging.INFO, 'path', 12, 'logging_msg', [], False)
      handler.emit(record)
      self.assertEqual(1, mock_stderr.getvalue().count('logging_msg'))

  def test_emit_fatal_absl(self):
    stream = _StreamIO()
    handler = logging.PythonHandler(stream)
    record = std_logging.LogRecord(
        'name', std_logging.FATAL, 'path', 12, 'logging_msg', [], False)
    record.__dict__[logging._ABSL_LOG_FATAL] = True
    with mock.patch.object(handler, 'flush') as mock_flush:
      with mock.patch.object(os, 'abort') as mock_abort:
        handler.emit(record)
        mock_abort.assert_called_once()
        mock_flush.assert_called()  # flush is also called by super class.

  def test_emit_fatal_non_absl(self):
    stream = _StreamIO()
    handler = logging.PythonHandler(stream)
    record = std_logging.LogRecord(
        'name', std_logging.FATAL, 'path', 12, 'logging_msg', [], False)
    with mock.patch.object(os, 'abort') as mock_abort:
      handler.emit(record)
      mock_abort.assert_not_called()

  def test_close(self):
    stream = mock.Mock()
    stream.isatty.return_value = True
    handler = logging.PythonHandler(stream)
    with mock.patch.object(handler, 'flush') as mock_flush:
      with mock.patch.object(std_logging.StreamHandler, 'close') as super_close:
        handler.close()
        mock_flush.assert_called_once()
        super_close.assert_called_once()
        stream.close.assert_not_called()

  def test_close_afile(self):
    stream = mock.Mock()
    stream.isatty.return_value = False
    stream.close.side_effect = ValueError
    handler = logging.PythonHandler(stream)
    with mock.patch.object(handler, 'flush') as mock_flush:
      with mock.patch.object(std_logging.StreamHandler, 'close') as super_close:
        handler.close()
        mock_flush.assert_called_once()
        super_close.assert_called_once()

  def test_close_stderr(self):
    with mock.patch.object(sys, 'stderr') as mock_stderr:
      mock_stderr.isatty.return_value = False
      handler = logging.PythonHandler(sys.stderr)
      handler.close()
      mock_stderr.close.assert_not_called()

  def test_close_stdout(self):
    with mock.patch.object(sys, 'stdout') as mock_stdout:
      mock_stdout.isatty.return_value = False
      handler = logging.PythonHandler(sys.stdout)
      handler.close()
      mock_stdout.close.assert_not_called()

  def test_close_fake_file(self):

    class FakeFile(object):
      """A file-like object that does not implement "isatty"."""

      def __init__(self):
        self.closed = False

      def close(self):
        self.closed = True

      def flush(self):
        pass

    fake_file = FakeFile()
    handler = logging.PythonHandler(fake_file)
    handler.close()
    self.assertTrue(fake_file.closed)


class PrefixFormatterTest(absltest.TestCase):
  """Tests the PrefixFormatter class."""

  def setUp(self):
    self.now_tuple = time.localtime(time.mktime(
        (1979, 10, 21, 18, 17, 16, 3, 15, 1)))
    self.new_prefix = lambda level: '(blah_prefix)'
    mock.patch.object(time, 'time').start()
    mock.patch.object(time, 'localtime').start()
    self.record = std_logging.LogRecord(
        'name', std_logging.INFO, 'path', 12, 'A Message', [], False)
    self.formatter = logging.PythonFormatter()

  def tearDown(self):
    mock.patch.stopall()

  @mock.patch.object(logging._thread_lib, 'get_ident')
  def test_get_thread_id(self, mock_get_ident):
    mock_get_ident.return_value = 12345
    self.assertEqual(12345, logging._get_thread_id())


class ABSLHandlerTest(absltest.TestCase):

  def setUp(self):
    formatter = logging.PythonFormatter()
    self.absl_handler = logging.ABSLHandler(formatter)

  def test_activate_python_handler(self):
    self.absl_handler.activate_python_handler()
    self.assertEqual(
        self.absl_handler._current_handler, self.absl_handler.python_handler)


class ABSLLoggerTest(absltest.TestCase):
  """Tests the ABSLLogger class."""

  def set_up_mock_frames(self):
    """Sets up mock frames for use with the testFindCaller methods."""
    logging_file = os.path.join('absl', 'logging', '__init__.py')

    # Set up mock frame 0
    mock_frame_0 = mock.Mock()
    mock_code_0 = mock.Mock()
    mock_code_0.co_filename = logging_file
    mock_code_0.co_name = 'LoggingLog'
    mock_code_0.co_firstlineno = 124
    mock_frame_0.f_code = mock_code_0
    mock_frame_0.f_lineno = 125

    # Set up mock frame 1
    mock_frame_1 = mock.Mock()
    mock_code_1 = mock.Mock()
    mock_code_1.co_filename = 'myfile.py'
    mock_code_1.co_name = 'Method1'
    mock_code_1.co_firstlineno = 124
    mock_frame_1.f_code = mock_code_1
    mock_frame_1.f_lineno = 125

    # Set up mock frame 2
    mock_frame_2 = mock.Mock()
    mock_code_2 = mock.Mock()
    mock_code_2.co_filename = 'myfile.py'
    mock_code_2.co_name = 'Method2'
    mock_code_2.co_firstlineno = 124
    mock_frame_2.f_code = mock_code_2
    mock_frame_2.f_lineno = 125

    # Set up mock frame 3
    mock_frame_3 = mock.Mock()
    mock_code_3 = mock.Mock()
    mock_code_3.co_filename = 'myfile.py'
    mock_code_3.co_name = 'Method3'
    mock_code_3.co_firstlineno = 124
    mock_frame_3.f_code = mock_code_3
    mock_frame_3.f_lineno = 125

    # Set up mock frame 4 that has the same function name as frame 2.
    mock_frame_4 = mock.Mock()
    mock_code_4 = mock.Mock()
    mock_code_4.co_filename = 'myfile.py'
    mock_code_4.co_name = 'Method2'
    mock_code_4.co_firstlineno = 248
    mock_frame_4.f_code = mock_code_4
    mock_frame_4.f_lineno = 249

    # Tie them together.
    mock_frame_4.f_back = None
    mock_frame_3.f_back = mock_frame_4
    mock_frame_2.f_back = mock_frame_3
    mock_frame_1.f_back = mock_frame_2
    mock_frame_0.f_back = mock_frame_1

    mock.patch.object(sys, '_getframe').start()
    sys._getframe.return_value = mock_frame_0

  def setUp(self):
    self.message = 'Hello Nurse'
    self.logger = logging.ABSLLogger('')

  def tearDown(self):
    mock.patch.stopall()
    self.logger._frames_to_skip.clear()

  def test_constructor_without_level(self):
    self.logger = logging.ABSLLogger('')
    self.assertEqual(std_logging.NOTSET, self.logger.getEffectiveLevel())

  def test_constructor_with_level(self):
    self.logger = logging.ABSLLogger('', std_logging.DEBUG)
    self.assertEqual(std_logging.DEBUG, self.logger.getEffectiveLevel())

  def test_find_caller_normal(self):
    self.set_up_mock_frames()
    expected_name = 'Method1'
    self.assertEqual(expected_name, self.logger.findCaller()[2])

  def test_find_caller_skip_method1(self):
    self.set_up_mock_frames()
    self.logger.register_frame_to_skip('myfile.py', 'Method1')
    expected_name = 'Method2'
    self.assertEqual(expected_name, self.logger.findCaller()[2])

  def test_find_caller_skip_method1_and_method2(self):
    self.set_up_mock_frames()
    self.logger.register_frame_to_skip('myfile.py', 'Method1')
    self.logger.register_frame_to_skip('myfile.py', 'Method2')
    expected_name = 'Method3'
    self.assertEqual(expected_name, self.logger.findCaller()[2])

  def test_find_caller_skip_method1_and_method3(self):
    self.set_up_mock_frames()
    self.logger.register_frame_to_skip('myfile.py', 'Method1')
    # Skipping Method3 should change nothing since Method2 should be hit.
    self.logger.register_frame_to_skip('myfile.py', 'Method3')
    expected_name = 'Method2'
    self.assertEqual(expected_name, self.logger.findCaller()[2])

  def test_find_caller_skip_method1_and_method4(self):
    self.set_up_mock_frames()
    self.logger.register_frame_to_skip('myfile.py', 'Method1')
    # Skipping frame 4's Method2 should change nothing for frame 2's Method2.
    self.logger.register_frame_to_skip('myfile.py', 'Method2', 248)
    expected_name = 'Method2'
    expected_frame_lineno = 125
    self.assertEqual(expected_name, self.logger.findCaller()[2])
    self.assertEqual(expected_frame_lineno, self.logger.findCaller()[1])

  def test_find_caller_skip_method1_method2_and_method3(self):
    self.set_up_mock_frames()
    self.logger.register_frame_to_skip('myfile.py', 'Method1')
    self.logger.register_frame_to_skip('myfile.py', 'Method2', 124)
    self.logger.register_frame_to_skip('myfile.py', 'Method3')
    expected_name = 'Method2'
    expected_frame_lineno = 249
    self.assertEqual(expected_name, self.logger.findCaller()[2])
    self.assertEqual(expected_frame_lineno, self.logger.findCaller()[1])

  def test_find_caller_stack_info(self):
    self.set_up_mock_frames()
    self.logger.register_frame_to_skip('myfile.py', 'Method1')
    with mock.patch.object(traceback, 'print_stack') as print_stack:
      self.assertEqual(
          ('myfile.py', 125, 'Method2', 'Stack (most recent call last):'),
          self.logger.findCaller(stack_info=True))
    print_stack.assert_called_once()

  @unittest.skipIf(six.PY3, 'Testing Python 2 specific behavior.')
  def test_find_caller_python2(self):
    """Ensure that we only return three items for base class compatibility."""
    self.set_up_mock_frames()
    self.logger.register_frame_to_skip('myfile.py', 'Method1')
    self.assertEqual(('myfile.py', 125, 'Method2'), self.logger.findCaller())

  def test_critical(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.critical(self.message)
      self.logger.log.assert_called_once_with(
          std_logging.CRITICAL, self.message)

  def test_fatal(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.fatal(self.message)
      self.logger.log.assert_called_once_with(std_logging.FATAL, self.message)

  def test_error(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.error(self.message)
      self.logger.log.assert_called_once_with(std_logging.ERROR, self.message)

  def test_warn(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.warn(self.message)
      self.logger.log.assert_called_once_with(std_logging.WARN, self.message)

  def test_warning(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.warning(self.message)
      self.logger.log.assert_called_once_with(std_logging.WARNING, self.message)

  def test_info(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.info(self.message)
      self.logger.log.assert_called_once_with(std_logging.INFO, self.message)

  def test_debug(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.debug(self.message)
      self.logger.log.assert_called_once_with(std_logging.DEBUG, self.message)

  def test_log_debug_with_python(self):
    with mock.patch.object(self.logger, 'log'):
      FLAGS.verbosity = 1
      self.logger.debug(self.message)
      self.logger.log.assert_called_once_with(std_logging.DEBUG, self.message)

  def test_log_fatal_with_python(self):
    with mock.patch.object(self.logger, 'log'):
      self.logger.fatal(self.message)
      self.logger.log.assert_called_once_with(std_logging.FATAL, self.message)

  def test_register_frame_to_skip(self):
    # This is basically just making sure that if I put something in a
    # list, it actually appears in that list.
    frame_tuple = ('file', 'method')
    self.logger.register_frame_to_skip(*frame_tuple)
    self.assertIn(frame_tuple, self.logger._frames_to_skip)

  def test_register_frame_to_skip_with_lineno(self):
    frame_tuple = ('file', 'method', 123)
    self.logger.register_frame_to_skip(*frame_tuple)
    self.assertIn(frame_tuple, self.logger._frames_to_skip)

  def test_logger_cannot_be_disabled(self):
    self.logger.disabled = True
    record = self.logger.makeRecord(
        'name', std_logging.INFO, 'fn', 20, 'msg', [], False)
    with mock.patch.object(self.logger, 'callHandlers') as mock_call_handlers:
      self.logger.handle(record)
    mock_call_handlers.assert_called_once()


class ABSLLogPrefixTest(parameterized.TestCase):

  def setUp(self):
    self.record = std_logging.LogRecord(
        'name', std_logging.INFO, 'path/to/source.py', 13, 'log message',
        None, None)

  @parameterized.named_parameters(
      ('debug', std_logging.DEBUG, 'I'),
      ('info', std_logging.INFO, 'I'),
      ('warning', std_logging.WARNING, 'W'),
      ('error', std_logging.ERROR, 'E'),
  )
  def test_default_prefixes(self, levelno, level_prefix):
    self.record.levelno = levelno
    self.record.created = 1494293880.378885
    thread_id = '{: >5}'.format(logging._get_thread_id())
    # Use UTC so the test passes regardless of the local time zone.
    with mock.patch.object(time, 'localtime', side_effect=time.gmtime):
      self.assertEqual(
          '{}0509 01:38:00.378885 {} source.py:13] '.format(
              level_prefix, thread_id),
          logging.get_absl_log_prefix(self.record))
      time.localtime.assert_called_once_with(self.record.created)

  def test_absl_prefix_regex(self):
    self.record.created = 1226888258.0521369
    # Use UTC so the test passes regardless of the local time zone.
    with mock.patch.object(time, 'localtime', side_effect=time.gmtime):
      prefix = logging.get_absl_log_prefix(self.record)

    match = re.search(logging.ABSL_LOGGING_PREFIX_REGEX, prefix)
    self.assertTrue(match)

    expect = {'severity': 'I',
              'month': '11',
              'day': '17',
              'hour': '02',
              'minute': '17',
              'second': '38',
              'microsecond': '052136',
              'thread_id': str(logging._get_thread_id()),
              'filename': 'source.py',
              'line': '13',
             }
    actual = {name: match.group(name) for name in expect}
    self.assertEqual(expect, actual)

  def test_critical_absl(self):
    self.record.levelno = std_logging.CRITICAL
    self.record.created = 1494293880.378885
    self.record._absl_log_fatal = True
    thread_id = '{: >5}'.format(logging._get_thread_id())
    # Use UTC so the test passes regardless of the local time zone.
    with mock.patch.object(time, 'localtime', side_effect=time.gmtime):
      self.assertEqual(
          'F0509 01:38:00.378885 {} source.py:13] '.format(thread_id),
          logging.get_absl_log_prefix(self.record))
      time.localtime.assert_called_once_with(self.record.created)

  def test_critical_non_absl(self):
    self.record.levelno = std_logging.CRITICAL
    self.record.created = 1494293880.378885
    thread_id = '{: >5}'.format(logging._get_thread_id())
    # Use UTC so the test passes regardless of the local time zone.
    with mock.patch.object(time, 'localtime', side_effect=time.gmtime):
      self.assertEqual(
          'E0509 01:38:00.378885 {} source.py:13] CRITICAL - '.format(
              thread_id),
          logging.get_absl_log_prefix(self.record))
      time.localtime.assert_called_once_with(self.record.created)


class LogCountTest(absltest.TestCase):

  def test_counter_threadsafe(self):
    threads_start = threading.Event()
    counts = set()
    k = object()

    def t():
      threads_start.wait()
      counts.add(logging._get_next_log_count_per_token(k))

    threads = [threading.Thread(target=t) for _ in range(100)]
    for thread in threads:
      thread.start()
    threads_start.set()
    for thread in threads:
      thread.join()
    self.assertEqual(counts, {i for i in range(100)})


class LoggingTest(absltest.TestCase):

  def test_fatal(self):
    with mock.patch.object(os, 'abort') as mock_abort:
      logging.fatal('Die!')
      mock_abort.assert_called_once()

  def test_find_log_dir_with_arg(self):
    with mock.patch.object(os, 'access'), \
        mock.patch.object(os.path, 'isdir'):
      os.path.isdir.return_value = True
      os.access.return_value = True
      log_dir = logging.find_log_dir(log_dir='./')
      self.assertEqual('./', log_dir)

  @flagsaver.flagsaver(log_dir='./')
  def test_find_log_dir_with_flag(self):
    with mock.patch.object(os, 'access'), \
        mock.patch.object(os.path, 'isdir'):
      os.path.isdir.return_value = True
      os.access.return_value = True
      log_dir = logging.find_log_dir()
      self.assertEqual('./', log_dir)

  @flagsaver.flagsaver(log_dir='')
  def test_find_log_dir_with_hda_tmp(self):
    with mock.patch.object(os, 'access'), \
        mock.patch.object(os.path, 'exists'), \
        mock.patch.object(os.path, 'isdir'):
      os.path.exists.return_value = True
      os.path.isdir.return_value = True
      os.access.return_value = True
      log_dir = logging.find_log_dir()
      self.assertEqual('/tmp/', log_dir)

  @flagsaver.flagsaver(log_dir='')
  def test_find_log_dir_with_tmp(self):
    with mock.patch.object(os, 'access'), \
        mock.patch.object(os.path, 'exists'), \
        mock.patch.object(os.path, 'isdir'):
      os.path.exists.return_value = False
      os.path.isdir.side_effect = lambda path: path == '/tmp/'
      os.access.return_value = True
      log_dir = logging.find_log_dir()
      self.assertEqual('/tmp/', log_dir)

  def test_find_log_dir_with_nothing(self):
    with mock.patch.object(os.path, 'exists'), \
        mock.patch.object(os.path, 'isdir'), \
        mock.patch.object(logging.get_absl_logger(), 'fatal') as mock_fatal:
      os.path.exists.return_value = False
      os.path.isdir.return_value = False
      log_dir = logging.find_log_dir()
      mock_fatal.assert_called()
      self.assertEqual(None, log_dir)

  def test_find_log_dir_and_names_with_args(self):
    user = 'test_user'
    host = 'test_host'
    log_dir = 'here'
    program_name = 'prog1'
    with mock.patch.object(getpass, 'getuser'), \
        mock.patch.object(logging, 'find_log_dir') as mock_find_log_dir, \
        mock.patch.object(socket, 'gethostname') as mock_gethostname:
      getpass.getuser.return_value = user
      mock_gethostname.return_value = host
      mock_find_log_dir.return_value = log_dir

      prefix = '%s.%s.%s.log' % (program_name, host, user)
      self.assertEqual((log_dir, prefix, program_name),
                       logging.find_log_dir_and_names(
                           program_name=program_name, log_dir=log_dir))

  def test_find_log_dir_and_names_without_args(self):
    user = 'test_user'
    host = 'test_host'
    log_dir = 'here'
    py_program_name = 'py_prog1'
    sys.argv[0] = 'path/to/prog1'
    with mock.patch.object(getpass, 'getuser'), \
        mock.patch.object(logging, 'find_log_dir') as mock_find_log_dir, \
        mock.patch.object(socket, 'gethostname'):
      getpass.getuser.return_value = user
      socket.gethostname.return_value = host
      mock_find_log_dir.return_value = log_dir
      prefix = '%s.%s.%s.log' % (py_program_name, host, user)
      self.assertEqual((log_dir, prefix, py_program_name),
                       logging.find_log_dir_and_names())

  def test_find_log_dir_and_names_wo_username(self):
    # Windows doesn't have os.getuid at all
    if hasattr(os, 'getuid'):
      mock_getuid = mock.patch.object(os, 'getuid')
      uid = 100
      logged_uid = '100'
    else:
      # The function doesn't exist, but our test code still tries to mock
      # it, so just use a fake thing.
      mock_getuid = _mock_windows_os_getuid()
      uid = -1
      logged_uid = 'unknown'

    host = 'test_host'
    log_dir = 'here'
    program_name = 'prog1'
    with mock.patch.object(getpass, 'getuser'), \
        mock_getuid as getuid, \
        mock.patch.object(logging, 'find_log_dir') as mock_find_log_dir, \
        mock.patch.object(socket, 'gethostname') as mock_gethostname:
      getpass.getuser.side_effect = KeyError()
      getuid.return_value = uid
      mock_gethostname.return_value = host
      mock_find_log_dir.return_value = log_dir

      prefix = '%s.%s.%s.log' % (program_name, host, logged_uid)
      self.assertEqual((log_dir, prefix, program_name),
                       logging.find_log_dir_and_names(
                           program_name=program_name, log_dir=log_dir))

  def test_errors_in_logging(self):
    with mock.patch.object(sys, 'stderr', new=_StreamIO()) as stderr:
      logging.info('not enough args: %s %s', 'foo')  # pylint: disable=logging-too-few-args
      self.assertIn('Traceback (most recent call last):', stderr.getvalue())
      self.assertIn('TypeError', stderr.getvalue())

  def test_dict_arg(self):
    # Tests that passing a dictionary as a single argument does not crash.
    logging.info('%(test)s', {'test': 'Hello world!'})

  def test_exception_dict_format(self):
    # Just verify that this doesn't raise a TypeError.
    logging.exception('%(test)s', {'test': 'Hello world!'})

  def test_logging_levels(self):
    old_level = logging.get_verbosity()

    logging.set_verbosity(logging.DEBUG)
    self.assertEquals(logging.get_verbosity(), logging.DEBUG)
    self.assertTrue(logging.level_debug())
    self.assertTrue(logging.level_info())
    self.assertTrue(logging.level_warning())
    self.assertTrue(logging.level_error())

    logging.set_verbosity(logging.INFO)
    self.assertEquals(logging.get_verbosity(), logging.INFO)
    self.assertFalse(logging.level_debug())
    self.assertTrue(logging.level_info())
    self.assertTrue(logging.level_warning())
    self.assertTrue(logging.level_error())

    logging.set_verbosity(logging.WARNING)
    self.assertEquals(logging.get_verbosity(), logging.WARNING)
    self.assertFalse(logging.level_debug())
    self.assertFalse(logging.level_info())
    self.assertTrue(logging.level_warning())
    self.assertTrue(logging.level_error())

    logging.set_verbosity(logging.ERROR)
    self.assertEquals(logging.get_verbosity(), logging.ERROR)
    self.assertFalse(logging.level_debug())
    self.assertFalse(logging.level_info())
    self.assertTrue(logging.level_error())

    logging.set_verbosity(old_level)

  def test_set_verbosity_strings(self):
    old_level = logging.get_verbosity()

    # Lowercase names.
    logging.set_verbosity('debug')
    self.assertEquals(logging.get_verbosity(), logging.DEBUG)
    logging.set_verbosity('info')
    self.assertEquals(logging.get_verbosity(), logging.INFO)
    logging.set_verbosity('warning')
    self.assertEquals(logging.get_verbosity(), logging.WARNING)
    logging.set_verbosity('warn')
    self.assertEquals(logging.get_verbosity(), logging.WARNING)
    logging.set_verbosity('error')
    self.assertEquals(logging.get_verbosity(), logging.ERROR)
    logging.set_verbosity('fatal')

    # Uppercase names.
    self.assertEquals(logging.get_verbosity(), logging.FATAL)
    logging.set_verbosity('DEBUG')
    self.assertEquals(logging.get_verbosity(), logging.DEBUG)
    logging.set_verbosity('INFO')
    self.assertEquals(logging.get_verbosity(), logging.INFO)
    logging.set_verbosity('WARNING')
    self.assertEquals(logging.get_verbosity(), logging.WARNING)
    logging.set_verbosity('WARN')
    self.assertEquals(logging.get_verbosity(), logging.WARNING)
    logging.set_verbosity('ERROR')
    self.assertEquals(logging.get_verbosity(), logging.ERROR)
    logging.set_verbosity('FATAL')
    self.assertEquals(logging.get_verbosity(), logging.FATAL)

    # Integers as strings.
    logging.set_verbosity(str(logging.DEBUG))
    self.assertEquals(logging.get_verbosity(), logging.DEBUG)
    logging.set_verbosity(str(logging.INFO))
    self.assertEquals(logging.get_verbosity(), logging.INFO)
    logging.set_verbosity(str(logging.WARNING))
    self.assertEquals(logging.get_verbosity(), logging.WARNING)
    logging.set_verbosity(str(logging.ERROR))
    self.assertEquals(logging.get_verbosity(), logging.ERROR)
    logging.set_verbosity(str(logging.FATAL))
    self.assertEquals(logging.get_verbosity(), logging.FATAL)

    logging.set_verbosity(old_level)

  def test_key_flags(self):
    key_flags = FLAGS.get_key_flags_for_module(logging)
    key_flag_names = [flag.name for flag in key_flags]
    self.assertIn('stderrthreshold', key_flag_names)
    self.assertIn('verbosity', key_flag_names)

  def test_get_absl_logger(self):
    self.assertIsInstance(
        logging.get_absl_logger(), logging.ABSLLogger)

  def test_get_absl_handler(self):
    self.assertIsInstance(
        logging.get_absl_handler(), logging.ABSLHandler)


@mock.patch.object(logging.ABSLLogger, 'register_frame_to_skip')
class LogSkipPrefixTest(absltest.TestCase):
  """Tests for logging.skip_log_prefix."""

  def _log_some_info(self):
    """Logging helper function for LogSkipPrefixTest."""
    logging.info('info')

  def _log_nested_outer(self):
    """Nested logging helper functions for LogSkipPrefixTest."""
    def _log_nested_inner():
      logging.info('info nested')
    return _log_nested_inner

  def test_skip_log_prefix_with_name(self, mock_skip_register):
    retval = logging.skip_log_prefix('_log_some_info')
    mock_skip_register.assert_called_once_with(__file__, '_log_some_info', None)
    self.assertEqual(retval, '_log_some_info')

  def test_skip_log_prefix_with_func(self, mock_skip_register):
    retval = logging.skip_log_prefix(self._log_some_info)
    mock_skip_register.assert_called_once_with(
        __file__, '_log_some_info', mock.ANY)
    self.assertEqual(retval, self._log_some_info)

  def test_skip_log_prefix_with_functools_partial(self, mock_skip_register):
    partial_input = functools.partial(self._log_some_info)
    with self.assertRaises(ValueError):
      _ = logging.skip_log_prefix(partial_input)
    mock_skip_register.assert_not_called()

  def test_skip_log_prefix_with_lambda(self, mock_skip_register):
    lambda_input = lambda _: self._log_some_info()
    retval = logging.skip_log_prefix(lambda_input)
    mock_skip_register.assert_called_once_with(__file__, '<lambda>', mock.ANY)
    self.assertEqual(retval, lambda_input)

  def test_skip_log_prefix_with_bad_input(self, mock_skip_register):
    dict_input = {1: 2, 2: 3}
    with self.assertRaises(TypeError):
      _ = logging.skip_log_prefix(dict_input)
    mock_skip_register.assert_not_called()

  def test_skip_log_prefix_with_nested_func(self, mock_skip_register):
    nested_input = self._log_nested_outer()
    retval = logging.skip_log_prefix(nested_input)
    mock_skip_register.assert_called_once_with(
        __file__, '_log_nested_inner', mock.ANY)
    self.assertEqual(retval, nested_input)

  def test_skip_log_prefix_decorator(self, mock_skip_register):

    @logging.skip_log_prefix
    def _log_decorated():
      logging.info('decorated')

    del _log_decorated
    mock_skip_register.assert_called_once_with(
        __file__, '_log_decorated', mock.ANY)


@contextlib.contextmanager
def override_python_handler_stream(stream):
  handler = logging.get_absl_handler().python_handler
  old_stream = handler.stream
  handler.stream = stream
  try:
    yield
  finally:
    handler.stream = old_stream


class GetLogFileNameTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('err', sys.stderr),
      ('out', sys.stdout),
  )
  def test_get_log_file_name_py_std(self, stream):
    with override_python_handler_stream(stream):
      self.assertEqual('', logging.get_log_file_name())

  def test_get_log_file_name_py_no_name(self):

    class FakeFile(object):
      pass

    with override_python_handler_stream(FakeFile()):
      self.assertEqual('', logging.get_log_file_name())

  def test_get_log_file_name_py_file(self):
    _, filename = tempfile.mkstemp(dir=FLAGS.test_tmpdir)
    with open(filename, 'a') as stream:
      with override_python_handler_stream(stream):
        self.assertEqual(filename, logging.get_log_file_name())


@contextlib.contextmanager
def _mock_windows_os_getuid():
  yield mock.MagicMock()


if __name__ == '__main__':
  absltest.main()
