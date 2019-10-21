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

"""Helper script for logging_functional_test."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging as std_logging
import os
import sys
import threading
import time
import timeit

from absl import app
from absl import flags
from absl import logging
import mock
from six.moves import xrange  # pylint: disable=redefined-builtin

FLAGS = flags.FLAGS


class VerboseDel(object):
  """Dummy class to test __del__ running."""

  def __init__(self, msg):
    self._msg = msg

  def __del__(self):
    sys.stderr.write(self._msg)
    sys.stderr.flush()


def _test_do_logging():
  """Do some log operations."""
  logging.vlog(3, 'This line is VLOG level 3')
  logging.vlog(2, 'This line is VLOG level 2')
  logging.log(2, 'This line is log level 2')
  if logging.vlog_is_on(2):
    logging.log(1, 'VLOG level 1, but only if VLOG level 2 is active')

  logging.vlog(1, 'This line is VLOG level 1')
  logging.log(1, 'This line is log level 1')
  logging.debug('This line is DEBUG')

  logging.vlog(0, 'This line is VLOG level 0')
  logging.log(0, 'This line is log level 0')
  logging.info('Interesting Stuff\0')
  logging.info('Interesting Stuff with Arguments: %d', 42)
  logging.info('%(a)s Stuff with %(b)s',
               {'a': 'Interesting', 'b': 'Dictionary'})

  with mock.patch.object(timeit, 'default_timer') as mock_timer:
    mock_timer.return_value = 0
    while timeit.default_timer() < 9:
      logging.log_every_n_seconds(logging.INFO, 'This should appear 5 times.',
                                  2)
      mock_timer.return_value = mock_timer() + .2

  for i in xrange(1, 5):
    logging.log_first_n(logging.INFO, 'Info first %d of %d', 2, i, 2)
    logging.log_every_n(logging.INFO, 'Info %d (every %d)', 3, i, 3)

  logging.vlog(-1, 'This line is VLOG level -1')
  logging.log(-1, 'This line is log level -1')
  logging.warning('Worrying Stuff')
  for i in xrange(1, 5):
    logging.log_first_n(logging.WARNING, 'Warn first %d of %d', 2, i, 2)
    logging.log_every_n(logging.WARNING, 'Warn %d (every %d)', 3, i, 3)

  logging.vlog(-2, 'This line is VLOG level -2')
  logging.log(-2, 'This line is log level -2')
  try:
    raise OSError('Fake Error')
  except OSError:
    saved_exc_info = sys.exc_info()
    logging.exception('An Exception %s')
    logging.exception('Once more, %(reason)s', {'reason': 'just because'})
    logging.error('Exception 2 %s', exc_info=True)
    logging.error('Non-exception', exc_info=False)

  try:
    sys.exc_clear()
  except AttributeError:
    # No sys.exc_clear() in Python 3, but this will clear sys.exc_info() too.
    pass

  logging.error('Exception %s', '3', exc_info=saved_exc_info)
  logging.error('No traceback', exc_info=saved_exc_info[:2] + (None,))

  logging.error('Alarming Stuff')
  for i in xrange(1, 5):
    logging.log_first_n(logging.ERROR, 'Error first %d of %d', 2, i, 2)
    logging.log_every_n(logging.ERROR, 'Error %d (every %d)', 3, i, 3)
  logging.flush()


def _test_fatal_main_thread_only():
  """Test logging.fatal from main thread, no other threads running."""
  v = VerboseDel('fatal_main_thread_only main del called\n')
  try:
    logging.fatal('fatal_main_thread_only message')
  finally:
    del v


def _test_fatal_with_other_threads():
  """Test logging.fatal from main thread, other threads running."""

  lock = threading.Lock()
  lock.acquire()

  def sleep_forever(lock=lock):
    v = VerboseDel('fatal_with_other_threads non-main del called\n')
    try:
      lock.release()
      while True:
        time.sleep(10000)
    finally:
      del v

  v = VerboseDel('fatal_with_other_threads main del called\n')
  try:
    # Start new thread
    t = threading.Thread(target=sleep_forever)
    t.start()

    # Wait for other thread
    lock.acquire()
    lock.release()

    # Die
    logging.fatal('fatal_with_other_threads message')
    while True:
      time.sleep(10000)
  finally:
    del v


def _test_fatal_non_main_thread():
  """Test logging.fatal from non main thread."""

  lock = threading.Lock()
  lock.acquire()

  def die_soon(lock=lock):
    v = VerboseDel('fatal_non_main_thread non-main del called\n')
    try:
      # Wait for signal from other thread
      lock.acquire()
      lock.release()
      logging.fatal('fatal_non_main_thread message')
      while True:
        time.sleep(10000)
    finally:
      del v

  v = VerboseDel('fatal_non_main_thread main del called\n')
  try:
    # Start new thread
    t = threading.Thread(target=die_soon)
    t.start()

    # Signal other thread
    lock.release()

    # Wait for it to die
    while True:
      time.sleep(10000)
  finally:
    del v


def _test_critical_from_non_absl_logger():
  """Test CRITICAL logs from non-absl loggers."""

  std_logging.critical('A critical message')


def _test_register_frame_to_skip():
  """Test skipping frames for line number reporting."""

  def _getline():

    def _getline_inner():
      return logging.get_absl_logger().findCaller()[1]

    return _getline_inner()

  # Check register_frame_to_skip function to see if log frame skipping works.
  line1 = _getline()
  line2 = _getline()
  logging.get_absl_logger().register_frame_to_skip(__file__, '_getline')
  line3 = _getline()
  # Both should be line number of the _getline_inner() call.
  assert (line1 == line2), (line1, line2)
  # line3 should be a line number in this function.
  assert (line2 != line3), (line2, line3)


def _test_flush():
  """Test flush in various difficult cases."""
  # Flush, but one of the logfiles is closed
  log_filename = os.path.join(FLAGS.log_dir, 'a_thread_with_logfile.txt')
  with open(log_filename, 'w') as log_file:
    logging.get_absl_handler().python_handler.stream = log_file
  logging.flush()


def _test_stderrthreshold():
  """Tests modifying --stderrthreshold after flag parsing will work."""

  def log_things():
    logging.debug('FLAGS.stderrthreshold=%s, debug log', FLAGS.stderrthreshold)
    logging.info('FLAGS.stderrthreshold=%s, info log', FLAGS.stderrthreshold)
    logging.warning('FLAGS.stderrthreshold=%s, warning log',
                    FLAGS.stderrthreshold)
    logging.error('FLAGS.stderrthreshold=%s, error log', FLAGS.stderrthreshold)

  FLAGS.stderrthreshold = 'debug'
  log_things()
  FLAGS.stderrthreshold = 'info'
  log_things()
  FLAGS.stderrthreshold = 'warning'
  log_things()
  FLAGS.stderrthreshold = 'error'
  log_things()


def _test_std_logging():
  """Tests logs from std logging."""
  std_logging.debug('std debug log')
  std_logging.info('std info log')
  std_logging.warning('std warning log')
  std_logging.error('std error log')


def _test_bad_exc_info():
  """Tests when a bad exc_info valud is provided."""
  logging.info('Bad exc_info', exc_info=(None, None))


def _test_none_exc_info():
  """Tests when exc_info is requested but not available."""
  # Clear exc_info first.
  try:
    sys.exc_clear()
  except AttributeError:
    # No sys.exc_clear() in Python 3, but this will clear sys.exc_info() too.
    pass
  logging.info('None exc_info', exc_info=True)


def _test_unicode():
  """Tests unicode handling."""

  test_names = []

  def log(name, msg, *args):
    """Logs the message, and ensures the same name is not logged again."""
    assert name not in test_names, ('test_unicode expects unique names to work,'
                                    ' found existing name {}').format(name)
    test_names.append(name)

    # Add line seprators so that tests can verify the output for each log
    # message.
    sys.stderr.write('-- begin {} --\n'.format(name))
    logging.info(msg, *args)
    sys.stderr.write('-- end {} --\n'.format(name))

  log('unicode', u'G\u00eete: Ch\u00e2tonnaye')
  log('unicode % unicode', u'G\u00eete: %s', u'Ch\u00e2tonnaye')
  log('bytes % bytes', u'G\u00eete: %s'.encode('utf-8'),
      u'Ch\u00e2tonnaye'.encode('utf-8'))
  log('unicode % bytes', u'G\u00eete: %s', u'Ch\u00e2tonnaye'.encode('utf-8'))
  log('bytes % unicode', u'G\u00eete: %s'.encode('utf-8'), u'Ch\u00e2tonnaye')
  log('unicode % iso8859-15', u'G\u00eete: %s',
      u'Ch\u00e2tonnaye'.encode('iso-8859-15'))
  log('str % exception', 'exception: %s', Exception(u'Ch\u00e2tonnaye'))


def main(argv):
  del argv  # Unused.

  test_name = os.environ.get('TEST_NAME', None)
  test_fn = globals().get('_test_%s' % test_name)
  if test_fn is None:
    raise AssertionError('TEST_NAME must be set to a valid value')
  # Flush so previous messages are written to file before we switch to a new
  # file with use_absl_log_file.
  logging.flush()
  if os.environ.get('USE_ABSL_LOG_FILE') == '1':
    logging.get_absl_handler().use_absl_log_file('absl_log_file', FLAGS.log_dir)

  test_fn()


if __name__ == '__main__':
  sys.argv[0] = 'py_argv_0'
  app.run(main)
