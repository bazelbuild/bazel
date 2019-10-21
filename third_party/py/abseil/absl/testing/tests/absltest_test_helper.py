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

"""Helper binary for absltest_test.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tempfile
import unittest

from absl import flags
from absl.testing import absltest

FLAGS = flags.FLAGS

flags.DEFINE_integer('test_id', 0, 'Which test to run.')


class HelperTest(absltest.TestCase):

  def test_flags(self):
    if FLAGS.test_id == 1:
      self.assertEqual(FLAGS.test_random_seed, 301)
      if os.name == 'nt':
        # On Windows, it's always in the temp dir, which doesn't start with '/'.
        expected_prefix = tempfile.gettempdir()
      else:
        expected_prefix = '/'
      self.assertTrue(
          FLAGS.test_tmpdir.startswith(expected_prefix),
          '--test_tmpdir={} does not start with {}'.format(
              FLAGS.test_tmpdir, expected_prefix))
      self.assertTrue(os.access(FLAGS.test_tmpdir, os.W_OK))
    elif FLAGS.test_id == 2:
      self.assertEqual(FLAGS.test_random_seed, 321)
      self.assertEqual(
          FLAGS.test_srcdir,
          os.environ['ABSLTEST_TEST_HELPER_EXPECTED_TEST_SRCDIR'])
      self.assertEqual(
          FLAGS.test_tmpdir,
          os.environ['ABSLTEST_TEST_HELPER_EXPECTED_TEST_TMPDIR'])
    elif FLAGS.test_id == 3:
      self.assertEqual(FLAGS.test_random_seed, 123)
      self.assertEqual(
          FLAGS.test_srcdir,
          os.environ['ABSLTEST_TEST_HELPER_EXPECTED_TEST_SRCDIR'])
      self.assertEqual(
          FLAGS.test_tmpdir,
          os.environ['ABSLTEST_TEST_HELPER_EXPECTED_TEST_TMPDIR'])
    elif FLAGS.test_id == 4:
      self.assertEqual(FLAGS.test_random_seed, 221)
      self.assertEqual(
          FLAGS.test_srcdir,
          os.environ['ABSLTEST_TEST_HELPER_EXPECTED_TEST_SRCDIR'])
      self.assertEqual(
          FLAGS.test_tmpdir,
          os.environ['ABSLTEST_TEST_HELPER_EXPECTED_TEST_TMPDIR'])
    else:
      raise unittest.SkipTest(
          'Not asked to run: --test_id={}'.format(FLAGS.test_id))

  @unittest.expectedFailure
  def test_expected_failure(self):
    if FLAGS.test_id == 5:
      self.assertEqual(1, 1)  # Expected failure, got success.
    else:
      self.assertEqual(1, 2)  # The expected failure.

  def test_xml_env_vars(self):
    if FLAGS.test_id == 6:
      self.assertEqual(
          FLAGS.xml_output_file,
          os.environ['ABSLTEST_TEST_HELPER_EXPECTED_XML_OUTPUT_FILE'])
    else:
      raise unittest.SkipTest(
          'Not asked to run: --test_id={}'.format(FLAGS.test_id))


class TempFileHelperTest(absltest.TestCase):
  tempfile_cleanup = absltest.TempFileCleanup[os.environ.get(
      'ABSLTEST_TEST_HELPER_TEMPFILE_CLEANUP', 'SUCCESS')]

  def test_failure(self):
    self.create_tempfile('failure')
    self.fail('expected failure')

  def test_success(self):
    self.create_tempfile('success')


if __name__ == '__main__':
  absltest.main()
