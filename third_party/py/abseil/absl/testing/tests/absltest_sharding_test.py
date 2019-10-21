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

"""Tests for test sharding protocol."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess

from absl import flags
from absl.testing import _bazelize_command
from absl.testing import absltest

FLAGS = flags.FLAGS
NUM_TEST_METHODS = 8  # Hard-coded, based on absltest_sharding_test_helper.py


class TestShardingTest(absltest.TestCase):
  """Integration tests: Runs a test binary with sharding.

  This is done by setting the sharding environment variables.
  """

  def setUp(self):
    self._test_name = 'absl/testing/tests/absltest_sharding_test_helper'
    self._shard_file = None

  def tearDown(self):
    if self._shard_file is not None and os.path.exists(self._shard_file):
      os.unlink(self._shard_file)

  def _run_sharded(self, total_shards, shard_index, shard_file=None):
    """Runs the py_test binary in a subprocess.

    Args:
      total_shards: int, the total number of shards.
      shard_index: int, the shard index.
      shard_file: string, if not 'None', the path to the shard file.
        This method asserts it is properly created.

    Returns:
      (stdout, exit_code) tuple of (string, int).
    """
    env = {'TEST_TOTAL_SHARDS': str(total_shards),
           'TEST_SHARD_INDEX': str(shard_index)}
    if 'SYSTEMROOT' in os.environ:
      # This is used by the random module on Windows to locate crypto
      # libraries.
      env['SYSTEMROOT'] = os.environ['SYSTEMROOT']
    if shard_file:
      self._shard_file = shard_file
      env['TEST_SHARD_STATUS_FILE'] = shard_file
      if os.path.exists(shard_file):
        os.unlink(shard_file)

    proc = subprocess.Popen(
        args=[_bazelize_command.get_executable_path(self._test_name)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)
    stdout = proc.communicate()[0]

    if shard_file:
      self.assertTrue(os.path.exists(shard_file))

    return (stdout, proc.wait())

  def _assert_sharding_correctness(self, total_shards):
    """Assert the primary correctness and performance of sharding.

    1. Completeness (all methods are run)
    2. Partition (each method run at most once)
    3. Balance (for performance)

    Args:
      total_shards: int, total number of shards.
    """

    outerr_by_shard = []  # A list of lists of strings
    combined_outerr = []  # A list of strings
    exit_code_by_shard = []  # A list of ints

    for i in range(total_shards):
      (out, exit_code) = self._run_sharded(total_shards, i)
      method_list = [x for x in out.split('\n') if x.startswith('class')]
      outerr_by_shard.append(method_list)
      combined_outerr.extend(method_list)
      exit_code_by_shard.append(exit_code)

    self.assertEquals(1, len([x for x in exit_code_by_shard if x != 0]),
                      'Expected exactly one failure')

    # Test completeness and partition properties.
    self.assertEquals(NUM_TEST_METHODS, len(combined_outerr),
                      'Partition requirement not met')
    self.assertEquals(NUM_TEST_METHODS, len(set(combined_outerr)),
                      'Completeness requirement not met')

    # Test balance:
    for i in range(len(outerr_by_shard)):
      self.assertGreaterEqual(len(outerr_by_shard[i]),
                              (NUM_TEST_METHODS / total_shards) - 1,
                              'Shard %d of %d out of balance' %
                              (i, len(outerr_by_shard)))

  def test_shard_file(self):
    self._run_sharded(3, 1, os.path.join(FLAGS.test_tmpdir, 'shard_file'))

  def test_zero_shards(self):
    out, exit_code = self._run_sharded(0, 0)
    self.assertEquals(1, exit_code)
    self.assertGreaterEqual(out.find('Bad sharding values. index=0, total=0'),
                            0, 'Bad output: %s' % (out))

  def test_with_four_shards(self):
    self._assert_sharding_correctness(4)

  def test_with_one_shard(self):
    self._assert_sharding_correctness(1)

  def test_with_ten_shards(self):
    self._assert_sharding_correctness(10)


if __name__ == '__main__':
  absltest.main()
