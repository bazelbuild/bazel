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

"""Tests for test randomization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import random
import subprocess

from absl import flags
from absl.testing import _bazelize_command
from absl.testing import absltest
from absl.testing import parameterized

FLAGS = flags.FLAGS


class TestOrderRandomizationTest(parameterized.TestCase):
  """Integration tests: Runs a py_test binary with randomization.

  This is done by setting flags and environment variables.
  """

  def setUp(self):
    self._test_name = 'absl/testing/tests/absltest_randomization_testcase'

  def _run_test(self, extra_argv, extra_env):
    """Runs the py_test binary in a subprocess, with the given args or env.

    Args:
      extra_argv: extra args to pass to the test
      extra_env: extra env vars to set when running the test

    Returns:
      (stdout, test_cases, exit_code) tuple of (str, list of strs, int).
    """
    env = dict(os.environ)
    # If *this* test is being run with this flag, we don't want to
    # automatically set it for all tests we run.
    env.pop('TEST_RANDOMIZE_ORDERING_SEED', '')
    if extra_env is not None:
      env.update(extra_env)

    command = (
        [_bazelize_command.get_executable_path(self._test_name)] + extra_argv)
    proc = subprocess.Popen(
        args=command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True)

    stdout, _ = proc.communicate()

    test_lines = [l for l in stdout.splitlines() if l.startswith('class ')]
    return stdout, test_lines, proc.wait()

  def test_no_args(self):
    output, tests, exit_code = self._run_test([], None)
    self.assertEqual(0, exit_code, msg='command output: ' + output)
    self.assertNotIn('Randomizing test order with seed:', output)
    cases = ['class A test ' + t for t in ('A', 'B', 'C')]
    self.assertEqual(cases, tests)

  @parameterized.parameters(
      {
          'argv': ['--test_randomize_ordering_seed=random'],
          'env': None,
      },
      {
          'argv': [],
          'env': {
              'TEST_RANDOMIZE_ORDERING_SEED': 'random',
          },
      },)
  def test_simple_randomization(self, argv, env):
    output, tests, exit_code = self._run_test(argv, env)
    self.assertEqual(0, exit_code, msg='command output: ' + output)
    self.assertIn('Randomizing test order with seed: ', output)
    cases = ['class A test ' + t for t in ('A', 'B', 'C')]
    # This may come back in any order; we just know it'll be the same
    # set of elements.
    self.assertSameElements(cases, tests)

  @parameterized.parameters(
      {
          'argv': ['--test_randomize_ordering_seed=1'],
          'env': None,
      },
      {
          'argv': [],
          'env': {
              'TEST_RANDOMIZE_ORDERING_SEED': '1'
          },
      },)
  def test_fixed_seed(self, argv, env):
    output, tests, exit_code = self._run_test(argv, env)
    self.assertEqual(0, exit_code, msg='command output: ' + output)
    self.assertIn('Randomizing test order with seed: 1', output)
    # Even though we know the seed, we need to shuffle the tests here, since
    # this behaves differently in Python2 vs Python3.
    shuffled_cases = ['A', 'B', 'C']
    random.Random(1).shuffle(shuffled_cases)
    cases = ['class A test ' + t for t in shuffled_cases]
    # We know what order this will come back for the random seed we've
    # specified.
    self.assertEqual(cases, tests)

  @parameterized.parameters(
      {
          'argv': ['--test_randomize_ordering_seed=0'],
          'env': {
              'TEST_RANDOMIZE_ORDERING_SEED': 'random'
          },
      },
      {
          'argv': [],
          'env': {
              'TEST_RANDOMIZE_ORDERING_SEED': '0'
          },
      },)
  def test_disabling_randomization(self, argv, env):
    output, tests, exit_code = self._run_test(argv, env)
    self.assertEqual(0, exit_code, msg='command output: ' + output)
    self.assertNotIn('Randomizing test order with seed:', output)
    cases = ['class A test ' + t for t in ('A', 'B', 'C')]
    self.assertEqual(cases, tests)


if __name__ == '__main__':
  absltest.main()
