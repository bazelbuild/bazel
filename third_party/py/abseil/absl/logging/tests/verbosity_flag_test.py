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

"""Tests -v/--verbosity flag and logging.root level's sync behavior."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

assert logging.root.getEffectiveLevel() == logging.WARN, (
    'default logging.root level should be WARN, but found {}'.format(
        logging.root.getEffectiveLevel()))

# This is here to test importing logging won't change the level.
logging.root.setLevel(logging.ERROR)

assert logging.root.getEffectiveLevel() == logging.ERROR, (
    'logging.root level should be changed to ERROR, but found {}'.format(
        logging.root.getEffectiveLevel()))

from absl import flags
from absl import logging as _  # pylint: disable=unused-import
from absl.testing import absltest

FLAGS = flags.FLAGS

assert FLAGS['verbosity'].value == -1, (
    '-v/--verbosity should be -1 before flags are parsed.')

assert logging.root.getEffectiveLevel() == logging.ERROR, (
    'logging.root level should be kept to ERROR, but found {}'.format(
        logging.root.getEffectiveLevel()))


class VerbosityFlagTest(absltest.TestCase):

  def test_default_value_after_init(self):
    self.assertEqual(0, FLAGS.verbosity)
    self.assertEqual(logging.INFO, logging.root.getEffectiveLevel())


if __name__ == '__main__':
  absltest.main()
