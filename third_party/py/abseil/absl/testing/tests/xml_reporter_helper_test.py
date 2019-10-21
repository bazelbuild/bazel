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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

from absl import flags
from absl.testing import absltest


FLAGS = flags.FLAGS
flags.DEFINE_boolean('set_up_module_error', False,
                     'Cause setupModule to error.')
flags.DEFINE_boolean('tear_down_module_error', False,
                     'Cause tearDownModule to error.')

flags.DEFINE_boolean('set_up_class_error', False, 'Cause setUpClass to error.')
flags.DEFINE_boolean('tear_down_class_error', False,
                     'Cause tearDownClass to error.')

flags.DEFINE_boolean('set_up_error', False, 'Cause setUp to error.')
flags.DEFINE_boolean('tear_down_error', False, 'Cause tearDown to error.')
flags.DEFINE_boolean('test_error', False, 'Cause the test to error.')

flags.DEFINE_boolean('set_up_fail', False, 'Cause setUp to fail.')
flags.DEFINE_boolean('tear_down_fail', False, 'Cause tearDown to fail.')
flags.DEFINE_boolean('test_fail', False, 'Cause the test to fail.')

flags.DEFINE_float('random_error', 0.0,
                   '0 - 1.0: fraction of a random failure at any step',
                   lower_bound=0.0, upper_bound=1.0)


def _random_error():
  return random.random() < FLAGS.random_error


def setUpModule():
  if FLAGS.set_up_module_error or _random_error():
    raise Exception('setUpModule Errored!')


def tearDownModule():
  if FLAGS.tear_down_module_error or _random_error():
    raise Exception('tearDownModule Errored!')


class FailableTest(absltest.TestCase):

  @classmethod
  def setUpClass(cls):
    if FLAGS.set_up_class_error or _random_error():
      raise Exception('setUpClass Errored!')

  @classmethod
  def tearDownClass(cls):
    if FLAGS.tear_down_class_error or _random_error():
      raise Exception('tearDownClass Errored!')

  def setUp(self):
    if FLAGS.set_up_error or _random_error():
      raise Exception('setUp Errored!')

    if FLAGS.set_up_fail:
      self.fail('setUp Failed!')

  def tearDown(self):
    if FLAGS.tear_down_error or _random_error():
      raise Exception('tearDown Errored!')

    if FLAGS.tear_down_fail:
      self.fail('tearDown Failed!')

  def test(self):
    if FLAGS.test_error or _random_error():
      raise Exception('test Errored!')

    if FLAGS.test_fail:
      self.fail('test Failed!')


if __name__ == '__main__':
  absltest.main()
