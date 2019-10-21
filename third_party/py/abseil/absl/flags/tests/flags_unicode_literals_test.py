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

"""Test the use of flags when from __future__ import unicode_literals is on."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from absl import flags
from absl.testing import absltest


flags.DEFINE_string('seen_in_crittenden', 'alleged mountain lion',
                    'This tests if unicode input to these functions works.')


class FlagsUnicodeLiteralsTest(absltest.TestCase):

  def testUnicodeFlagNameAndValueAreGood(self):
    alleged_mountain_lion = flags.FLAGS.seen_in_crittenden
    self.assertTrue(
        isinstance(alleged_mountain_lion, type(u'')),
        msg='expected flag value to be a {} not {}'.format(
            type(u''), type(alleged_mountain_lion)))
    self.assertEqual(alleged_mountain_lion, u'alleged mountain lion')


if __name__ == '__main__':
  absltest.main()
