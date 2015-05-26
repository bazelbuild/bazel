# Copyright 2015 Google Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import unittest

from tools.android import proguard_whitelister


class ValidateProguardTest(unittest.TestCase):

  def testValidConfig(self):
    path = os.path.join(
        os.path.dirname(__file__), "proguard_whitelister_input.cfg")
    with open(path) as config:
      self.assertEqual([], proguard_whitelister.Validate(config.read()))

  def testInvalidNoteConfig(self):
    self.assertEqual(["-dontnote"], proguard_whitelister.Validate(
        """# We don't want libraries disabling notes globally.
        -dontnote
        """))

  def testInvalidWarnConfig(self):
    self.assertEqual(["-dontwarn"], proguard_whitelister.Validate(
        """# We don't want libraries disabling warnings globally.
        -dontwarn
        """))

  def testInvalidOptimizationConfig(self):
    self.assertEqual(["-optimizations"], proguard_whitelister.Validate(
        """#We don't want libraries disabling global optimizations.
        -optimizations !class/merging/*,!code/allocation/variable
        """))

  def testMultipleInvalidArgs(self):
    self.assertEqual(
        ["-optimizations", "-dontnote"], proguard_whitelister.Validate(
            """#We don't want libraries disabling global optimizations.
            -optimizations !class/merging/*,!code/allocation/variable
            -dontnote
            """))


if __name__ == "__main__":
  unittest.main()
