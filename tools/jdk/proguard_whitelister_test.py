# Copyright 2015 The Bazel Authors. All rights reserved.
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

from tools.jdk import proguard_whitelister


class ProguardConfigValidatorTest(unittest.TestCase):

  def _CreateValidator(self, input_path, output_path):
    return proguard_whitelister.ProguardConfigValidator(input_path, output_path)

  def testValidConfig(self):
    input_path = os.path.join(
        os.path.dirname(__file__), "proguard_whitelister_test_input.cfg")
    tmpdir = os.environ["TEST_TMPDIR"]
    output_path = os.path.join(tmpdir, "proguard_whitelister_test_output.cfg")
    # This will raise an exception if the config is invalid.
    self._CreateValidator(input_path, output_path).ValidateAndWriteOutput()
    with file(output_path) as output:
      self.assertTrue(("# Merged from %s" % input_path) in output.read())

  def _TestInvalidConfig(self, invalid_args, config):
    tmpdir = os.environ["TEST_TMPDIR"]
    input_path = os.path.join(tmpdir, "proguard_whitelister_test_input.cfg")
    with open(input_path, "w") as f:
      f.write(config)
    output_path = os.path.join(tmpdir, "proguard_whitelister_test_output.cfg")
    validator = self._CreateValidator(input_path, output_path)
    try:
      validator.ValidateAndWriteOutput()
      self.fail()
    except RuntimeError as e:
      for invalid_arg in invalid_args:
        self.assertTrue(invalid_arg in str(e))

  def testInvalidNoteConfig(self):
    self._TestInvalidConfig(["-dontnote"], """\
# We don"t want libraries disabling notes globally.
-dontnote""")

  def testInvalidWarnConfig(self):
    self._TestInvalidConfig(["-dontwarn"], """\
# We don"t want libraries disabling warnings globally.
-dontwarn""")

  def testInvalidOptimizationConfig(self):
    self._TestInvalidConfig(["-optimizations"], """\
# We don"t want libraries disabling global optimizations.
-optimizations !class/merging/*,!code/allocation/variable""")

  def testMultipleInvalidArgs(self):
    self._TestInvalidConfig(["-optimizations", "-dontnote"], """\
# We don"t want libraries disabling global optimizations.
-optimizations !class/merging/*,!code/allocation/variable
-dontnote""")


if __name__ == "__main__":
  unittest.main()
