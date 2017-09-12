# Copyright 2017 The Bazel Authors. All rights reserved.
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

import subprocess
import unittest


class SkylarkTest(unittest.TestCase):

  def testSuccess(self):
    binary_path = (
        "third_party/bazel/src/main/java/com/google/devtools/skylark/Skylark")
    test_dir = "third_party/bazel/src/test/skylark/testdata/"

    tests = ["int.sky", "equality.sky", "and_or_not.sky"]
    for t in tests:
      print subprocess.check_output([binary_path, test_dir + t])

  # TODO(laurentlb): Add support for negative tests (e.g. syntax errors).


if __name__ == "__main__":
  unittest.main()
