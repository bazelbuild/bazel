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

from __future__ import print_function

import os.path
import re
import subprocess
import tempfile
import unittest

from src.test.skylark import testenv


class SkylarkTest(unittest.TestCase):
  """Tests for Skylark.

  In a test file, chunks are separated by "---". Each chunk is evaluated
  separately. Use "###" to specify the expected error. If there is no "###",
  the test will succeed iff there is no error.
  """

  CHUNK_SEP = "---"
  ERR_SEP = "###"

  def chunks(self, path):
    code = []
    expected_errors = []
    with open(path) as f:
      for line in f:
        if line.strip() == self.CHUNK_SEP:
          yield code, expected_errors
          expected_errors = []
          code = []
        else:
          code.append(line)
          i = line.find(self.ERR_SEP)
          if i >= 0:
            expected_errors.append(line[i + len(self.ERR_SEP):].strip())
    yield code, expected_errors

  def evaluate(self, f):
    """Execute Skylark file, return stderr."""
    proc = subprocess.Popen(
        [testenv.SKYLARK_BINARY_PATH, f], stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    return stderr

  def check_output(self, output, expected):
    if expected and not output:
      raise Exception("Expected error:", expected)

    if output and not expected:
      raise Exception("Unexpected error:", output)

    for exp in expected:
      if not re.search(exp, output):
        raise Exception("Error `{}` not found, got: {}".format(exp, output))

  def testSuccess(self):
    tests = ["int.sky", "equality.sky", "and_or_not.sky"]
    for t in tests:
      print("===", t, "===")
      f = os.path.join(testenv.SKYLARK_TESTDATA_PATH, t)
      for chunk, expected in self.chunks(f):
        with tempfile.NamedTemporaryFile(suffix=".sky", delete=False) as tmp:
          tmp.writelines(chunk)
        output = self.evaluate(tmp.name)
        os.unlink(tmp.name)
        self.check_output(output, expected)


if __name__ == "__main__":
  unittest.main()
