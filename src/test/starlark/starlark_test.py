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
import sys
import tempfile
import unittest

from src.test.starlark import testenv


class StarlarkTest(unittest.TestCase):
  """Tests for Starlark.

  In a test file, chunks are separated by "---". Each chunk is evaluated
  separately. Use "###" to specify the expected error. If there is no "###",
  the test will succeed iff there is no error.
  """

  CHUNK_SEP = "---"
  ERR_SEP = "###"
  seen_error = False

  def chunks(self, path):
    code = []
    expected_errors = []
    with open(path, mode="rb") as f:
      for line in f:
        line = line.decode("utf-8")
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
    """Execute Starlark file, return stderr."""
    proc = subprocess.Popen(
        [testenv.STARLARK_BINARY_PATH, f], stderr=subprocess.PIPE)
    _, stderr = proc.communicate()
    return stderr

  def check_output(self, output, expected):
    if expected and not output:
      self.seen_error = True
      print("Expected error:", expected)

    if output and not expected:
      self.seen_error = True
      print("Unexpected error:", output)

    for exp in expected:
      # Try both substring and regex matching.
      if exp not in output and not re.search(exp, output):
        self.seen_error = True
        print("Error `{}` not found, got: `{}`".format(exp, output))

  PRELUDE = """
def assert_eq(x, y):
  if x != y:
    fail("%r != %r" % (x, y))

def assert_(cond, msg="assertion failed"):
  if not cond:
    fail(msg)
"""

  def testFile(self):
    t = test_file
    print("===", t, "===")
    f = os.path.join(testenv.STARLARK_TESTDATA_PATH, t)
    for chunk, expected in self.chunks(f):
      with tempfile.NamedTemporaryFile(
          mode="wb", suffix=".sky", delete=False) as tmp:
        lines = [line.encode("utf-8") for line in
                 [self.PRELUDE] + chunk]
        tmp.writelines(lines)
      output = self.evaluate(tmp.name).decode("utf-8")
      os.unlink(tmp.name)
      self.check_output(output, expected)
    if self.seen_error:
      raise Exception("Test failed")


if __name__ == "__main__":
  # Test filename is the last argument on the command-line.
  test_file = sys.argv[-1]
  unittest.main(argv=sys.argv[1:])
