# Copyright 2022 The Bazel Authors.
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

# Load the shared library from data and call the "foo" function defined by it to
# verify that its runfiles are sufficient to run the shared library.
import ctypes
import os
import unittest
import python.runfiles as runfiles


class TestStandaloneSharedLibrary(unittest.TestCase):

  def test_call_foo(self):
    lib_path = runfiles.Create().Rlocation(os.getenv("FOO_SO"))
    lib = ctypes.CDLL(lib_path)
    if lib_path.endswith(".dll"):
      self.assertEqual(getattr(lib, "?foo@@YAHXZ")(), 42)
    else:
      self.assertEqual(lib._Z3foov(), 42)


if __name__ == "__main__":
  unittest.main()
