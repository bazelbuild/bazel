# Copyright 2016 The Bazel Authors. All rights reserved.
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

"""Tests for aar_native_libs_zip_creator."""

import io
import unittest
import zipfile

from tools.android import aar_native_libs_zip_creator


class AarNativeLibsZipCreatorTest(unittest.TestCase):
  """Unit tests for aar_native_libs_zip_creator.py."""

  def testAarWithNoLibs(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    outzip = zipfile.ZipFile(io.BytesIO(), "w")
    aar_native_libs_zip_creator.CreateNativeLibsZip(aar, "x86", outzip)
    self.assertEquals([], outzip.namelist())

  def testAarWithMissingLibs(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("jni/armeabi/foo.so", "")
    outzip = zipfile.ZipFile(io.BytesIO(), "w")
    self.assertRaises(
        aar_native_libs_zip_creator.UnsupportedArchitectureException,
        aar_native_libs_zip_creator.CreateNativeLibsZip,
        aar, "x86", outzip)

  def testAarWithAllLibs(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("jni/x86/foo.so", "")
    aar.writestr("jni/armeabi/foo.so", "")
    outzip = zipfile.ZipFile(io.BytesIO(), "w")
    aar_native_libs_zip_creator.CreateNativeLibsZip(aar, "x86", outzip)
    self.assertIn("lib/x86/foo.so", outzip.namelist())
    self.assertNotIn("lib/armeabi/foo.so", outzip.namelist())


if __name__ == "__main__":
  unittest.main()
