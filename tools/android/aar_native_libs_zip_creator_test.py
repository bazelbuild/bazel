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

import hashlib
import io
import tempfile
import time
import unittest
import zipfile

from tools.android import aar_native_libs_zip_creator


def md5(buf):
  hash_md5 = hashlib.md5()
  hash_md5.update(buf)
  return hash_md5.hexdigest()


def md5_from_file(fname):
  hash_md5 = hashlib.md5()
  with open(fname, "rb") as f:
    for chunk in iter(lambda: f.read(4096), b""):
      hash_md5.update(chunk)
  return hash_md5.hexdigest()


class AarNativeLibsZipCreatorTest(unittest.TestCase):
  """Unit tests for aar_native_libs_zip_creator.py."""

  def testAarWithNoLibs(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    outzip = zipfile.ZipFile(io.BytesIO(), "w")
    aar_native_libs_zip_creator.CreateNativeLibsZip(aar, "x86", outzip)
    self.assertEqual([], outzip.namelist())

  def testAarWithMissingLibs(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("jni/armeabi/foo.so", "")
    outzip = zipfile.ZipFile(io.BytesIO(), "w")
    self.assertRaises(
        aar_native_libs_zip_creator.UnsupportedArchitectureError,
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

  def testMultipleInvocationConsistency(self):
    input_aar = tempfile.NamedTemporaryFile(delete=False)
    aar = zipfile.ZipFile(input_aar.name, "w")
    aar.writestr(zipfile.ZipInfo(filename="jni/x86/foo.so"), "foo")
    aar.writestr(zipfile.ZipInfo(filename="jni/x86/bar.so"), "bar")
    aar.close()
    input_aar.close()
    # CreateNativeLibsZip expects a readonly file, this is not required but
    # more correct
    readonly_aar = zipfile.ZipFile(input_aar.name, "r")

    outfile1 = tempfile.NamedTemporaryFile(delete=False)
    outzip1 = zipfile.ZipFile(outfile1.name, "w")
    aar_native_libs_zip_creator.CreateNativeLibsZip(readonly_aar, "x86",
                                                    outzip1)
    outfile1.close()

    # Must be more than 1 second because last modified date changes on second
    # basis
    time.sleep(2)

    outfile2 = tempfile.NamedTemporaryFile(delete=False)
    outzip2 = zipfile.ZipFile(outfile2.name, "w")
    aar_native_libs_zip_creator.CreateNativeLibsZip(readonly_aar, "x86",
                                                    outzip2)
    outfile2.close()

    self.assertIn("lib/x86/foo.so", outzip1.namelist())
    self.assertIn("lib/x86/bar.so", outzip1.namelist())
    self.assertNotEqual(
        md5(outzip1.read("lib/x86/foo.so")),
        md5(outzip1.read("lib/x86/bar.so")))

    self.assertIn("lib/x86/foo.so", outzip2.namelist())
    self.assertIn("lib/x86/bar.so", outzip2.namelist())
    self.assertNotEqual(
        md5(outzip1.read("lib/x86/foo.so")),
        md5(outzip1.read("lib/x86/bar.so")))

    # The hash for the output zips must always match if the inputs match.
    # Otherwise, there will be a cache miss which will produce poort build
    # times.
    self.assertEqual(md5_from_file(outfile1.name), md5_from_file(outfile2.name))


if __name__ == "__main__":
  unittest.main()
