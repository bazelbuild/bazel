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

"""Tests for aar_embedded_jars_extractor."""

import io
import os
import shutil
import unittest
import zipfile

from tools.android import aar_embedded_jars_extractor


class AarEmbeddedJarsExtractor(unittest.TestCase):
  """Unit tests for aar_embedded_jars_extractor.py."""

  # Python 2 alias
  if not hasattr(unittest.TestCase, "assertCountEqual"):

    def assertCountEqual(self, *args):
      return self.assertItemsEqual(*args)

  def setUp(self):
    os.chdir(os.environ["TEST_TMPDIR"])

  def tearDown(self):
    shutil.rmtree("out_dir")

  def testNoJars(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    param_file = io.BytesIO()
    os.makedirs("out_dir")
    aar_embedded_jars_extractor.ExtractEmbeddedJars(aar, param_file, "out_dir")
    self.assertEqual([], os.listdir("out_dir"))
    param_file.seek(0)
    self.assertEqual(b"--exclude_build_data\n", param_file.read())

  def testClassesJarAndLibsJars(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("classes.jar", "")
    aar.writestr("libs/a.jar", "")
    aar.writestr("libs/b.jar", "")
    param_file = io.BytesIO()
    os.makedirs("out_dir")
    aar_embedded_jars_extractor.ExtractEmbeddedJars(aar, param_file, "out_dir")
    self.assertCountEqual(["classes.jar", "libs"], os.listdir("out_dir"))
    self.assertCountEqual(["a.jar", "b.jar"], os.listdir("out_dir/libs"))
    param_file.seek(0)
    self.assertEqual(
        [b"--exclude_build_data\n",
         b"--sources\n",
         b"out_dir/classes.jar\n",
         b"--sources\n",
         b"out_dir/libs/a.jar\n",
         b"--sources\n",
         b"out_dir/libs/b.jar\n"],
        param_file.readlines())

  def testOnlyClassesJar(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("classes.jar", "")
    param_file = io.BytesIO()
    os.makedirs("out_dir")
    aar_embedded_jars_extractor.ExtractEmbeddedJars(aar, param_file, "out_dir")
    self.assertEqual(["classes.jar"], os.listdir("out_dir"))
    param_file.seek(0)
    self.assertEqual(
        [b"--exclude_build_data\n",
         b"--sources\n",
         b"out_dir/classes.jar\n"],
        param_file.readlines())


if __name__ == "__main__":
  unittest.main()
