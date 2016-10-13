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

import filecmp
import os
import unittest
import zipfile

from tools.android import aar_embedded_jars_extractor


class EmbeddedJarExtractorTest(unittest.TestCase):
  """Unit tests for aar_embedded_jars_extractor.py."""

  def testPassingJarFile(self):
    bjar = zipfile.ZipFile("b.jar", "w")
    bjar.close()
    azip = zipfile.ZipFile("a.zip", "w")
    azip.write("b.jar")
    azip.close()
    if not os.path.exists("output"):
      os.mkdir("output")
    aar_embedded_jars_extractor.ExtractEmbeddedJar("a.zip", "b.jar", "output")
    self.assertTrue(filecmp.cmp("b.jar", "output/b.jar"))

  def testMissingJarFile(self):
    azip = zipfile.ZipFile("a.zip", "w")
    azip.close()
    if not os.path.exists("output"):
      os.mkdir("output")
    aar_embedded_jars_extractor.ExtractEmbeddedJar("a.zip", "b.jar", "output")
    bjar = zipfile.ZipFile("output/b.jar", "r")
    self.assertEqual(["META-INF/MANIFEST.MF"], bjar.namelist())


if __name__ == "__main__":
  unittest.main()
