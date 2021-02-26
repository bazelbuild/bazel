# Copyright 2021 The Bazel Authors. All rights reserved.
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
"""Tests for aar_embedded_proguard_extractor."""

import io
import os
import unittest
import zipfile

from tools.android import aar_embedded_proguard_extractor


class AarEmbeddedProguardExtractor(unittest.TestCase):
  """Unit tests for aar_embedded_proguard_extractor.py."""

  # Python 2 alias
  if not hasattr(unittest.TestCase, "assertCountEqual"):

    def assertCountEqual(self, *args):
      return self.assertItemsEqual(*args)

  def setUp(self):
    super(AarEmbeddedProguardExtractor, self).setUp()
    os.chdir(os.environ["TEST_TMPDIR"])

  def testNoProguardTxt(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    proguard_file = io.BytesIO()
    aar_embedded_proguard_extractor.ExtractEmbeddedProguard(aar, proguard_file)
    proguard_file.seek(0)
    self.assertEqual(b"", proguard_file.read())

  def testWithProguardTxt(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("proguard.txt", "hello world")
    proguard_file = io.BytesIO()
    aar_embedded_proguard_extractor.ExtractEmbeddedProguard(aar, proguard_file)
    proguard_file.seek(0)
    self.assertEqual(b"hello world", proguard_file.read())


if __name__ == "__main__":
  unittest.main()
