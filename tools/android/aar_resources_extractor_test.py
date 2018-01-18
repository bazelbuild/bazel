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

"""Tests for aar_resources_extractor."""

import io
import os
import shutil
import unittest
import zipfile

from tools.android import aar_resources_extractor


def _HostPath(path):
  return os.path.normpath(path)


class AarResourcesExtractorTest(unittest.TestCase):
  """Unit tests for aar_resources_extractor.py."""

  # Python 2 alias
  if not hasattr(unittest.TestCase, "assertCountEqual"):

    def assertCountEqual(self, *args):
      return self.assertItemsEqual(*args)

  def setUp(self):
    os.chdir(os.environ["TEST_TMPDIR"])

  def tearDown(self):
    shutil.rmtree("out_dir")

  def DirContents(self, d):
    return [
        _HostPath(path + "/" + f)
        for (path, _, files) in os.walk(d) for f in files
    ]

  def testNoResources(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("res/", "")
    os.makedirs("out_dir")
    aar_resources_extractor.ExtractResources(aar, "out_dir")
    self.assertEqual([_HostPath("out_dir/res/values/empty.xml")],
                     self.DirContents("out_dir"))
    with open("out_dir/res/values/empty.xml", "r") as empty_xml:
      self.assertEqual("<resources/>", empty_xml.read())

  def testContainsResources(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("res/values/values.xml", "some values")
    aar.writestr("res/layouts/layout.xml", "some layout")
    aar.writestr("assets/a", "some asset")
    os.makedirs("out_dir")
    aar_resources_extractor.ExtractResources(aar, "out_dir")
    expected_resources = [
        _HostPath("out_dir/res/values/values.xml"),
        _HostPath("out_dir/res/layouts/layout.xml")
    ]
    self.assertCountEqual(expected_resources, self.DirContents("out_dir"))
    with open("out_dir/res/values/values.xml", "r") as values_xml:
      self.assertEqual("some values", values_xml.read())
    with open("out_dir/res/layouts/layout.xml", "r") as layout_xml:
      self.assertEqual("some layout", layout_xml.read())

  def testNoAssets(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("assets/", "")
    os.makedirs("out_dir")
    aar_resources_extractor.ExtractAssets(aar, "out_dir")
    expected_assets = [
        _HostPath("out_dir/assets/empty_asset_generated_by_bazel~")
    ]
    self.assertEqual(expected_assets, self.DirContents("out_dir"))
    self.assertEqual(
        os.stat("out_dir/assets/empty_asset_generated_by_bazel~").st_size, 0)

  def testContainsAssets(self):
    aar = zipfile.ZipFile(io.BytesIO(), "w")
    aar.writestr("res/values/values.xml", "some values")
    aar.writestr("assets/a", "some asset")
    aar.writestr("assets/b", "some other asset")
    os.makedirs("out_dir")
    aar_resources_extractor.ExtractAssets(aar, "out_dir")
    expected_resources = [
        _HostPath("out_dir/assets/a"),
        _HostPath("out_dir/assets/b")
    ]
    self.assertCountEqual(expected_resources, self.DirContents("out_dir"))
    with open("out_dir/assets/a", "r") as values_xml:
      self.assertEqual("some asset", values_xml.read())
    with open("out_dir/assets/b", "r") as layout_xml:
      self.assertEqual("some other asset", layout_xml.read())


if __name__ == "__main__":
  unittest.main()
