# Copyright 2015 The Bazel Authors. All rights reserved.
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

"""Unit tests for stubify_application_manifest."""

import unittest
from xml.etree import ElementTree

from tools.android.build_split_manifest import BuildSplitManifest


MAIN_MANIFEST = """
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.package"
    android:versionCode="1"
    android:versionName="1.0">
</manifest>
"""


class BuildSplitManifestTest(unittest.TestCase):

  def testNoPackageOveride(self):
    split = BuildSplitManifest(MAIN_MANIFEST, None, "split", False)
    manifest = ElementTree.fromstring(split)
    self.assertEqual("com.google.package",
                     manifest.get("package"))

  def testPackageOveride(self):
    split = BuildSplitManifest(MAIN_MANIFEST, "package.other", "split", False)
    manifest = ElementTree.fromstring(split)
    self.assertEqual("package.other",
                     manifest.get("package"))

  def testSplitName(self):
    split = BuildSplitManifest(MAIN_MANIFEST, None, "my.little.splony", False)
    manifest = ElementTree.fromstring(split)
    self.assertEqual("my.little.splony", manifest.get("split"))


if __name__ == "__main__":
  unittest.main()
