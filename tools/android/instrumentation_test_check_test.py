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
"""Unit tests for instrumentation_test_check."""

import unittest

from tools.android.instrumentation_test_check import _ExtractTargetPackageName
from tools.android.instrumentation_test_check import _ExtractTargetPackageToInstrument
from tools.android.instrumentation_test_check import _ValidateManifestPackageNames
from tools.android.instrumentation_test_check import ManifestError

INSTRUMENTATION_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.test" >
    <instrumentation android:targetPackage="com.example"
                     android:name="android.support.test.runner.AndroidJUnitRunner"/>
    <application android:label="Test"/>
</manifest>
"""

INCORRECT_INSTRUMENTATION_MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example.test" >
    <instrumentation android:targetPackage="not.com.example"
                     android:name="android.support.test.runner.AndroidJUnitRunner"/>
    <application android:label="Test"/>
</manifest>
"""

TARGET_MANIFEST = """<?xml version="2.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example" >
    <application android:label="App" />
</manifest>
"""


class InstrumentationTestCheckTest(unittest.TestCase):

  def test_extract_instrumentation_target_package(self):
    self.assertEqual(
        _ExtractTargetPackageToInstrument(INSTRUMENTATION_MANIFEST, ""),
        "com.example")

  def test_extract_target_package(self):
    self.assertEqual(
        _ExtractTargetPackageName(TARGET_MANIFEST, "unused"), "com.example")

  def test_target_package_check(self):
    self.assertEqual(
        _ValidateManifestPackageNames(INSTRUMENTATION_MANIFEST, "unused",
                                      TARGET_MANIFEST, "unused"),
        ("com.example", "com.example"))

  def test_target_package_check_failure(self):
    with self.assertRaises(ManifestError):
      _ValidateManifestPackageNames(INCORRECT_INSTRUMENTATION_MANIFEST,
                                    "unused", TARGET_MANIFEST, "unused")


if __name__ == "__main__":
  unittest.main()
