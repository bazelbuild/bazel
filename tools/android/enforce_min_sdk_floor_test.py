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
"""Unit tests for enforce_min_sdk_floor.py."""

import unittest
from lxml import etree

from tools.android.enforce_min_sdk_floor import _BumpMinSdk
from tools.android.enforce_min_sdk_floor import _ValidateMinSdk

from tools.android.enforce_min_sdk_floor import MIN_SDK_ATTRIB
from tools.android.enforce_min_sdk_floor import MinSdkError
from tools.android.enforce_min_sdk_floor import USES_SDK

MANIFEST_NO_USES_SDK = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example" >
</manifest>
""".encode("utf-8")

MANIFEST_NO_MIN_SDK = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example" >
<uses-sdk/>
</manifest>
""".encode("utf-8")

MANIFEST_MIN_SDK_PLACEHOLDER = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example" >
<uses-sdk android:minSdkVersion="${minSdkVersion}" />
</manifest>
""".encode("utf-8")

MANIFEST_MIN_SDK = """<?xml version="1.0" encoding="utf-8"?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.example" >
<uses-sdk android:minSdkVersion="12" />
</manifest>
""".encode("utf-8")


class EnforceMinSdkFloorTest(unittest.TestCase):

  def test_bump_no_min_sdk_floor(self):
    out, _ = _BumpMinSdk(MANIFEST_NO_USES_SDK, 0)
    self.assertEqual(out, MANIFEST_NO_USES_SDK)

  def test_bump_no_uses_sdk(self):
    out, _ = _BumpMinSdk(MANIFEST_NO_USES_SDK, 11)
    min_sdk = etree.fromstring(out).find(USES_SDK).get(MIN_SDK_ATTRIB)
    self.assertEqual(min_sdk, "11")

  def test_bump_no_min_sdk_attrib(self):
    out, _ = _BumpMinSdk(MANIFEST_NO_MIN_SDK, 7)
    min_sdk = etree.fromstring(out).find(USES_SDK).get(MIN_SDK_ATTRIB)
    self.assertEqual(min_sdk, "7")

  def test_bump_min_sdk_attrib_placeholder(self):
    out, _ = _BumpMinSdk(MANIFEST_MIN_SDK_PLACEHOLDER, 13)
    self.assertEqual(out, MANIFEST_MIN_SDK_PLACEHOLDER)

  def test_bump_higher_min_sdk(self):
    out, _ = _BumpMinSdk(MANIFEST_MIN_SDK, 10)
    self.assertEqual(out, MANIFEST_MIN_SDK)

  def test_bump_lower_min_sdk(self):
    out, _ = _BumpMinSdk(MANIFEST_MIN_SDK, 14)
    min_sdk = etree.fromstring(out).find(USES_SDK).get(MIN_SDK_ATTRIB)
    self.assertEqual(min_sdk, "14")

  def test_validate_no_min_sdk_floor(self):
    _ = _ValidateMinSdk(MANIFEST_NO_USES_SDK, 0)

  def test_validate_no_uses_sdk(self):
    self.assertRaises(MinSdkError,
                      _ValidateMinSdk,
                      xml_content=MANIFEST_NO_USES_SDK,
                      min_sdk_floor=5)

  def test_validate_no_min_sdk_attrib(self):
    self.assertRaises(MinSdkError,
                      _ValidateMinSdk,
                      xml_content=MANIFEST_NO_MIN_SDK,
                      min_sdk_floor=19)

  def test_validate_min_sdk_attrib_placeholder(self):
    _ = _ValidateMinSdk(MANIFEST_MIN_SDK_PLACEHOLDER, 21)

  def test_validate_higher_min_sdk(self):
    _ = _ValidateMinSdk(MANIFEST_MIN_SDK, 8)

  def test_validate_lower_min_sdk(self):
    self.assertRaises(MinSdkError,
                      _ValidateMinSdk,
                      xml_content=MANIFEST_MIN_SDK,
                      min_sdk_floor=18)

if __name__ == "__main__":
  unittest.main()
