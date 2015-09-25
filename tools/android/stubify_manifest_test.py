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

from tools.android.stubify_manifest import ANDROID
from tools.android.stubify_manifest import BadManifestException
from tools.android.stubify_manifest import READ_EXTERNAL_STORAGE
from tools.android.stubify_manifest import STUB_APPLICATION
from tools.android.stubify_manifest import Stubify


MANIFEST_WITH_APPLICATION = """
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.package">
  <application android:name="old.application">
  </application>
</manifest>
"""

MANIFEST_WITHOUT_APPLICATION = """
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
  package="com.google.package">
</manifest>
"""

MANIFEST_WITH_PERMISSION = """
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
  package="com.google.package">
  <uses-permission android:name="android.permission.READ_EXTERNAL_STORAGE"/>
</manifest>
"""

BAD_MANIFEST = """
<b>Hello World!</b>
"""

MULTIPLE_APPLICATIONS = """
<manifest
  xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.package">
  <application android:name="old.application">
  </application>
  <application android:name="old.application">
  </application>
</manifest>
"""


class StubifyTest(unittest.TestCase):

  def GetApplication(self, manifest_string):
    manifest = ElementTree.fromstring(manifest_string)
    application = manifest.find("application")
    return application.get("{%s}name" % ANDROID)

  def testReplacesOldApplication(self):
    new_manifest, old_application, app_pkg = Stubify(MANIFEST_WITH_APPLICATION)
    self.assertEqual("com.google.package", app_pkg)
    self.assertEqual("old.application", old_application)
    self.assertEqual(STUB_APPLICATION, self.GetApplication(new_manifest))

  def testAddsNewAplication(self):
    new_manifest, old_application, app_pkg = (
        Stubify(MANIFEST_WITHOUT_APPLICATION))
    self.assertEqual("com.google.package", app_pkg)
    self.assertEqual("android.app.Application", old_application)
    self.assertEqual(STUB_APPLICATION, self.GetApplication(new_manifest))

  def assertHasPermission(self, manifest_string, permission):
    manifest = ElementTree.fromstring(manifest_string)
    nodes = manifest.findall(
        'uses-permission[@android:name="%s"]' % permission,
        namespaces={"android": ANDROID})
    self.assertEqual(1, len(nodes))

  def testAddsPermission(self):
    self.assertHasPermission(
        Stubify(MANIFEST_WITH_APPLICATION)[0], READ_EXTERNAL_STORAGE)

  def testDoesNotDuplicatePermission(self):
    self.assertHasPermission(
        Stubify(MANIFEST_WITH_PERMISSION)[0], READ_EXTERNAL_STORAGE)

  def testBadManifest(self):
    with self.assertRaises(BadManifestException):
      Stubify(BAD_MANIFEST)

  def testTooManyApplications(self):
    with self.assertRaises(BadManifestException):
      Stubify(MULTIPLE_APPLICATIONS)


if __name__ == "__main__":
  unittest.main()
