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

"""This file contains unit tests for the merge_manifests script."""

import re
import unittest
import xml.dom.minidom

from tools.android import merge_manifests

FIRST_MANIFEST = """<?xml version='1.0' encoding='utf-8'?>
<manifest
    xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.android.apps.testapp"
    android:versionCode="70"
    android:versionName="1.0">
  <uses-sdk android:minSdkVersion="10"/>
  <uses-feature android:name="android.hardware.nfc" android:required="true" />
  <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
  <application
      android:icon="@drawable/icon"
      android:name="com.google.android.apps.testapp.TestApplication"
      android:theme="@style/Theme.Test"
      android:label="@string/app_name">
    <!--  START LIBRARIES (Maintain Alphabetic order) -->
    <!-- NFC extras -->
    <uses-library android:name="com.google.android.nfc_extras" android:required="false"/>
    <!--  END LIBRARIES -->
    <!--  START ACTIVITIES (Maintain Alphabetic order) -->
    <!-- Entry point activity - navigation and title bar. -->
    <activity
        android:name=".entrypoint.EntryPointActivityGroup"
        android:screenOrientation="portrait"
        android:launchMode="singleTop"/>
    <activity android:name=".ui.topup.TopUpActivity" />
    <service android:name=".nfcevent.NfcEventService" />
    <receiver
        android:name="com.receiver.TestReceiver"
        android:process="@string/receiver_service_name">
      <!-- Receive the actual message -->
      <intent-filter>
        <action
            android:name="android.intent.action.USER_PRESENT"/>
      </intent-filter>
    </receiver>
    <provider
        android:name=".dataaccess.persistence.ContentProvider"
        android:authorities="com.google.android.apps.testapp"
        android:exported="false" />
  </application>
</manifest>
"""

SECOND_MANIFEST = """<?xml version='1.0' encoding='utf-8'?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.android.apps.testapp2"
    android:versionCode="1"
    android:versionName="1.0">
  <permission android:name="com.google.android.apps.foo.C2D_MESSAGE"
      android:protectionLevel="signature" />
  <uses-sdk android:minSdkVersion="5" />
  <uses-feature android:name="android.hardware.nfc" android:required="true" />
  <uses-permission android:name="android.permission.READ_LOGS" />
  <uses-permission android:name="android.permission.INTERNET" />
  <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
  <!-- Comment for permission android.permission.GET_ACCOUNTS.
    This is just to make sure the comment is being merged correctly.
  -->
  <uses-permission android:name="android.permission.GET_ACCOUNTS" />
  <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />

  <application
      android:icon="@drawable/icon"
      android:name="com.google.android.apps.testapp.TestApplication2"
      android:theme="@style/Theme.Test2"
      android:label="@string/app_name"
      android:backupAgent="FooBar">
    <activity android:name=".ui.home.HomeActivity"
        android:label="@string/app_name" >
      <intent-filter>
        <action android:name="android.intent.action.MAIN"/>
        <category android:name="android.intent.category.LAUNCHER"/>
      </intent-filter>
    </activity>
    <activity android:name=".TestActivity2"></activity>
    <activity android:name=".PreviewActivity"></activity>
    <activity android:name=".ShowTextActivity" android:excludeFromRecents="true"></activity>
    <activity android:name=".ShowStringListActivity"
        android:excludeFromRecents="true"
        android:parentActivityName=".ui.home.HomeActivity">
    </activity>
    <service android:name=".TestService">
      <meta-data android:name="param" android:value="value"/>
    </service>
    <service android:name=".nfcevent.NfcEventService" />
    <receiver android:name=".ConnectivityReceiver" android:enabled="false" >
      <intent-filter>
        <action android:name="android.net.conn.CONNECTIVITY_CHANGE" />
      </intent-filter>
    </receiver>
    <activity-alias android:name="BarFoo" android:targetActivity=".FooBar" />
    <provider
        android:name="some.package.with.inner.class$AnInnerClass" />
    <provider
        android:name="${packageName}"
        android:authorities="${packageName}.${packageName}"
        android:exported="false" />
    <provider
        android:name="${packageName}.PlaceHolderProviderName"
        android:authorities="PlaceHolderProviderAuthorities.${packageName}"
        android:exported="false" />
    <activity
        android:name="activityPrefix.${packageName}.activitySuffix">
      <intent-filter>
        <action android:name="actionPrefix.${packageName}.actionSuffix" />
      </intent-filter>
    </activity>
  </application>
</manifest>
"""

THIRD_MANIFEST = """<?xml version='1.0' encoding='utf-8'?>
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.android.apps.testapp3"
    android:versionCode="3"
    android:versionName="1.30">
  <uses-sdk android:minSdkVersion="14" />
  <uses-feature android:name="android.hardware.nfc" android:required="true" />
  <uses-permission android:name="android.permission.READ_LOGS" />
  <uses-permission android:name="android.permission.INTERNET" />
  <application>
    <activity android:name=".ui.home.HomeActivity"
        android:label="@string/app_name" >
      <intent-filter>
        <action android:name="android.intent.action.MAIN"/>
        <category android:name="android.intent.category.LAUNCHER"/>
      </intent-filter>
    </activity>
    <activity android:name="TestActivity"></activity>
    <service android:name=".TestService" />
    <receiver android:name=".ConnectivityReceiver" android:enabled="true" >
      <intent-filter>
        <action android:name="android.net.conn.CONNECTIVITY_CHANGER" />
      </intent-filter>
    </receiver>
  </application>
</manifest>
"""

MANUALLY_MERGED = """<?xml version='1.0' encoding='utf-8'?>
<manifest
    xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.android.apps.testapp"
    android:versionCode="70"
    android:versionName="1.0">
  <!-- *** WARNING *** DO NOT EDIT! THIS IS GENERATED MANIFEST BY MERGE_MANIFEST TOOL.
  Merger manifest:
    FIRST_MANIFEST
  Mergee manifests:
    SECOND_MANIFEST
    THIRD_MANIFEST
   -->
  <uses-sdk android:minSdkVersion="10"/>
  <uses-feature android:name="android.hardware.nfc" android:required="true" />
  <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
  <!-- Merged from file: SECOND_MANIFEST -->
  <permission android:name="com.google.android.apps.foo.C2D_MESSAGE" android:protectionLevel="signature" />
  <!-- Merged from file: SECOND_MANIFEST -->
  <uses-permission android:name="android.permission.INTERNET" />
  <!-- Merged from file: SECOND_MANIFEST -->
  <uses-permission android:name="android.permission.ACCESS_NETWORK_STATE" />
  <!-- Merged from file: SECOND_MANIFEST -->
  <!-- Comment for permission android.permission.GET_ACCOUNTS.
    This is just to make sure the comment is being merged correctly.
  -->
  <uses-permission android:name="android.permission.GET_ACCOUNTS" />
  <application
      android:icon="@drawable/icon"
      android:name="com.google.android.apps.testapp.TestApplication"
      android:theme="@style/Theme.Test"
      android:label="@string/app_name">
    <!--  START LIBRARIES (Maintain Alphabetic order) -->
    <!-- NFC extras -->
    <uses-library android:name="com.google.android.nfc_extras" android:required="false"/>
    <!--  END LIBRARIES -->
    <!--  START ACTIVITIES (Maintain Alphabetic order) -->
    <!-- Entry point activity - navigation and title bar. -->
    <activity
        android:name="com.google.android.apps.testapp.entrypoint.EntryPointActivityGroup"
        android:screenOrientation="portrait"
        android:launchMode="singleTop"/>
    <activity android:name="com.google.android.apps.testapp.ui.topup.TopUpActivity" />
    <service android:name="com.google.android.apps.testapp.nfcevent.NfcEventService" />
    <receiver
         android:name="com.receiver.TestReceiver"
         android:process="@string/receiver_service_name">
      <!-- Receive the actual message -->
      <intent-filter>
        <action
            android:name="android.intent.action.USER_PRESENT"/>
      </intent-filter>
    </receiver>
    <provider android:authorities="com.google.android.apps.testapp" android:exported="false"
        android:name="com.google.android.apps.testapp.dataaccess.persistence.ContentProvider"/>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity android:label="@string/app_name" android:name="com.google.android.apps.testapp2.ui.home.HomeActivity">
      <intent-filter>
        <action android:name="android.intent.action.MAIN"/>
        <category android:name="android.intent.category.LAUNCHER"/>
      </intent-filter>
    </activity>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity android:name="com.google.android.apps.testapp2.TestActivity2"></activity>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity android:name="com.google.android.apps.testapp2.PreviewActivity"></activity>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity android:name="com.google.android.apps.testapp2.ShowTextActivity"
        android:excludeFromRecents="true"></activity>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity android:name="com.google.android.apps.testapp2.ShowStringListActivity"
        android:excludeFromRecents="true"
        android:parentActivityName="com.google.android.apps.testapp2.ui.home.HomeActivity">
    </activity>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity
        android:name="activityPrefix.com.google.android.apps.testapp.activitySuffix">
      <intent-filter>
        <action android:name="actionPrefix.com.google.android.apps.testapp.actionSuffix" />
      </intent-filter>
    </activity>
    <!-- Merged from file: SECOND_MANIFEST -->
    <provider
        android:name="some.package.with.inner.class$AnInnerClass" />
    <!-- Merged from file: SECOND_MANIFEST -->
    <provider
        android:name="com.google.android.apps.testapp"
        android:authorities="com.google.android.apps.testapp.com.google.android.apps.testapp"
        android:exported="false" />
    <!-- Merged from file: SECOND_MANIFEST -->
    <provider
        android:name="com.google.android.apps.testapp.PlaceHolderProviderName"
        android:authorities="PlaceHolderProviderAuthorities.com.google.android.apps.testapp"
        android:exported="false" />
    <!-- Merged from file: SECOND_MANIFEST -->
    <receiver android:name="com.google.android.apps.testapp2.ConnectivityReceiver"
        android:enabled="false" >
      <intent-filter>
        <action android:name="android.net.conn.CONNECTIVITY_CHANGE" />
      </intent-filter>
    </receiver>
    <!-- Merged from file: SECOND_MANIFEST -->
    <service android:name="com.google.android.apps.testapp2.TestService">
      <meta-data android:name="param" android:value="value"/>
    </service>
    <!-- Merged from file: SECOND_MANIFEST -->
    <service android:name="com.google.android.apps.testapp2.nfcevent.NfcEventService"/>
    <!-- Merged from file: THIRD_MANIFEST -->
    <activity android:label="@string/app_name" android:name="com.google.android.apps.testapp3.ui.home.HomeActivity">
    <intent-filter>
      <action android:name="android.intent.action.MAIN"/>
      <category android:name="android.intent.category.LAUNCHER"/>
    </intent-filter>
    </activity>
    <!-- Merged from file: THIRD_MANIFEST -->
    <activity android:name="com.google.android.apps.testapp3.TestActivity"/>
    <!-- Merged from file: THIRD_MANIFEST -->
    <receiver android:enabled="true"
        android:name="com.google.android.apps.testapp3.ConnectivityReceiver">
      <intent-filter>
        <action android:name="android.net.conn.CONNECTIVITY_CHANGER"/>
      </intent-filter>
    </receiver>
    <!-- Merged from file: THIRD_MANIFEST -->
    <service android:name="com.google.android.apps.testapp3.TestService"/>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity-alias android:name="com.google.android.apps.testapp2.BarFoo"
        android:targetActivity="com.google.android.apps.testapp2.FooBar"/>
  </application>
</manifest>
"""


ALIAS_MANIFEST = """<?xml version='1.0' encoding='utf-8'?>
<manifest
    xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.android.apps.testapp"
    android:versionCode="70"
    android:versionName="1.0">
  <uses-sdk android:minSdkVersion="10"/>
  <uses-feature android:name="android.hardware.nfc" android:required="true" />
  <uses-permission android:name="android.permission.ACCESS_COARSE_LOCATION" />
  <application
      android:icon="@drawable/icon"
      android:name="com.google.android.apps.testapp.TestApplication"
      android:theme="@style/Theme.Test"
      android:label="@string/app_name">
    <activity-alias android:name="com.google.foo.should.not.be.first"
        android:targetActivity=".entrypoint.EntryPointActivityGroup"/>
    <!--  START LIBRARIES (Maintain Alphabetic order) -->
    <!-- NFC extras -->
    <uses-library android:name="com.google.android.nfc_extras" android:required="false"/>
    <!--  END LIBRARIES -->
    <!--  START ACTIVITIES (Maintain Alphabetic order) -->
    <!-- Entry point activity - navigation and title bar. -->
    <activity
        android:name=".entrypoint.EntryPointActivityGroup"
        android:screenOrientation="portrait"
        android:launchMode="singleTop"/>
    <activity android:name=".ui.topup.TopUpActivity" />
    <service android:name=".nfcevent.NfcEventService" />
    <receiver
         android:name="com.receiver.TestReceiver"
         android:process="@string/receiver_service_name">
      <!-- Receive the actual message -->
       <intent-filter>
         <action
             android:name="android.intent.action.USER_PRESENT"/>
       </intent-filter>
    </receiver>
    <provider
        android:name=".dataaccess.persistence.ContentProvider"
        android:authorities="com.google.android.apps.testapp"
        android:exported="false" />
  </application>
</manifest>
"""


# This case exists when a library manifest relies on
# dependent manifests to provide required elements, i.e. a <application>
INVALID_MERGER_MANIFEST = """
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.android.invalid"
    android:versionCode="9100000"
    android:versionName="9.1.0.0x">
  <uses-sdk android:minSdkVersion="14" android:targetSdkVersion="21" />
</manifest>
"""


INVALID_MERGEE_MANIFEST = """
<manifest xmlns:android="http://schemas.android.com/apk/res/android"
    package="com.google.android.invalid"
    android:versionCode="9100000"
    android:versionName="9.1.0.0x">
  <uses-sdk android:minSdkVersion="14" android:targetSdkVersion="21" />
  <permission android:name="com.google.android.apps.foo.C2D_MESSAGE" android:protectionLevel="signature"/>
  <uses-feature android:name="android.hardware.nfc" android:required="true"/>
</manifest>
"""

VALID_MANIFEST = """
<manifest
    android:versionCode="9100000"
    android:versionName="9.1.0.0x"
    package="com.google.android.invalid"
    xmlns:android="http://schemas.android.com/apk/res/android">
  <!-- *** WARNING *** DO NOT EDIT! THIS IS GENERATED MANIFEST BY MERGE_MANIFEST TOOL.
  Merger manifest:
    INVALID_MANIFEST
  Mergee manifests:
    SECOND_MANIFEST
  -->
  <uses-sdk android:minSdkVersion="14" android:targetSdkVersion="21"/>
  <!-- Merged from file: SECOND_MANIFEST -->
  <permission android:name="com.google.android.apps.foo.C2D_MESSAGE" android:protectionLevel="signature"/>
  <!-- Merged from file: SECOND_MANIFEST -->
  <uses-feature android:name="android.hardware.nfc" android:required="true"/>
  <application
      android:backupAgent="com.google.android.apps.testapp2.FooBar"
      android:icon="@drawable/icon"
      android:label="@string/app_name"
      android:name="com.google.android.apps.testapp.TestApplication2"
      android:theme="@style/Theme.Test2">
    <activity
        android:name="com.google.android.apps.testapp2.TestActivity2"/>
    <activity
        android:name="com.google.android.apps.testapp2.PreviewActivity"/>
    <activity android:excludeFromRecents="true"
        android:name="com.google.android.apps.testapp2.ShowTextActivity"/>
    <activity android:excludeFromRecents="true"
        android:name="com.google.android.apps.testapp2.ShowStringListActivity"
        android:parentActivityName="com.google.android.apps.testapp2.ui.home.HomeActivity">
    </activity>
    <service
        android:name="com.google.android.apps.testapp2.TestService">
      <meta-data android:name="param"
          android:value="value"/>
    </service>
    <service
        android:name="com.google.android.apps.testapp2.nfcevent.NfcEventService"/>
    <receiver
        android:enabled="false"
        android:name="com.google.android.apps.testapp2.ConnectivityReceiver">
      <intent-filter>
        <action android:name="android.net.conn.CONNECTIVITY_CHANGE"/>
      </intent-filter>
    </receiver>
    <provider android:name="some.package.with.inner.class$AnInnerClass"/>
    <provider
        android:authorities="com.google.android.invalid.com.google.android.invalid"
        android:exported="false" android:name="com.google.android.invalid"/>
    <provider android:authorities="PlaceHolderProviderAuthorities.com.google.android.invalid"
        android:exported="false"
        android:name="com.google.android.invalid.PlaceHolderProviderName"/>
    <activity android:name="activityPrefix.com.google.android.invalid.activitySuffix">
      <intent-filter>
        <action android:name="actionPrefix.com.google.android.invalid.actionSuffix"/>
      </intent-filter>
    </activity>
    <!-- Merged from file: SECOND_MANIFEST -->
    <activity android:label="@string/app_name"
        android:name="com.google.android.apps.testapp2.ui.home.HomeActivity">
      <intent-filter>
        <action android:name="android.intent.action.MAIN"/>
        <category android:name="android.intent.category.LAUNCHER"/>
      </intent-filter>
    </activity>
    <activity-alias
        android:name="com.google.android.apps.testapp2.BarFoo"
        android:targetActivity="com.google.android.apps.testapp2.FooBar"/>
  </application>
</manifest>
"""


def Reformat(string):
  """Reformat for comparison."""
  string = re.compile(r'^[ \t]*\n?', re.MULTILINE).sub('', string)
  return string


class MergeManifestsTest(unittest.TestCase):
  """Unit tests for the MergeManifest class."""

  def testMerge(self):
    self.maxDiff = None
    merger = merge_manifests.MergeManifests(
        (FIRST_MANIFEST, 'FIRST_MANIFEST'),
        [(SECOND_MANIFEST, 'SECOND_MANIFEST'),
         (THIRD_MANIFEST, 'THIRD_MANIFEST')],
        ['android.permission.READ_LOGS'])
    result = merger.Merge()
    expected = xml.dom.minidom.parseString(MANUALLY_MERGED).toprettyxml()
    self.assertEquals(Reformat(expected), Reformat(result))

  def testReformat(self):
    text = '  a\n  b\n\n\n \t c'
    expected = 'a\nb\nc'
    self.assertEquals(expected, Reformat(text))

  def testValidateAndWarnPermissions(self):
    permissions = ['android.permission.VIBRATE', 'android.permission.LAUGH']
    warnings = merge_manifests._ValidateAndWarnPermissions(permissions)
    self.assertTrue('android.permission.VIBRATE' not in warnings)
    self.assertTrue('android.permission.LAUGH' in warnings)

  def testExcludeAllPermissions(self):
    merger = merge_manifests.MergeManifests(
        (FIRST_MANIFEST, 'FIRST_MANIFEST'),
        [(SECOND_MANIFEST, 'SECOND_MANIFEST'),
         (THIRD_MANIFEST, 'THIRD_MANIFEST')],
        ['all'])
    result = merger.Merge()
    self.assertFalse('android.permission.READ_LOGS' in result)
    self.assertFalse('android.permission.INTERNET' in result)
    self.assertTrue('android.permission.ACCESS_COARSE_LOCATION' in result)

  def testUndefinedArgumentPlaceholder(self):
    bad_manifest = SECOND_MANIFEST.replace(
        '${packageName}', '${unknownPlaceHolder}')
    merger = merge_manifests.MergeManifests(
        (FIRST_MANIFEST, 'FIRST_MANIFEST'),
        [(bad_manifest, 'invalidManifest'),
         (THIRD_MANIFEST, 'THIRD_MANIFEST')])
    try:
      merger.Merge()
      self.fail('merging manifests with unknown placeholders didn\'t fail')
    except merge_manifests.UndefinedPlaceholderException:
      pass

  def testActivityAliasesAreAlwaysLast(self):
    merger = merge_manifests.MergeManifests(
        (FIRST_MANIFEST, 'FIRST_MANIFEST'),
        [(SECOND_MANIFEST, 'SECOND_MANIFEST'),
         (ALIAS_MANIFEST, 'THIRD_MANIFEST')],
        ['all'])
    result = merger.Merge()
    last_occurence_of_activity = result.rfind('<activity ')
    first_occurence_of_alias = result.find('<activity-alias ')
    self.assertLess(last_occurence_of_activity, first_occurence_of_alias,
                    msg='First activity-alias is not after the last activity!')

  def testMergeToCreateValidManifest(self):
    self.maxDiff = None
    merger = merge_manifests.MergeManifests(
        (INVALID_MERGER_MANIFEST, 'INVALID_MANIFEST'),
        [(SECOND_MANIFEST, 'SECOND_MANIFEST')],
        ['all'])
    result = merger.Merge()
    expected = xml.dom.minidom.parseString(VALID_MANIFEST).toprettyxml()
    self.assertEquals(Reformat(expected), Reformat(result))

  def testMergeWithNoApplication(self):
    merger = merge_manifests.MergeManifests(
        (INVALID_MERGER_MANIFEST, 'INVALID_MERGER_MANIFEST'),
        [(INVALID_MERGEE_MANIFEST, 'INVALID_MERGEE_MANIFEST')],
        ['all'])
    merger.Merge()

if __name__ == '__main__':
  unittest.main()

