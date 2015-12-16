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

"""Unit tests for stubify_incremental_install."""

import os
import unittest
import zipfile

from tools.android import incremental_install
from third_party.py import mock


class MockAdb(object):
  """Mocks the Android ADB binary."""

  def __init__(self):
    # Map of file name -> contents.
    self.files = {}
    self.split_apks = set()
    self._error = None
    self.package_timestamp = None
    self._last_package_timestamp = 1
    self.shell_cmdlns = []
    self.abi = "armeabi-v7a"

  def Exec(self, args):
    if self._error:
      error_info, arg = self._error  # pylint: disable=unpacking-non-sequence
      if not arg or arg in args:
        return self._CreatePopenMock(*error_info)

    returncode = 0
    stdout = ""
    stderr = ""
    cmd = args[1]
    if cmd == "push":
      # "/test/adb push local remote"
      with open(args[2]) as f:
        content = f.read()
      self.files[args[3]] = content
    elif cmd == "pull":
      # "/test/adb pull remote local"
      remote = args[2]
      local = args[3]
      content = self.files.get(remote)
      if content is not None:
        with open(local, "w") as f:
          f.write(content)
      else:
        returncode = 1
        stderr = "remote object '%s' does not exist\n" % remote
    elif cmd == "install":
      self.package_timestamp = self._last_package_timestamp
      self._last_package_timestamp += 1
      return self._CreatePopenMock(0, "Success", "")
    elif cmd == "install-multiple":
      if args[3] == "-p":
        with open(args[5]) as f:
          content = f.read()
        self.split_apks.add(content)
      else:
        self.package_timestamp = self._last_package_timestamp
        self._last_package_timestamp += 1
      return self._CreatePopenMock(0, "Success", "")
    elif cmd == "uninstall":
      self._CreatePopenMock(0, "Success", "")
      self.split_apks = set()
      self.package_timestamp = None
    elif cmd == "shell":
      # "/test/adb shell ..."
      # mkdir, rm, am (application manager), or monkey
      shell_cmdln = args[2]
      self.shell_cmdlns.append(shell_cmdln)
      if shell_cmdln.startswith(("mkdir", "am", "monkey", "input")):
        pass
      elif shell_cmdln.startswith("dumpsys package "):
        if self.package_timestamp is not None:
          timestamp = "firstInstallTime=%s" % self.package_timestamp
        else:
          timestamp = ""
        return self._CreatePopenMock(0, timestamp, "")
      elif shell_cmdln.startswith("rm"):
        file_path = shell_cmdln.split()[2]
        self.files.pop(file_path, None)
      elif shell_cmdln.startswith("getprop ro.product.cpu.abi"):
        return self._CreatePopenMock(0, self.abi, "")
      else:
        raise Exception("Unknown shell command line: %s" % shell_cmdln)
    # Return a mock subprocess.Popen object
    return self._CreatePopenMock(returncode, stdout, stderr)

  def _CreatePopenMock(self, returncode, stdout, stderr):
    return mock.Mock(
        returncode=returncode, communicate=lambda: (stdout, stderr))

  def SetError(self, returncode, stdout, stderr, for_arg=None):
    self._error = ((returncode, stdout, stderr), for_arg)

  def SetAbi(self, abi):
    self.abi = abi


class IncrementalInstallTest(unittest.TestCase):
  """Unit tests for incremental install."""

  _DEXMANIFEST = "dexmanifest.txt"
  _ADB_PATH = "/test/adb"
  _OUTPUT_MARKER = "full_deploy_marker"
  _APK = "myapp_incremental.apk"
  _RESOURCE_APK = "incremental.ap_"
  _STUB_DATAFILE = "stub_application_data.txt"
  _OLD_APP_PACKGE = "old.app.package"
  _APP_PACKAGE = "new.app.package"
  _EXEC_ROOT = "."

  def setUp(self):
    os.chdir(os.environ["TEST_TMPDIR"])

    self._mock_adb = MockAdb()

    # Write the stub datafile which contains the package name of the app.
    with open(self._STUB_DATAFILE, "w") as f:
      f.write("\n".join([self._OLD_APP_PACKGE, self._APP_PACKAGE]))

    # Write the local resource apk file.
    with open(self._RESOURCE_APK, "w") as f:
      f.write("resource apk")

    # Mock out subprocess.Popen to use our mock adb.
    self._popen_patch = mock.patch.object(incremental_install, "subprocess")
    self._popen = self._popen_patch.start().Popen
    self._popen.side_effect = lambda args, **kwargs: self._mock_adb.Exec(args)

  def tearDown(self):
    self._popen_patch.stop()

  def _CreateZip(self, name="zip1", *files):
    if not files:
      files = [("zp1", "content1"), ("zp2", "content2")]
    with zipfile.ZipFile(name, "w") as z:
      for f, content in files:
        z.writestr(f, content)

  def _CreateLocalManifest(self, *lines):
    content = "\n".join(lines)
    with open(self._DEXMANIFEST, "w") as f:
      f.write(content)
    return content

  def _CreateRemoteManifest(self, *lines):
    self._PutDeviceFile("dex/manifest", "\n".join(lines))

  def _GetDeviceAppPath(self, f):
    return os.path.join(
        incremental_install.DEVICE_DIRECTORY, self._APP_PACKAGE, f)

  def _GetDeviceFile(self, f):
    return self._mock_adb.files[self._GetDeviceAppPath(f)]

  def _PutDeviceFile(self, f, content):
    self._mock_adb.files[self._GetDeviceAppPath(f)] = content

  def _DeleteDeviceFile(self, f):
    self._mock_adb.files.pop(self._GetDeviceAppPath(f), None)

  def _CallIncrementalInstall(self, incremental, native_libs=None,
                              split_main_apk=None, split_apks=None,
                              start_type="no"):
    if split_main_apk:
      apk = split_main_apk
    elif incremental:
      apk = None
    else:
      apk = self._APK

    incremental_install.IncrementalInstall(
        adb_path=self._ADB_PATH,
        execroot=self._EXEC_ROOT,
        stub_datafile=self._STUB_DATAFILE,
        dexmanifest=self._DEXMANIFEST,
        apk=apk,
        resource_apk=self._RESOURCE_APK,
        split_main_apk=split_main_apk,
        split_apks=split_apks,
        native_libs=native_libs,
        output_marker=self._OUTPUT_MARKER,
        adb_jobs=1,
        start_type=start_type,
        user_home_dir="/home/root")

  def testUploadToPristineDevice(self):
    self._CreateZip()

    with open("dex1", "w") as f:
      f.write("content3")

    manifest = self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zp2 ip2 0",
        "dex1 - ip3 0")

    self._CallIncrementalInstall(incremental=False)

    resources_checksum_path = self._GetDeviceAppPath("resources_checksum")
    self.assertTrue(resources_checksum_path in self._mock_adb.files)
    self.assertEquals(manifest, self._GetDeviceFile("dex/manifest"))
    self.assertEquals("content1", self._GetDeviceFile("dex/ip1"))
    self.assertEquals("content2", self._GetDeviceFile("dex/ip2"))
    self.assertEquals("content3", self._GetDeviceFile("dex/ip3"))
    self.assertEquals("resource apk", self._GetDeviceFile("resources.ap_"))

  def testSplitInstallToPristineDevice(self):
    with open("split1", "w") as f:
      f.write("split_content1")

    with open("main", "w") as f:
      f.write("main_Content")

    self._CallIncrementalInstall(
        incremental=False, split_main_apk="main", split_apks=["split1"])
    self.assertEquals(set(["split_content1"]), self._mock_adb.split_apks)

  def testSplitInstallUnchanged(self):
    with open("split1", "w") as f:
      f.write("split_content1")

    with open("main", "w") as f:
      f.write("main_Content")

    self._CallIncrementalInstall(
        incremental=False, split_main_apk="main", split_apks=["split1"])
    self.assertEquals(set(["split_content1"]), self._mock_adb.split_apks)
    self._mock_adb.split_apks = set()
    self._CallIncrementalInstall(
        incremental=False, split_main_apk="main", split_apks=["split1"])
    self.assertEquals(set([]), self._mock_adb.split_apks)

  def testSplitInstallChanges(self):
    with open("split1", "w") as f:
      f.write("split_content1")

    with open("main", "w") as f:
      f.write("main_Content")

    self._CallIncrementalInstall(
        incremental=False, split_main_apk="main", split_apks=["split1"])
    self.assertEquals(set(["split_content1"]), self._mock_adb.split_apks)

    with open("split1", "w") as f:
      f.write("split_content2")
    self._mock_adb.split_apks = set()
    self._CallIncrementalInstall(
        incremental=False, split_main_apk="main", split_apks=["split1"])
    self.assertEquals(set(["split_content2"]), self._mock_adb.split_apks)

  def testMissingNativeManifestWithIncrementalInstall(self):
    self._CreateZip()
    with open("liba.so", "w") as f:
      f.write("liba_1")

    # Upload a library to the device.
    native_libs = ["armeabi-v7a:liba.so"]
    self._CallIncrementalInstall(incremental=False, native_libs=native_libs)
    self.assertEquals("liba_1", self._GetDeviceFile("native/liba.so"))

    # Delete the manifest, overwrite the library and check that even an
    # incremental install straightens things out.
    self._PutDeviceFile("native/liba.so", "GARBAGE")
    self._CallIncrementalInstall(incremental=False, native_libs=native_libs)
    self.assertEquals("liba_1", self._GetDeviceFile("native/liba.so"))

  def testNonIncrementalInstallOverwritesNativeLibs(self):
    self._CreateZip()
    with open("liba.so", "w") as f:
      f.write("liba_1")

    # Upload a library to the device.
    native_libs = ["armeabi-v7a:liba.so"]
    self._CallIncrementalInstall(incremental=False, native_libs=native_libs)
    self.assertEquals("liba_1", self._GetDeviceFile("native/liba.so"))

    # Change a library on the device. Incremental install should not replace the
    # changed file, because it only checks the manifest.
    self._PutDeviceFile("native/liba.so", "GARBAGE")
    self._CallIncrementalInstall(incremental=True, native_libs=native_libs)
    self.assertEquals("GARBAGE", self._GetDeviceFile("native/liba.so"))

    # However, a full install should overwrite it.
    self._CallIncrementalInstall(incremental=False, native_libs=native_libs)
    self.assertEquals("liba_1", self._GetDeviceFile("native/liba.so"))

  def testNativeAbiCompatibility(self):
    self._CreateZip()
    with open("liba.so", "w") as f:
      f.write("liba")

    native_libs = ["armeabi:liba.so"]
    self._mock_adb.SetAbi("arm64-v8a")
    self._CallIncrementalInstall(incremental=False, native_libs=native_libs)
    self.assertEquals("liba", self._GetDeviceFile("native/liba.so"))

  def testUploadNativeLibs(self):
    self._CreateZip()
    with open("liba.so", "w") as f:
      f.write("liba_1")
    with open("libb.so", "w") as f:
      f.write("libb_1")

    native_libs = ["armeabi-v7a:liba.so", "armeabi-v7a:libb.so"]
    self._CallIncrementalInstall(incremental=False, native_libs=native_libs)
    self.assertEquals("liba_1", self._GetDeviceFile("native/liba.so"))
    self.assertEquals("libb_1", self._GetDeviceFile("native/libb.so"))

    # Change a library
    with open("libb.so", "w") as f:
      f.write("libb_2")
    self._CallIncrementalInstall(incremental=True, native_libs=native_libs)
    self.assertEquals("libb_2", self._GetDeviceFile("native/libb.so"))

    # Delete a library
    self._CallIncrementalInstall(
        incremental=True, native_libs=["armeabi-v7a:liba.so"])
    self.assertFalse(
        self._GetDeviceAppPath("native/libb.so") in self._mock_adb.files)

    # Add the deleted library back
    self._CallIncrementalInstall(incremental=True, native_libs=native_libs)
    self.assertEquals("libb_2", self._GetDeviceFile("native/libb.so"))

  def testUploadWithOneChangedFile(self):
    # Existing manifest from a previous install.
    self._CreateRemoteManifest(
        "zip1 zp1 ip1 0",
        "zip1 zp2 ip2 1")

    # Existing files from a previous install.
    self._PutDeviceFile("dex/ip1", "old content1")
    self._PutDeviceFile("dex/ip2", "old content2")
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateZip()

    # Updated dex manifest.
    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zp2 ip2 2")

    self._CallIncrementalInstall(incremental=True)

    # This is a bit of a dishonest test: the local content for "ip1" is
    # "content1" and the remote content for it is "old content1", but
    # the checksums for that file are the same in the local and remote manifest.
    # We just want to make sure that only one file was updated, so to
    # distinguish that we force the local and remote content to be different but
    # keep the checksum the same.
    self.assertEquals("old content1", self._GetDeviceFile("dex/ip1"))
    self.assertEquals("content2", self._GetDeviceFile("dex/ip2"))

  def testFullUploadWithOneChangedFile(self):

    # Existing manifest from a previous install.
    self._CreateRemoteManifest(
        "zip1 zp1 ip1 0",
        "zip1 zp2 ip2 1")

    self._PutDeviceFile("dex/ip1", "old content1")
    self._PutDeviceFile("dex/ip2", "old content2")
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateZip()

    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zp2 ip2 2")

    self._CallIncrementalInstall(incremental=False)

    # Even though the checksums for ip1 were the same, the file still got
    # updated. This is a bit of a dishonest test because the local and remote
    # content for ip1 were different, but their checksums were the same.
    self.assertEquals("content1", self._GetDeviceFile("dex/ip1"))
    self.assertEquals("content2", self._GetDeviceFile("dex/ip2"))

  def testUploadWithNewFile(self):

    self._CreateRemoteManifest("zip1 zp1 ip1 0")
    self._PutDeviceFile("dex/ip1", "content1")
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zp2 ip2 1")

    self._CreateZip()

    self._CallIncrementalInstall(incremental=True)

    self.assertEquals("content1", self._GetDeviceFile("dex/ip1"))
    self.assertEquals("content2", self._GetDeviceFile("dex/ip2"))

  def testDeletesFile(self):

    self._CreateRemoteManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1")
    self._PutDeviceFile("dex/ip1", "content1")
    self._PutDeviceFile("dex/ip2", "content2")
    self._PutDeviceFile("install_timestamp", "1")
    self._mock_adb.package_timestamp = "1"

    self._CreateZip("zip1", ("zp1", "content1"))
    self._CreateLocalManifest("zip1 zp1 ip1 0")

    self.assertTrue(self._GetDeviceAppPath("dex/ip2") in self._mock_adb.files)
    self._CallIncrementalInstall(incremental=True)
    self.assertFalse(self._GetDeviceAppPath("dex/ip2") in self._mock_adb.files)

  def testNothingToUpdate(self):
    self._CreateRemoteManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")
    self._PutDeviceFile("dex/ip1", "content1")
    self._PutDeviceFile("dex/ip2", "content2")
    self._PutDeviceFile("dex/ip3", "content3")
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateZip()
    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")

    self._CallIncrementalInstall(incremental=True)
    self.assertEquals("content1", self._GetDeviceFile("dex/ip1"))
    self.assertEquals("content2", self._GetDeviceFile("dex/ip2"))
    self.assertEquals("content3", self._GetDeviceFile("dex/ip3"))

  def testNoResourcesToUpdate(self):
    self._CreateRemoteManifest("zip1 zp1 ip1 0")
    self._PutDeviceFile("dex/ip1", "content1")
    # The local file is actually "resource apk", but the checksum on the device
    # for the resources file is set to be the same as the checksum for the
    # local file so that we can confirm that it was not updated.
    self._PutDeviceFile("resources.ap_", "resources")
    self._PutDeviceFile("resources_checksum",
                        incremental_install.Checksum(self._RESOURCE_APK))
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateZip()
    self._CreateLocalManifest("zip1 zp1 ip1 0")

    self._CallIncrementalInstall(incremental=True)
    self.assertEquals("resources", self._GetDeviceFile("resources.ap_"))

  def testUpdateResources(self):
    self._CreateRemoteManifest("zip1 zp1 ip1 0")
    self._PutDeviceFile("dex/ip1", "content1")
    self._PutDeviceFile("resources.ap_", "resources")
    self._PutDeviceFile("resources_checksum", "checksum")
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateZip()
    self._CreateLocalManifest("zip1 zp1 ip1 0")

    self._CallIncrementalInstall(incremental=True)
    self.assertEquals("resource apk", self._GetDeviceFile("resources.ap_"))

  def testNoDevice(self):
    self._mock_adb.SetError(1, "", "device not found")
    try:
      self._CallIncrementalInstall(incremental=True)
      self.fail("Should have quit if there is no device")
    except SystemExit as e:
      # make sure it's the right SystemExit reason
      self.assertTrue("Device not found" in str(e))

  def testUnauthorizedDevice(self):
    self._mock_adb.SetError(1, "", "device unauthorized. Please check the "
                            "confirmation dialog on your device")
    try:
      self._CallIncrementalInstall(incremental=True)
      self.fail("Should have quit if the device is unauthorized.")
    except SystemExit as e:
      # make sure it's the right SystemExit reason
      self.assertTrue("Device unauthorized." in str(e))

  def testInstallFailure(self):
    self._mock_adb.SetError(0, "Failure", "INSTALL_FAILED", for_arg="install")
    self._CreateZip()
    self._CreateLocalManifest("zip1 zp1 ip1 0")
    try:
      self._CallIncrementalInstall(incremental=False)
      self.fail("Should have quit if the install failed.")
    except SystemExit as e:
      # make sure it's the right SystemExit reason
      self.assertTrue("Failure" in str(e))

  def testStartCold(self):
    # Based on testUploadToPristineDevice
    self._CreateZip()

    with open("dex1", "w") as f:
      f.write("content3")

    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zp2 ip2 0",
        "dex1 - ip3 0")

    self._CallIncrementalInstall(incremental=False, start_type="cold")

    self.assertTrue(("monkey -p %s -c android.intent.category.LAUNCHER 1" %
                     self._APP_PACKAGE) in self._mock_adb.shell_cmdlns)

  def testColdStop(self):
    self._CreateRemoteManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")
    self._PutDeviceFile("dex/ip1", "content1")
    self._PutDeviceFile("dex/ip2", "content2")
    self._PutDeviceFile("dex/ip3", "content3")
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateZip()
    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")
    self._CallIncrementalInstall(incremental=True, start_type="cold")

    stop_cmd = "am force-stop %s" % self._APP_PACKAGE
    self.assertTrue(stop_cmd in self._mock_adb.shell_cmdlns)

  def testWarmStop(self):
    self._CreateRemoteManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")
    self._PutDeviceFile("dex/ip1", "content1")
    self._PutDeviceFile("dex/ip2", "content2")
    self._PutDeviceFile("dex/ip3", "content3")
    self._PutDeviceFile("install_timestamp", "0")
    self._mock_adb.package_timestamp = "0"

    self._CreateZip()
    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")
    self._CallIncrementalInstall(incremental=True, start_type="warm")

    background_cmd = "input keyevent KEYCODE_APP_SWITCH"
    stop_cmd = "am kill %s" % self._APP_PACKAGE
    self.assertTrue(background_cmd in self._mock_adb.shell_cmdlns)
    self.assertTrue(stop_cmd in self._mock_adb.shell_cmdlns)

  def testMultipleDevicesError(self):
    errors = [
        "more than one device and emulator",
        "more than one device",
        "more than one emulator",
    ]
    for error in errors:
      self._mock_adb.SetError(1, "", error)
      try:
        self._CallIncrementalInstall(incremental=True)
        self.fail("Should have quit if there were multiple devices.")
      except SystemExit as e:
        # make sure it's the right SystemExit reason
        self.assertTrue("Try specifying a device serial" in str(e))

  def testIncrementalInstallOnPristineDevice(self):
    self._CreateZip()
    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")

    try:
      self._CallIncrementalInstall(incremental=True)
      self.fail("Should have quit for incremental install on pristine device")
    except SystemExit:
      pass

  def testIncrementalInstallWithWrongInstallTimestamp(self):
    self._CreateRemoteManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")
    self._PutDeviceFile("dex/ip1", "content1")
    self._PutDeviceFile("dex/ip2", "content2")
    self._PutDeviceFile("dex/ip3", "content3")
    self._mock_adb.package_timestamp = "WRONG"

    self._CreateZip()
    self._CreateLocalManifest(
        "zip1 zp1 ip1 0",
        "zip1 zip2 ip2 1",
        "dex1 - ip3 0")

    try:
      self._CallIncrementalInstall(incremental=True)
      self.fail("Should have quit if install timestamp is wrong")
    except SystemExit:
      pass

  def testSdkTooOld(self):
    self._mock_adb.SetError(
        0, "INSTALL_FAILED_OLDER_SDK", "", for_arg="install")
    self._CreateZip()
    self._CreateLocalManifest("zip1 zp1 ip1 0")
    try:
      self._CallIncrementalInstall(incremental=False)
      self.fail("Should have quit if the SDK is too old.")
    except SystemExit as e:
      # make sure it's the right SystemExit reason
      self.assertTrue("minSdkVersion" in str(e))


if __name__ == "__main__":
  unittest.main()
