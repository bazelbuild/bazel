# Lint as: python2, python3
# pylint: disable=g-direct-third-party-import
# pylint: disable=g-bad-file-header
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

"""Installs an Android application, possibly in an incremental way."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
from concurrent import futures
import hashlib
import logging
import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

# Do not edit this line. Copybara replaces it with PY2 migration helper.
from absl import app
from absl import flags
import six

flags.DEFINE_string("split_main_apk", None, "The main APK for split install")
flags.DEFINE_multi_string("split_apk", [], "Split APKs to install")
flags.DEFINE_string("dexmanifest", None, "The .dex manifest")
flags.DEFINE_multi_string("native_lib", None, "Native libraries to install")
flags.DEFINE_string("resource_apk", None, "The resource .apk")
flags.DEFINE_string(
    "apk", None, "The app .apk. If not specified, "
    "do incremental deployment")
flags.DEFINE_string("adb", None, "ADB to use")
flags.DEFINE_string("stub_datafile", None, "The stub data file")
flags.DEFINE_string("output_marker", None, "The output marker file")
flags.DEFINE_multi_string("extra_adb_arg", [], "Extra arguments to adb")
flags.DEFINE_string("execroot", ".", "The exec root")
flags.DEFINE_integer(
    "adb_jobs",
    2, "The number of instances of adb to use in parallel to "
    "update files on the device",
    lower_bound=1)
flags.DEFINE_enum(
    "start", "no", ["no", "cold", "warm", "debug"],
    "Whether/how to start the app after installing it. 'cold' "
    "and 'warm' will both cause the app to be started, 'warm' "
    "will start it with previously saved application state, "
    "'debug' will wait for the debugger before a clean start.")
flags.DEFINE_boolean("start_app", False, "Deprecated, use 'start'.")
flags.DEFINE_string("user_home_dir", None, "Path to the user's home directory")
flags.DEFINE_string("flagfile", None,
                    "Path to a file to read additional flags from")

FLAGS = flags.FLAGS

DEVICE_DIRECTORY = "/data/local/tmp/incrementaldeployment"

# Some devices support ABIs other than those reported by getprop. In this case,
# if the most specific ABI is not available in the .apk, we push the more
# general ones.
COMPATIBLE_ABIS = {
    "armeabi-v7a": ["armeabi"],
    "arm64-v8a": ["armeabi-v7a", "armeabi"]
}


class AdbError(Exception):
  """An exception class signaling an error in an adb invocation."""

  def __init__(self, args, returncode, stdout, stderr):
    self.args = args
    self.returncode = returncode
    self.stdout = stdout
    self.stderr = stderr
    details = "\n".join([
        "adb command: %s" % args,
        "return code: %s" % returncode,
        "stdout: %s" % stdout,
        "stderr: %s" % stderr,
    ])
    super(AdbError, self).__init__(details)


class DeviceNotFoundError(Exception):
  """Raised when the device could not be found."""


class MultipleDevicesError(Exception):
  """Raised when > 1 device is attached and no device serial was given."""

  @staticmethod
  def CheckError(s):
    return re.search("more than one (device and emulator|device|emulator)", s)


class DeviceUnauthorizedError(Exception):
  """Raised when the local machine is not authorized to the device."""


class TimestampException(Exception):
  """Raised when there is a problem with timestamp reading/writing."""


class OldSdkException(Exception):
  """Raised when the SDK on the target device is older than the app allows."""


class EnvvarError(Exception):
  """Raised when a required environment variable is not set."""


hostpath = os.path
targetpath = posixpath


class Adb(object):
  """A class to handle interaction with adb."""

  def __init__(self, adb_path, temp_dir, adb_jobs, user_home_dir,
               extra_adb_args):
    self._adb_path = adb_path
    self._temp_dir = temp_dir
    self._user_home_dir = user_home_dir
    self._file_counter = 1
    self._executor = futures.ThreadPoolExecutor(max_workers=adb_jobs)
    self._extra_adb_args = extra_adb_args or []

  def _Exec(self, adb_args):
    """Executes the given adb command + args."""
    args = [self._adb_path] + self._extra_adb_args + adb_args
    # TODO(ahumesky): Because multiple instances of adb are executed in
    # parallel, these debug logging lines will get interleaved.
    logging.debug("Executing: %s", " ".join(args))

    # adb sometimes requires the user's home directory to access things in
    # $HOME/.android (e.g. keys to authorize with the device). To avoid any
    # potential problems with python picking up things in the user's home
    # directory, HOME is not set in the environment around python and is instead
    # passed explicitly as a flag.
    env = {}
    if self._user_home_dir:
      env["HOME"] = self._user_home_dir

    # On Windows, adb requires the SystemRoot environment variable.
    if Adb._IsHostOsWindows():
      value = os.getenv("SYSTEMROOT")
      if not value:
        raise EnvvarError(("The %SYSTEMROOT% environment variable must "
                           "be set or Adb won't work"))
      env["SYSTEMROOT"] = value

    adb = subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env)
    stdout, stderr = adb.communicate()
    stdout = stdout.strip()
    stderr = stderr.strip()
    logging.debug("adb ret: %s", adb.returncode)
    logging.debug("adb out: %s", stdout)
    logging.debug("adb err: %s", stderr)

    # Check these first so that the more specific error gets raised instead of
    # the more generic AdbError.
    stdout = six.ensure_str(stdout)
    stderr = six.ensure_str(stderr)
    if "device not found" in stderr:
      raise DeviceNotFoundError()
    elif "device unauthorized" in stderr:
      raise DeviceUnauthorizedError()
    elif MultipleDevicesError.CheckError(stderr):
      # The error messages are from adb's transport.c, but something adds
      # "error: " to the beginning, so take it off so that we don't end up
      # printing "Error: error: ..."
      raise MultipleDevicesError(re.sub("^error: ", "", stderr))
    elif "INSTALL_FAILED_OLDER_SDK" in stdout:
      raise OldSdkException()

    if adb.returncode != 0:
      raise AdbError(args, adb.returncode, stdout, stderr)

    return adb.returncode, stdout, stderr, args

  def _ExecParallel(self, adb_args):
    return self._executor.submit(self._Exec, adb_args)

  def _CreateLocalFile(self):
    """Returns a path to a temporary local file in the temp directory."""
    local = hostpath.join(self._temp_dir, "adbfile_%d" % self._file_counter)
    self._file_counter += 1
    return local

  def GetInstallTime(self, package):
    """Get the installation time of a package."""
    _, stdout, _, _ = self._Shell("dumpsys package %s" % package)
    match = re.search("firstInstallTime=(.*)$", six.ensure_str(stdout),
                      re.MULTILINE)
    if match:
      return match.group(1)
    else:
      return None

  def GetAbi(self):
    """Returns the ABI the device supports."""
    _, stdout, _, _ = self._Shell("getprop ro.product.cpu.abi")
    return stdout

  def Push(self, local, remote):
    """Invoke 'adb push' in parallel."""
    return self._ExecParallel(["push", local, remote])

  def PushString(self, contents, remote):
    """Push a given string to a given path on the device in parallel."""
    local = self._CreateLocalFile()
    with open(local, "wb") as f:
      f.write(contents.encode("utf-8"))
    return self.Push(local, remote)

  def Pull(self, remote):
    """Invoke 'adb pull'.

    Args:
      remote: The path to the remote file to pull.

    Returns:
      The contents of a file or None if the file didn't exist.
    """
    local = self._CreateLocalFile()
    try:
      self._Exec(["pull", remote, local])
      with open(local, "rb") as f:
        return six.ensure_str(f.read(), "utf-8")
    except (AdbError, IOError):
      return None

  def InstallMultiple(self, apk, pkg=None):
    """Invoke 'adb install-multiple'."""

    pkg_args = ["-p", pkg] if pkg else []
    ret, stdout, stderr, args = self._Exec(
        ["install-multiple", "-r"] + pkg_args + [apk])
    if "FAILED" in stdout or "FAILED" in stderr:
      raise AdbError(args, ret, stdout, stderr)

  def Install(self, apk):
    """Invoke 'adb install'."""
    ret, stdout, stderr, args = self._Exec(["install", "-r", apk])

    # adb install could fail with a message on stdout like this:
    #
    #   pkg: /data/local/tmp/Gmail_dev_sharded_incremental.apk
    #   Failure [INSTALL_PARSE_FAILED_INCONSISTENT_CERTIFICATES]
    #
    # and yet it will still have a return code of 0. At least for the install
    # command, it will print "Success" if it succeeded, so check for that in
    # standard out instead of relying on the return code.
    if "FAILED" in stdout or "FAILED" in stderr:
      raise AdbError(args, ret, stdout, stderr)

  def Uninstall(self, pkg):
    """Invoke 'adb uninstall'."""
    self._Exec(["uninstall", pkg])
    # No error checking. If this fails, we assume that the app was not installed
    # in the first place.

  def Delete(self, remote):
    """Delete the given file (or directory) on the device."""
    self.DeleteMultiple([remote])

  def DeleteMultiple(self, remote_files):
    """Delete the given files (or directories) on the device."""
    files_str = " ".join(remote_files)
    if files_str:
      self._Shell("rm -fr %s" % files_str)

  def Mkdir(self, d):
    """Invokes mkdir with the specified directory on the device."""
    self._Shell("mkdir -p %s" % d)

  def StopApp(self, package):
    """Force stops the app with the given package."""
    self._Shell("am force-stop %s" % package)

  def StopAppAndSaveState(self, package):
    """Stops the app with the given package, saving state for the next run."""
    # 'am kill' will only kill processes in the background, so we must make sure
    # our process is in the background first. We accomplish this by bringing up
    # the app switcher.
    self._Shell("input keyevent KEYCODE_APP_SWITCH")
    self._Shell("am kill %s" % package)

  def StartApp(self, package, start_type):
    """Starts the app with the given package."""
    if start_type == "debug":
      self._Shell("am set-debug-app -w --persistent %s" % package)
    else:
      self._Shell("am clear-debug-app %s" % package)
    self._Shell("monkey -p %s -c android.intent.category.LAUNCHER 1" % package)

  def _Shell(self, cmd):
    """Invoke 'adb shell'."""
    return self._Exec(["shell", cmd])

  @staticmethod
  def _IsHostOsWindows():
    return os.name == "nt"


ManifestEntry = collections.namedtuple(
    "ManifestEntry", ["input_file", "zippath", "installpath", "sha256"])


def ParseManifest(contents):
  """Parses a dexmanifest file.

  Args:
    contents: the contents of the manifest file to be parsed.

  Returns:
    A dict of install path -> ManifestEntry.
  """
  result = {}

  for l in contents.split("\n"):
    entry = ManifestEntry(*(l.strip().split(" ")))
    result[entry.installpath] = entry

  return result


def GetAppPackage(stub_datafile):
  """Returns the app package specified in a stub data file."""
  with open(stub_datafile, "rb") as f:
    return six.ensure_str(f.readlines()[1], "utf-8").strip()


def UploadDexes(adb, execroot, app_dir, temp_dir, dexmanifest, full_install):
  """Uploads dexes to the device so that the state.

  Does the minimum amount of work necessary to make the state of the device
  consistent with what was built.

  Args:
    adb: the Adb instance representing the device to install to
    execroot: the execroot
    app_dir: the directory things should be installed under on the device
    temp_dir: a local temporary directory
    dexmanifest: contents of the dex manifest
    full_install: whether to do a full install

  Returns:
    None.
  """

  # Fetch the manifest on the device
  dex_dir = targetpath.join(app_dir, "dex")
  adb.Mkdir(dex_dir)

  old_manifest = None

  if not full_install:
    logging.info("Fetching dex manifest from device...")
    old_manifest_contents = adb.Pull(targetpath.join(dex_dir, "manifest"))
    if old_manifest_contents:
      old_manifest = ParseManifest(old_manifest_contents)
    else:
      logging.info("Dex manifest not found on device")

  if old_manifest is None:
    # If the manifest is not found, maybe a previous installation attempt
    # was interrupted. Wipe the slate clean. Do this also in case we do a full
    # installation.
    old_manifest = {}
    adb.Delete(targetpath.join(dex_dir, "*"))

  new_manifest = ParseManifest(dexmanifest)
  dexes_to_delete = set(old_manifest) - set(new_manifest)

  # Figure out which dexes to upload: those that are present in the new manifest
  # but not in the old one and those whose checksum was changed
  common_dexes = set(new_manifest).intersection(old_manifest)
  dexes_to_upload = set(d for d in common_dexes
                        if new_manifest[d].sha256 != old_manifest[d].sha256)
  dexes_to_upload.update(set(new_manifest) - set(old_manifest))

  if not dexes_to_delete and not dexes_to_upload:
    # If we have nothing to do, don't bother removing and rewriting the manifest
    logging.info("Application dexes up-to-date")
    return

  # Delete the manifest so that we know how to get back to a consistent state
  # if we are interrupted.
  adb.Delete(targetpath.join(dex_dir, "manifest"))

  # Tuple of (local, remote) files to push to the device.
  files_to_push = []

  # Sort dexes to be uploaded by the zip file they are in so that we only need
  # to open each zip only once.
  dexzips_in_upload = set(new_manifest[d].input_file for d in dexes_to_upload
                          if new_manifest[d].zippath != "-")
  for i, dexzip_name in enumerate(dexzips_in_upload):
    zip_dexes = [
        d for d in dexes_to_upload if new_manifest[d].input_file == dexzip_name]
    dexzip_tempdir = hostpath.join(temp_dir, "dex", str(i))
    with zipfile.ZipFile(hostpath.join(execroot, dexzip_name)) as dexzip:
      for dex in zip_dexes:
        zippath = new_manifest[dex].zippath
        dexzip.extract(zippath, dexzip_tempdir)
        files_to_push.append((hostpath.join(dexzip_tempdir, zippath),
                              targetpath.join(dex_dir, dex)))

  # Now gather all the dexes that are not within a .zip file.
  dexes_to_upload = set(
      d for d in dexes_to_upload if new_manifest[d].zippath == "-")
  for dex in dexes_to_upload:
    files_to_push.append((new_manifest[dex].input_file, targetpath.join(
        dex_dir, dex)))

  num_files = len(dexes_to_delete) + len(files_to_push)
  logging.info("Updating %d dex%s...", num_files, "es" if num_files > 1 else "")

  # Delete the dexes that are not in the new manifest
  adb.DeleteMultiple(targetpath.join(dex_dir, dex) for dex in dexes_to_delete)

  # Upload all the files.
  upload_walltime_start = time.time()
  fs = [adb.Push(local, remote) for local, remote in files_to_push]
  done, not_done = futures.wait(fs, return_when=futures.FIRST_EXCEPTION)
  upload_walltime = time.time() - upload_walltime_start
  logging.debug("Dex upload walltime: %s seconds", upload_walltime)

  # If there is anything in not_done, then some adb call failed and we
  # can cancel the rest.
  if not_done:
    for f in not_done:
      f.cancel()

  # If any adb call resulted in an exception, re-raise it.
  for f in done:
    f.result()

  # If no dex upload failed, upload the manifest. If any upload failed, the
  # exception should have been re-raised above.
  # Call result() to raise the exception if there was one.
  adb.PushString(dexmanifest, targetpath.join(dex_dir, "manifest")).result()


def Checksum(filename):
  """Compute the SHA-256 checksum of a file."""
  h = hashlib.sha256()
  with open(filename, "rb") as f:
    while True:
      data = f.read(65536)
      if not data:
        break

      h.update(data)

  return h.hexdigest()


def UploadResources(adb, resource_apk, app_dir):
  """Uploads resources to the device.

  Args:
    adb: The Adb instance representing the device to install to.
    resource_apk: Path to the resource apk.
    app_dir: The directory things should be installed under on the device.

  Returns:
    None.
  """

  # Compute the checksum of the new resources file
  new_checksum = Checksum(resource_apk)

  # Fetch the checksum of the resources file on the device, if it exists
  device_checksum_file = targetpath.join(app_dir, "resources_checksum")
  old_checksum = adb.Pull(device_checksum_file)
  if old_checksum == new_checksum:
    logging.info("Application resources up-to-date")
    return
  logging.info("Updating application resources...")

  # Remove the checksum file on the device so that if the transfer is
  # interrupted, we know how to get the device back to a consistent state.
  adb.Delete(device_checksum_file)
  adb.Push(resource_apk, targetpath.join(app_dir, "resources.ap_")).result()

  # Write the new checksum to the device.
  adb.PushString(new_checksum, device_checksum_file).result()


def ConvertNativeLibs(args):
  """Converts the --native_libs command line argument to an arch -> libs map."""
  native_libs = {}
  if args is not None:
    for native_lib in args:
      abi, path = six.ensure_str(native_lib).split(":")
      if abi not in native_libs:
        native_libs[abi] = set()

      native_libs[abi].add(path)

  return native_libs


def FindAbi(device_abi, app_abis):
  """Selects which ABI native libs should be installed for."""
  if device_abi in app_abis:
    return device_abi

  if device_abi in COMPATIBLE_ABIS:
    for abi in COMPATIBLE_ABIS[device_abi]:
      if abi in app_abis:
        logging.warn("App does not have native libs for ABI '%s'. Using ABI "
                     "'%s'.", device_abi, abi)
        return abi

  logging.warn("No native libs for device ABI '%s'. App has native libs for "
               "ABIs: %s", device_abi, ", ".join(app_abis))
  return None


def UploadNativeLibs(adb, native_lib_args, app_dir, full_install):
  """Uploads native libraries to the device."""

  native_libs = ConvertNativeLibs(native_lib_args)
  libs = set()
  if native_libs:
    abi = FindAbi(adb.GetAbi(), list(native_libs.keys()))
    if abi:
      libs = native_libs[abi]

  basename_to_path = {}
  install_checksums = {}
  for lib in sorted(libs):
    install_checksums[os.path.basename(lib)] = Checksum(lib)
    basename_to_path[os.path.basename(lib)] = lib

  device_manifest = None
  if not full_install:
    device_manifest = adb.Pull(
        targetpath.join(app_dir, "native", "native_manifest"))

  device_checksums = {}
  if device_manifest is None:
    # If we couldn't fetch the device manifest or if this is a non-incremental
    # install, wipe the slate clean
    adb.Delete(targetpath.join(app_dir, "native"))

    # From Android 28 onwards, `adb push` creates directories with insufficient
    # permissions, resulting in errors when pushing files. `adb shell mkdir`
    # works correctly however, so we create the directory here.
    # See https://github.com/bazelbuild/examples/issues/77 for more information.
    adb.Mkdir(targetpath.join(app_dir, "native"))
  else:
    # Otherwise, parse the manifest. Note that this branch is also taken if the
    # manifest is empty.
    for manifest_line in device_manifest.split("\n"):
      if manifest_line:
        name, checksum = manifest_line.split(" ")
        device_checksums[name] = checksum

  libs_to_delete = set(device_checksums) - set(install_checksums)
  libs_to_upload = set(install_checksums) - set(device_checksums)
  common_libs = set(install_checksums).intersection(set(device_checksums))
  libs_to_upload.update([l for l in common_libs
                         if install_checksums[l] != device_checksums[l]])

  libs_to_push = [(basename_to_path[lib], targetpath.join(
      app_dir, "native", lib)) for lib in libs_to_upload]

  if not libs_to_delete and not libs_to_push and device_manifest is not None:
    logging.info("Native libs up-to-date")
    return

  num_files = len(libs_to_delete) + len(libs_to_push)
  logging.info("Updating %d native lib%s...",
               num_files, "s" if num_files != 1 else "")

  adb.Delete(targetpath.join(app_dir, "native", "native_manifest"))

  if libs_to_delete:
    adb.DeleteMultiple(
        [targetpath.join(app_dir, "native", lib) for lib in libs_to_delete])

  upload_walltime_start = time.time()
  fs = [adb.Push(local, remote) for local, remote in libs_to_push]
  done, not_done = futures.wait(fs, return_when=futures.FIRST_EXCEPTION)
  upload_walltime = time.time() - upload_walltime_start
  logging.debug("Native library upload walltime: %s seconds", upload_walltime)

  # If there is anything in not_done, then some adb call failed and we
  # can cancel the rest.
  if not_done:
    for f in not_done:
      f.cancel()

  # If any adb call resulted in an exception, re-raise it.
  for f in done:
    f.result()

  install_manifest = [
      six.ensure_str(name) + " " + checksum
      for name, checksum in install_checksums.items()
  ]
  adb.PushString("\n".join(install_manifest),
                 targetpath.join(app_dir, "native",
                                 "native_manifest")).result()


def VerifyInstallTimestamp(adb, app_package):
  """Verifies that the app is unchanged since the last mobile-install."""
  expected_timestamp = adb.Pull(
      targetpath.join(DEVICE_DIRECTORY, app_package, "install_timestamp"))
  if not expected_timestamp:
    raise TimestampException(
        "Cannot verify last mobile install. At least one non-incremental "
        "'mobile-install' must precede incremental installs")

  actual_timestamp = adb.GetInstallTime(app_package)
  if actual_timestamp is None:
    raise TimestampException(
        "Package '%s' is not installed on the device. At least one "
        "non-incremental 'mobile-install' must precede incremental "
        "installs." % app_package)

  if actual_timestamp != expected_timestamp:
    raise TimestampException("Installed app '%s' has an unexpected timestamp. "
                             "Did you last install the app in a way other than "
                             "'mobile-install'?" % app_package)


def SplitIncrementalInstall(adb, app_package, execroot, split_main_apk,
                            split_apks):
  """Does incremental installation using split packages."""
  app_dir = targetpath.join(DEVICE_DIRECTORY, app_package)
  device_manifest_path = targetpath.join(app_dir, "split_manifest")
  device_manifest = adb.Pull(device_manifest_path)
  expected_timestamp = adb.Pull(targetpath.join(app_dir, "install_timestamp"))
  actual_timestamp = adb.GetInstallTime(app_package)
  device_checksums = {}
  if device_manifest is not None:
    for manifest_line in device_manifest.split("\n"):
      if manifest_line:
        name, checksum = manifest_line.split(" ")
        device_checksums[name] = checksum

  install_checksums = {}
  install_checksums["__MAIN__"] = Checksum(
      hostpath.join(execroot, split_main_apk))
  for apk in split_apks:
    install_checksums[apk] = Checksum(hostpath.join(execroot, apk))

  reinstall_main = False
  if (device_manifest is None or actual_timestamp is None or
      actual_timestamp != expected_timestamp or
      install_checksums["__MAIN__"] != device_checksums["__MAIN__"] or
      set(device_checksums.keys()) != set(install_checksums.keys())):
    # The main app is not up to date or not present or something happened
    # with the on-device manifest. Start from scratch. Notably, we cannot
    # uninstall a split package, so if the set of packages changes, we also
    # need to do a full reinstall.
    reinstall_main = True
    device_checksums = {}

  apks_to_update = [
      apk for apk in split_apks if
      apk not in device_checksums or
      device_checksums[apk] != install_checksums[apk]]

  if not apks_to_update and not reinstall_main:
    # Nothing to do
    return

  # Delete the device manifest so that if something goes wrong, we do a full
  # reinstall next time
  adb.Delete(device_manifest_path)

  if reinstall_main:
    logging.info("Installing main APK...")
    adb.Uninstall(app_package)
    adb.InstallMultiple(targetpath.join(execroot, split_main_apk))
    adb.PushString(
        adb.GetInstallTime(app_package),
        targetpath.join(app_dir, "install_timestamp")).result()

  logging.info("Reinstalling %s APKs...", len(apks_to_update))

  for apk in apks_to_update:
    adb.InstallMultiple(targetpath.join(execroot, apk), app_package)

  install_manifest = [
      six.ensure_str(name) + " " + checksum
      for name, checksum in install_checksums.items()
  ]
  adb.PushString("\n".join(install_manifest),
                 targetpath.join(app_dir, "split_manifest")).result()


def IncrementalInstall(adb_path,
                       execroot,
                       stub_datafile,
                       output_marker,
                       adb_jobs,
                       start_type,
                       dexmanifest=None,
                       apk=None,
                       native_libs=None,
                       resource_apk=None,
                       split_main_apk=None,
                       split_apks=None,
                       user_home_dir=None,
                       extra_adb_args=None):
  """Performs an incremental install.

  Args:
    adb_path: Path to the adb executable.
    execroot: Exec root.
    stub_datafile: The stub datafile containing the app's package name.
    output_marker: Path to the output marker file.
    adb_jobs: The number of instances of adb to use in parallel.
    start_type: A string describing whether/how to start the app after
                installing it. Can be 'no', 'cold', or 'warm'.
    dexmanifest: Path to the .dex manifest file.
    apk: Path to the .apk file. May be None to perform an incremental install.
    native_libs: Native libraries to install.
    resource_apk: Path to the apk containing the app's resources.
    split_main_apk: the split main .apk if split installation is desired.
    split_apks: the list of split .apks to be installed.
    user_home_dir: Path to the user's home directory.
    extra_adb_args: Extra arguments that will always be passed to adb.
  """
  temp_dir = tempfile.mkdtemp()
  try:
    adb = Adb(adb_path, temp_dir, adb_jobs, user_home_dir, extra_adb_args)
    app_package = GetAppPackage(hostpath.join(execroot, stub_datafile))
    app_dir = targetpath.join(DEVICE_DIRECTORY, app_package)
    if split_main_apk:
      SplitIncrementalInstall(adb, app_package, execroot, split_main_apk,
                              split_apks)
    else:
      if not apk:
        VerifyInstallTimestamp(adb, app_package)

      with open(hostpath.join(execroot, dexmanifest), "rb") as f:
        dexmanifest = six.ensure_str(f.read(), "utf-8")
      UploadDexes(adb, execroot, app_dir, temp_dir, dexmanifest, bool(apk))
      # TODO(ahumesky): UploadDexes waits for all the dexes to be uploaded, and
      # then UploadResources is called. We could instead enqueue everything
      # onto the threadpool so that uploading resources happens sooner.
      UploadResources(adb, hostpath.join(execroot, resource_apk), app_dir)
      UploadNativeLibs(adb, native_libs, app_dir, bool(apk))
      if apk:
        apk_path = targetpath.join(execroot, apk)
        adb.Install(apk_path)
        future = adb.PushString(
            adb.GetInstallTime(app_package),
            targetpath.join(DEVICE_DIRECTORY, app_package, "install_timestamp"))
        future.result()
      else:
        if start_type == "warm":
          adb.StopAppAndSaveState(app_package)
        else:
          adb.StopApp(app_package)

    if start_type in ["cold", "warm", "debug"]:
      logging.info("Starting application %s", app_package)
      adb.StartApp(app_package, start_type)

    with open(output_marker, "wb") as _:
      pass
  except DeviceNotFoundError:
    sys.exit("Error: Device not found")
  except DeviceUnauthorizedError:
    sys.exit("Error: Device unauthorized. Please check the confirmation "
             "dialog on your device.")
  except MultipleDevicesError as e:
    sys.exit("Error: " + str(e) + "\nTry specifying a device serial with "
             "\"bazel mobile-install --adb_arg=-s --adb_arg=$ANDROID_SERIAL\"")
  except OldSdkException as e:
    sys.exit("Error: The device does not support the API level specified in "
             "the application's manifest. Check minSdkVersion in "
             "AndroidManifest.xml")
  except TimestampException as e:
    sys.exit("Error:\n%s" % str(e))
  except AdbError as e:
    sys.exit("Error:\n%s" % str(e))
  finally:
    shutil.rmtree(temp_dir, True)


def main(unused_argv):
  if FLAGS.verbosity == "1":  # 'verbosity' flag is defined in absl.logging
    level = logging.DEBUG
    fmt = "%(levelname)-5s %(asctime)s %(module)s:%(lineno)3d] %(message)s"
  else:
    level = logging.INFO
    fmt = "%(message)s"
  logging.basicConfig(stream=sys.stdout, level=level, format=fmt)

  start_type = FLAGS.start
  if FLAGS.start_app and start_type == "no":
    start_type = "cold"

  IncrementalInstall(
      adb_path=FLAGS.adb,
      adb_jobs=FLAGS.adb_jobs,
      execroot=FLAGS.execroot,
      stub_datafile=FLAGS.stub_datafile,
      output_marker=FLAGS.output_marker,
      start_type=start_type,
      native_libs=FLAGS.native_lib,
      split_main_apk=FLAGS.split_main_apk,
      split_apks=FLAGS.split_apk,
      dexmanifest=FLAGS.dexmanifest,
      apk=FLAGS.apk,
      resource_apk=FLAGS.resource_apk,
      user_home_dir=FLAGS.user_home_dir,
      extra_adb_args=FLAGS.extra_adb_arg)


if __name__ == "__main__":
  FLAGS(sys.argv)
  # process any additional flags in --flagfile
  if FLAGS.flagfile:
    with open(FLAGS.flagfile, "rb") as flagsfile:
      FLAGS.Reset()
      FLAGS(sys.argv + [line.strip() for line in flagsfile.readlines()])

  app.run(main)
