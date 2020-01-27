# pylint: disable=g-direct-third-party-import
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

"""Stubifies an AndroidManifest.xml.

Does the following things:
  - Replaces the Application class in an Android manifest with a stub one
  - Resolve string and integer resources to their default values

usage: %s [input manifest] [output manifest] [file for old application class]

Writes the old application class into the file designated by the third argument.
"""

from __future__ import print_function

import sys
from xml.etree import ElementTree

# Do not edit this line. Copybara replaces it with PY2 migration helper.
from absl import app
from absl import flags

flags.DEFINE_string("mode", "mobile_install",
                    "mobile_install or instant_run mode")
flags.DEFINE_string("input_manifest", None, "The input manifest")
flags.DEFINE_string("output_manifest", None, "The output manifest")
flags.DEFINE_string(
    "output_datafile", None, "The output data file that will "
    "be embedded in the final APK")
flags.DEFINE_string(
    "override_package", None,
    "The Android package. Override the one specified in the "
    "input manifest")

FLAGS = flags.FLAGS

ANDROID = "http://schemas.android.com/apk/res/android"
READ_EXTERNAL_STORAGE = "android.permission.READ_EXTERNAL_STORAGE"
MOBILE_INSTALL_STUB_APPLICATION = (
    "com.google.devtools.build.android.incrementaldeployment.StubApplication")
INSTANT_RUN_BOOTSTRAP_APPLICATION = (
    "com.android.tools.fd.runtime.BootstrapApplication")

# This is global state, but apparently that's the best one can to with
# ElementTree :(
ElementTree.register_namespace("android", ANDROID)


class BadManifestException(Exception):
  pass


def StubifyMobileInstall(manifest_string):
  """Does the stubification on an XML string for mobile-install.

  Args:
    manifest_string: the input manifest as a string.
  Returns:
    A tuple of (output manifest, old application class, app package)
  Raises:
    Exception: if something goes wrong
  """
  manifest, application = _ParseManifest(manifest_string)

  old_application = application.get(
      "{%s}name" % ANDROID, "android.app.Application")

  application.set("{%s}name" % ANDROID, MOBILE_INSTALL_STUB_APPLICATION)
  application.attrib.pop("{%s}hasCode" % ANDROID, None)
  read_permission = manifest.findall(
      './uses-permission[@android:name="%s"]' % READ_EXTERNAL_STORAGE,
      namespaces={"android": ANDROID})

  if not read_permission:
    read_permission = ElementTree.Element("uses-permission")
    read_permission.set("{%s}name" % ANDROID, READ_EXTERNAL_STORAGE)
    manifest.insert(0, read_permission)

  new_manifest = ElementTree.tostring(manifest)
  app_package = manifest.get("package")
  if not app_package:
    raise BadManifestException("manifest tag does not have a package specified")
  return (new_manifest, old_application, app_package)


def StubifyInstantRun(manifest_string):
  """Stubifies the manifest for Instant Run.

  Args:
    manifest_string: the input manifest as a string.
  Returns:
    The new manifest as a string.
  Raises:
    Exception: if something goes wrong
  """
  manifest, application = _ParseManifest(manifest_string)
  old_application = application.get("{%s}name" % ANDROID)
  if old_application:
    application.set("name", old_application)
  application.set("{%s}name" % ANDROID, INSTANT_RUN_BOOTSTRAP_APPLICATION)
  return ElementTree.tostring(manifest)


def _ParseManifest(manifest_string):
  """Parses the given manifest xml.

  Args:
    manifest_string: the manifest as a string.
  Returns:
    a tuple of the manifest ElementTree and the application tag.
  Raises:
    BadManifestException: if the manifest is bad.
  """
  manifest = ElementTree.fromstring(manifest_string)
  if manifest.tag != "manifest":
    raise BadManifestException("invalid input manifest")

  app_list = manifest.findall("application")
  if len(app_list) == 1:
    # <application> element is present
    application = app_list[0]
  elif len(app_list) == 0:  # pylint: disable=g-explicit-length-test
    # <application> element is not present
    application = ElementTree.Element("application")
    manifest.insert(0, application)
  else:
    raise BadManifestException("multiple <application> elements present")
  return (manifest, application)


def main(unused_argv):
  if FLAGS.mode == "mobile_install":
    with open(FLAGS.input_manifest, "rb") as input_manifest:
      new_manifest, old_application, app_package = (
          StubifyMobileInstall(input_manifest.read()))

    if FLAGS.override_package:
      app_package = FLAGS.override_package

    with open(FLAGS.output_manifest, "wb") as output_xml:
      output_xml.write(new_manifest)

    with open(FLAGS.output_datafile, "wb") as output_file:
      output_file.write("\n".join([old_application, app_package]).encode())

  elif FLAGS.mode == "instant_run":
    with open(FLAGS.input_manifest, "rb") as input_manifest:
      new_manifest = StubifyInstantRun(input_manifest.read())

    with open(FLAGS.output_manifest, "wb") as output_xml:
      output_xml.write(new_manifest)


if __name__ == "__main__":
  try:
    app.run(main)
  except BadManifestException as e:
    print(e)
    sys.exit(1)
