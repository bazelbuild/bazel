# Copyright 2015 Google Inc. All rights reserved.
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

import sys
from xml.etree import ElementTree

from third_party.py import gflags


gflags.DEFINE_string("input_manifest", None, "The input manifest")
gflags.DEFINE_string("output_manifest", None, "The output manifest")
gflags.DEFINE_string("output_datafile", None, "The output data file that will "
                     "be embedded in the final APK")
gflags.DEFINE_string("override_package", None,
                     "The Android package. Override the one specified in the "
                     "input manifest")

FLAGS = gflags.FLAGS

ANDROID = "http://schemas.android.com/apk/res/android"
READ_EXTERNAL_STORAGE = "android.permission.READ_EXTERNAL_STORAGE"
STUB_APPLICATION = (
    "com.google.devtools.build.android.incrementaldeployment.StubApplication")

# This is global state, but apparently that's the best one can to with
# ElementTree :(
ElementTree.register_namespace("android", ANDROID)


class BadManifestException(Exception):
  pass


def Stubify(manifest_string):
  """Does the stubification on an XML string.

  Args:
    manifest_string: the input manifest as a string.
  Returns:
    A tuple of (output manifest, old application class, app package)
  Raises:
    Exception: if something goes wrong
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

  old_application = application.get("{%s}name" % ANDROID)
  if old_application is None:
    old_application = "android.app.Application"

  application.set("{%s}name" % ANDROID, STUB_APPLICATION)

  read_permission = manifest.findall(
      './uses-permission[@android:name="%s"]' % READ_EXTERNAL_STORAGE,
      namespaces={"android": ANDROID})

  if not read_permission:
    read_permission = ElementTree.Element("uses-permission")
    read_permission.set("{%s}name" % ANDROID, READ_EXTERNAL_STORAGE)
    manifest.insert(0, read_permission)

  new_manifest = ElementTree.tostring(manifest)
  app_package = manifest.get("package")
  return (new_manifest, old_application, app_package)


def main():
  with file(FLAGS.input_manifest) as input_manifest:
    new_manifest, old_application, app_package = Stubify(input_manifest.read())

  if FLAGS.override_package:
    app_package = FLAGS.override_package

  with file(FLAGS.output_manifest, "w") as output_xml:
    output_xml.write(new_manifest)

  with file(FLAGS.output_datafile, "w") as output_file:
    output_file.write("\n".join([old_application, app_package]))


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
