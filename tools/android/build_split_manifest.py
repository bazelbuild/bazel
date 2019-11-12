# Lint as: python2, python3
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

from xml.etree import ElementTree

# Do not edit this line. Copybara replaces it with PY2 migration helper.
from absl import app
from absl import flags

flags.DEFINE_string("main_manifest", None, "The main manifest of the app")
flags.DEFINE_string("split_manifest", None, "The output manifest")
flags.DEFINE_string(
    "override_package", None,
    "The Android package. Override the one specified in the "
    "input manifest")
flags.DEFINE_string("split", None, "The name of the split")
flags.DEFINE_boolean("hascode", False, "Whether this split .apk has dexes")

FLAGS = flags.FLAGS

MANIFEST_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<manifest
    xmlns:android="http://schemas.android.com/apk/res/android"
    %(version_code_attribute)s
    %(version_name_attribute)s
    package="%(package)s"
    split="%(split)s">
  <application android:hasCode="%(hascode)s">
  </application>
</manifest>
"""


def BuildSplitManifest(main_manifest, override_package, split, hascode):
  """Builds a split manifest based on the manifest of the main APK.

  Args:
    main_manifest: the XML manifest of the main APK as a string
    override_package: if not None, override the package in the main manifest
    split: the name of the split as a string
    hascode: if this split APK will contain .dex files

  Returns:
    The XML split manifest as a string

  Raises:
    Exception if something goes wrong.
  """

  manifest = ElementTree.fromstring(main_manifest)
  android_namespace_prefix = "{http://schemas.android.com/apk/res/android}"

  if override_package:
    package = override_package
  else:
    package = manifest.get("package")

  version_code = manifest.get(android_namespace_prefix + "versionCode")
  version_name = manifest.get(android_namespace_prefix + "versionName")

  return MANIFEST_TEMPLATE % {
      "version_code_attribute":
          'android:versionCode="%s"' % version_code if version_code else "",
      "version_name_attribute":
          'android:versionName="%s"' % version_name if version_name else "",
      "package": package,
      "split": split,
      "hascode": str(hascode).lower()
  }


def main(unused_argv):
  split_manifest = BuildSplitManifest(
      open(FLAGS.main_manifest, "rb").read(), FLAGS.override_package,
      FLAGS.split, FLAGS.hascode)

  with open(FLAGS.split_manifest, "wb") as output_xml:
    output_xml.write(split_manifest.encode("utf-8"))


if __name__ == "__main__":
  app.run(main)
