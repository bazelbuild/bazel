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

import sys
from xml.etree import ElementTree

from third_party.py import gflags


gflags.DEFINE_string("main_manifest", None, "The main manifest of the app")
gflags.DEFINE_string("split_manifest", None, "The output manifest")
gflags.DEFINE_string("override_package", None,
                     "The Android package. Override the one specified in the "
                     "input manifest")
gflags.DEFINE_string("split", None, "The name of the split")
gflags.DEFINE_boolean("hascode", False, "Whether this split .apk has dexes")

FLAGS = gflags.FLAGS

MANIFEST_TEMPLATE = """<?xml version="1.0" encoding="utf-8"?>
<manifest
    xmlns:android="http://schemas.android.com/apk/res/android"
    android:versionCode="%(version_code)s"
    android:versionName="%(version_name)s"
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
      "version_code": version_code,
      "version_name": version_name,
      "package": package,
      "split": split,
      "hascode": str(hascode).lower()
  }


def main():
  split_manifest = BuildSplitManifest(
      file(FLAGS.main_manifest).read(),
      FLAGS.override_package,
      FLAGS.split,
      FLAGS.hascode)

  with file(FLAGS.split_manifest, "w") as output_xml:
    output_xml.write(split_manifest)


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
