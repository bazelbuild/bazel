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

"""Removes the resources from a resource APK for incremental deployment.

The reason this utility exists is that the only way we can build a binary
AndroidManifest.xml is by invoking aapt, which builds a whole resource .apk.

Thus, in order to build the AndroidManifest.xml for an incremental .apk, we
invoke aapt, then extract AndroidManifest.xml from its output.
"""

import sys
import zipfile

from third_party.py import gflags


gflags.DEFINE_string("input_resource_apk", None, "The input resource .apk")
gflags.DEFINE_string("output_resource_apk", None, "The output resource .apk")

FLAGS = gflags.FLAGS
HERMETIC_TIMESTAMP = (2001, 1, 1, 0, 0, 0)


def main():
  with zipfile.ZipFile(FLAGS.input_resource_apk) as input_zip:
    with input_zip.open("AndroidManifest.xml") as android_manifest_entry:
      android_manifest = android_manifest_entry.read()

  with zipfile.ZipFile(FLAGS.output_resource_apk, "w") as output_zip:
    # Timestamp is explicitly set so that the resulting zip file is hermetic
    zipinfo = zipfile.ZipInfo(
        filename="AndroidManifest.xml",
        date_time=HERMETIC_TIMESTAMP)
    output_zip.writestr(zipinfo, android_manifest)


if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
