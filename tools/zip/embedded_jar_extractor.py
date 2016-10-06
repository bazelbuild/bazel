# Copyright 2016 The Bazel Authors. All rights reserved.
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

"""A tool for extracting a jar file from an archive and failing gracefully.

If the jar file is present within the archive, it is extracted into the output
directory. If not, an empty jar is created in the output directory.
"""

import os
import sys
import zipfile

from third_party.py import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string("input_archive", None, "Input archive")
gflags.MarkFlagAsRequired("input_archive")
gflags.DEFINE_string("filename", None, "Filename of JAR to extract")
gflags.MarkFlagAsRequired("filename")
gflags.DEFINE_string("output_dir", None, "Output directory")
gflags.MarkFlagAsRequired("output_dir")


def ExtractEmbeddedJar(input_archive, filename, output_dir):
  with zipfile.ZipFile(input_archive, "r") as archive:
    if filename in archive.namelist():
      archive.extract(filename, output_dir)
    else:
      with zipfile.ZipFile(os.path.join(output_dir, filename), "w") as jar:
        # All jar files must contain META-INF/MANIFEST.MF.
        jar.writestr("META-INF/MANIFEST.MF", ("Manifest-Version: 1.0\n"
                                              "Created-By: Bazel\n"))


def main():
  ExtractEmbeddedJar(FLAGS.input_archive, FLAGS.filename, FLAGS.output_dir)

if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
