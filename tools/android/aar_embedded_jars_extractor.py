# pylint: disable=g-direct-third-party-import
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

"""A tool for extracting all jar files from an AAR.

An AAR may contain JARs at /classes.jar and /libs/*.jar. This tool extracts all
of the jars and creates a param file for singlejar to merge them into one jar.
"""

import re
import sys
import zipfile

from third_party.py import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string("input_aar", None, "Input AAR")
gflags.MarkFlagAsRequired("input_aar")
gflags.DEFINE_string(
    "output_singlejar_param_file", None, "Output parameter file for singlejar")
gflags.MarkFlagAsRequired("output_singlejar_param_file")
gflags.DEFINE_string("output_dir", None, "Output directory to extract jars in")
gflags.MarkFlagAsRequired("output_dir")


def ExtractEmbeddedJars(aar, singlejar_param_file, output_dir):
  jar_pattern = re.compile("^(classes|libs/.+)\\.jar$")
  singlejar_param_file.write("--exclude_build_data\n")
  for name in aar.namelist():
    if jar_pattern.match(name):
      singlejar_param_file.write("--sources\n")
      singlejar_param_file.write(output_dir + "/" + name + "\n")
      aar.extract(name, output_dir)


def main():
  with zipfile.ZipFile(FLAGS.input_aar, "r") as aar:
    with open(FLAGS.output_singlejar_param_file, "wb") as singlejar_param_file:
      ExtractEmbeddedJars(aar, singlejar_param_file, FLAGS.output_dir)

if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
