# pylint: disable=g-direct-third-party-import
# Copyright 2017 The Bazel Authors. All rights reserved.
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

"""A tool for extracting resource files from an AAR.

An AAR may contain resources under the /res directory. This tool extracts all
of the resources into a directory. If no resources exist, it creates an
empty.xml file that defines no resources.

In the future, this script may be extended to also extract assets.
"""

import os
import sys
import zipfile

from tools.android import junction
from third_party.py import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string("input_aar", None, "Input AAR")
gflags.MarkFlagAsRequired("input_aar")
gflags.DEFINE_string("output_res_dir", None, "Output resources directory")
gflags.MarkFlagAsRequired("output_res_dir")


def ExtractResources(aar, output_res_dir):
  """Extract resource from an `aar` file to the `output_res_dir` directory."""
  aar_contains_no_resources = True
  output_res_dir_abs = os.path.normpath(
      os.path.join(os.getcwd(), output_res_dir))
  for name in aar.namelist():
    if name.startswith("res/"):
      fullpath = os.path.normpath(os.path.join(output_res_dir_abs, name))
      if os.name == "nt" and len(fullpath) >= 260:  # MAX_PATH in <windows.h>
        with junction.TempJunction(os.path.dirname(fullpath)) as juncpath:
          shortpath = os.path.join(juncpath, os.path.basename(fullpath))
          aar.extract(name, shortpath)
      else:
        aar.extract(name, output_res_dir)
      aar_contains_no_resources = False
  if aar_contains_no_resources:
    empty_xml_filename = output_res_dir + "/res/values/empty.xml"
    os.makedirs(os.path.dirname(empty_xml_filename))
    with open(empty_xml_filename, "wb") as empty_xml:
      empty_xml.write("<resources/>")


def main():
  with zipfile.ZipFile(FLAGS.input_aar, "r") as aar:
    ExtractResources(aar, FLAGS.output_res_dir)

if __name__ == "__main__":
  FLAGS(sys.argv)
  main()
