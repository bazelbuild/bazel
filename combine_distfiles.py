# pylint: disable=g-bad-file-header
# pylint: disable=g-direct-third-party-import
#
# Copyright 2017 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Creates the Bazel source distribution archive."""

import contextlib
import os.path
import sys
import zipfile

from src.create_embedded_tools_lib import copy_tar_to_zip
from src.create_embedded_tools_lib import copy_zip_to_zip


def main():
  output_zip = os.path.join(os.getcwd(), sys.argv[1])
  input_files = sorted(sys.argv[2:])

  # Copy all the input_files into output_zip.
  # Adding contextlib.closing to be python 2.6 (for centos 6.7) compatible
  with contextlib.closing(
      zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED)) as output_zip:

    def _normalize(path):
      return path[2:] if path.startswith("./") else path

    for input_file in input_files:
      if input_file.endswith(".tar"):
        copy_tar_to_zip(output_zip, input_file, _normalize)
      elif input_file.endswith(".zip"):
        copy_zip_to_zip(output_zip, input_file, _normalize)
      else:
        raise Exception("unknown archive type \"%s\"" % input_file)


if __name__ == "__main__":
  main()
