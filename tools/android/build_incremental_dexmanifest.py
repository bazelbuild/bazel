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

"""Construct a dex manifest from a set of input .dex.zip files.

Usage: %s <output manifest> <input zip file>*
       %s @<params file>

Input files must be either .zip files containing one or more .dex files or
.dex files.

A manifest file is written that contains one line for each input dex in the
following form:

<input zip> <path in input zip> <path in output zip> <MD5 checksum>

or

<input dex> - <path in output zip> <SHA-256 checksum>
"""

import hashlib
import os
import shutil
import sys
import tempfile
import zipfile


class DexmanifestBuilder(object):
  """Implementation of the dex manifest builder."""

  def __init__(self):
    self.manifest_lines = []
    self.dir_counter = 1
    self.output_dex_counter = 1
    self.checksums = set()
    self.tmpdir = None

  def __enter__(self):
    self.tmpdir = tempfile.mkdtemp()
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    shutil.rmtree(self.tmpdir, True)

  def Checksum(self, filename):
    """Compute the SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with file(filename, "r") as f:
      while True:
        data = f.read(65536)
        if not data:
          break

        h.update(data)

    return h.hexdigest()

  def AddDex(self, input_dex_or_zip, zippath, dex):
    """Adds a dex file to the output.

    Args:
      input_dex_or_zip: the input file written to the manifest
      zippath: the zip path written to the manifest or None if the input file
          is not a .zip .
      dex: the dex file to be added

    Returns:
      None.
    """

    fs_checksum = self.Checksum(dex)
    if fs_checksum in self.checksums:
      return

    self.checksums.add(fs_checksum)
    zip_dex = "incremental_classes%d.dex" % self.output_dex_counter
    self.output_dex_counter += 1
    self.manifest_lines.append("%s %s %s %s" %(
        input_dex_or_zip, zippath if zippath else "-", zip_dex, fs_checksum))

  def Run(self, argv):
    """Creates a dex manifest."""
    if len(argv) < 1:
      raise Exception("At least one argument expected")

    if argv[0][0] == "@":
      if len(argv) != 1:
        raise IOError("A parameter file should be the only argument")
      with file(argv[0][1:]) as param_file:
        argv = [a.strip() for a in param_file.readlines()]

    for input_filename in argv[1:]:
      input_filename = input_filename.strip()
      if input_filename.endswith(".zip"):
        with zipfile.ZipFile(input_filename, "r") as input_dex_zip:
          input_dex_dir = os.path.join(self.tmpdir, str(self.dir_counter))
          os.makedirs(input_dex_dir)
          self.dir_counter += 1

          for input_dex_dex in input_dex_zip.namelist():
            if not input_dex_dex.endswith(".dex"):
              continue

            input_dex_zip.extract(input_dex_dex, input_dex_dir)
            fs_dex = input_dex_dir + "/" + input_dex_dex
            self.AddDex(input_filename, input_dex_dex, fs_dex)
      elif input_filename.endswith(".dex"):
        self.AddDex(input_filename, None, input_filename)

    with file(argv[0], "w") as manifest:
      manifest.write("\n".join(self.manifest_lines))


def main(argv):
  with DexmanifestBuilder() as b:
    b.Run(argv[1:])


if __name__ == "__main__":
  main(sys.argv)
