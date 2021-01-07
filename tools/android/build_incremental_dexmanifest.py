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
from queue import Queue
import shutil
import sys
import tempfile
from threading import Thread
import zipfile


class DexmanifestBuilder(object):
  """Implementation of the dex manifest builder."""

  def __init__(self):
    self.manifest_lines = []
    self.dir_counter = 1
    self.output_dex_counter = 1
    self.checksums = set()
    self.tmpdir = None
    self.queue = Queue()
    self.threads_list = list()

  def __enter__(self):
    self.tmpdir = tempfile.mkdtemp()
    return self

  def __exit__(self, unused_type, unused_value, unused_traceback):
    shutil.rmtree(self.tmpdir, True)

  def Checksum(self, filename, input_dex_or_zip, zippath):
    """Compute the SHA-256 checksum of a file.

    This method could be invoked concurrently.

    Therefore we need to include other metadata like input_dex_or_zip to
    keep the context.
    """
    h = hashlib.sha256()
    with open(filename, "rb") as f:
      while True:
        data = f.read(65536)
        if not data:
          break

        h.update(data)

    return h.hexdigest(), input_dex_or_zip, zippath

  def AddDexes(self, dex_metadata_list):
    """Adds all dex file together to the output.

    Sort the result to make sure the dexes order are always the same given
    the same input.
    Args:
      dex_metadata_list: A list of [fs_checksum, input_dex_or_zip, zippath],
        where fs_checksum is the SHA-256 checksum for dex file, input_dex_or_zip
        is the input file written to the manifest, zippath is the zip path
        written to the manifest or None if the input file is not a .zip.

    Returns:
      None.
    """
    dex_metadata_list_sorted = sorted(
        dex_metadata_list, key=lambda x: (x[1], x[2]))
    for dex_metadata in dex_metadata_list_sorted:
      fs_checksum, input_dex_or_zip, zippath = dex_metadata[0], dex_metadata[
          1], dex_metadata[2]
      if fs_checksum in self.checksums:
        return
      self.checksums.add(fs_checksum)
      zip_dex = "incremental_classes%d.dex" % self.output_dex_counter
      self.output_dex_counter += 1
      self.manifest_lines.append(
          "%s %s %s %s" %
          (input_dex_or_zip, zippath if zippath else "-", zip_dex, fs_checksum))

  def ComputeChecksumConcurrently(self, input_dex_or_zip, zippath, dex):
    """Call Checksum concurrently to improve build performance when an app contains multiple dex files."""
    t = Thread(target=lambda q, arg1, arg2, arg3: q.put(self.Checksum(arg1, arg2, arg3)), \
      args=(self.queue, dex, input_dex_or_zip, zippath))
    t.start()
    self.threads_list.append(t)

  def Run(self, argv):
    """Creates a dex manifest."""
    if len(argv) < 1:
      raise Exception("At least one argument expected")

    if argv[0][0] == "@":
      if len(argv) != 1:
        raise IOError("A parameter file should be the only argument")
      with open(argv[0][1:]) as param_file:
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
            self.ComputeChecksumConcurrently(input_filename, input_dex_dex,
                                             fs_dex)
      elif input_filename.endswith(".dex"):
        self.ComputeChecksumConcurrently(input_filename, None, input_filename)
    # Collect results from all threads
    for t in self.threads_list:
      t.join()

    results = []
    while not self.queue.empty():
      fs_checksum, input_dex_or_zip, zippath = self.queue.get()
      results.append([fs_checksum, input_dex_or_zip, zippath])
    self.AddDexes(results)

    with open(argv[0], "wb") as manifest:
      manifest.write(("\n".join(self.manifest_lines)).encode("utf-8"))


def main(argv):
  with DexmanifestBuilder() as b:
    b.Run(argv[1:])


if __name__ == "__main__":
  main(sys.argv)
