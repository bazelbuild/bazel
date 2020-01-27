# pylint: disable=g-bad-file-header
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
"""Utils to the contents of a tar or zip file into another zip file."""

import contextlib
import os.path
import stat
import tarfile
import zipfile


def is_mode_executable(mode):
  """Returns true if `mode` has any of the executable bits set."""
  return mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH) > 0


def is_executable(path):
  """Returns true if `path` is an executable file/directory."""
  return is_mode_executable(os.stat(path)[stat.ST_MODE])


def copy_tar_to_zip(output_zip, input_file, process_filename=None):
  """Copy a tar file's contents into a zip file.

  This function unpacks every file from `input_file` and puts them into
  `output_zip`. The unpacking is performed in-memory.

  Args:
    output_zip: zipfile.ZipFile; the destination archive
    input_file: string; path to the source tar file
    process_filename: function(str) -> str; optional; for a packed file entry in
      `input_file` it computes the path in `output_zip`
  """
  with tarfile.open(input_file, 'r', errorlevel=2) as tar_file:
    while True:
      tar_entry = tar_file.next()
      if tar_entry is None:
        break
      filename = (process_filename(tar_entry.name)
                  if process_filename else tar_entry.name)
      zipinfo = zipfile.ZipInfo(filename, (1980, 1, 1, 0, 0, 0))
      if tar_entry.isreg():
        if is_mode_executable(tar_entry.mode):
          zipinfo.external_attr = 0o755 << 16
        else:
          zipinfo.external_attr = 0o644 << 16
        zipinfo.compress_type = zipfile.ZIP_DEFLATED
        output_zip.writestr(zipinfo, tar_file.extractfile(tar_entry).read())
      elif tar_entry.issym():
        # 0120000 originally comes from the definition of S_IFLNK and
        # marks a symbolic link in the Zip file format.
        zipinfo.external_attr = 0o120000 << 16
        output_zip.writestr(zipinfo, tar_entry.linkname)
      else:
        # Ignore directories, hard links, special files, ...
        pass


def copy_zip_to_zip(output_zip, input_file, process_filename=None):
  """Copy a zip file's contents into another zip file.

  This function unpacks every file from `input_file` and puts them into
  `output_zip`. The unpacking is performed in-memory.

  Args:
    output_zip: zipfile.ZipFile; the destination archive
    input_file: string; path to the source tar file
    process_filename: function(str) -> str; optional; for a packed file entry in
      `input_file` it computes the path in `output_zip`
  """
  # Adding contextlib.closing to be python 2.6 (for centos 6.7) compatible
  with contextlib.closing(zipfile.ZipFile(input_file, 'r')) as zip_file:
    for zip_entry in zip_file.infolist():
      filename = (process_filename(zip_entry.filename)
                  if process_filename else zip_entry.filename)
      zipinfo = zipfile.ZipInfo(filename, (1980, 1, 1, 0, 0, 0))
      if is_mode_executable(zip_entry.external_attr >> 16 & 0xFFFF):
        zipinfo.external_attr = 0o755 << 16
      else:
        zipinfo.external_attr = 0o644 << 16
      zipinfo.compress_type = zip_entry.compress_type
      output_zip.writestr(zipinfo, zip_file.read(zip_entry))
