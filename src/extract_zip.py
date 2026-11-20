# Copyright 2026 The Bazel Authors. All rights reserved.
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
"""Extracts a zip file without depending on `unzip`.

Usage:
  extract_zip.py archive output_dir
      Extracts all entries of archive into output_dir, like
      `unzip -q -o archive -d output_dir`.
  extract_zip.py --print archive name
      Writes the content of the entry name to stdout, like
      `unzip -p archive name`.

Unix file permissions, symlinks and timestamps of the entries are preserved.
Entries whose path would escape output_dir are an error.
"""

import os
import shutil
import stat
import sys
import time
import zipfile

_UNIX = 3


def _unix_mode(info):
  """Returns the Unix mode of info, or 0 if it has none."""
  if info.create_system == _UNIX:
    return info.external_attr >> 16
  return 0


def _target_path(output_dir, name):
  """Returns the path of the entry name below output_dir."""
  path = os.path.normpath(os.path.join(output_dir, name.lstrip("/")))
  root = os.path.normpath(output_dir)
  if os.path.commonpath([root, path]) != root:
    raise ValueError("Entry outside of the output directory: %s" % name)
  return path


def _remove(path):
  if os.path.islink(path) or os.path.isfile(path):
    os.remove(path)


def extract_zip(archive, output_dir):
  """Extracts all entries of archive into output_dir."""
  os.makedirs(output_dir, exist_ok=True)
  # (path, mode, date_time) of the extracted entries, to set their attributes
  # once all files are written.
  extracted = []
  with zipfile.ZipFile(archive, "r") as zf:
    for info in zf.infolist():
      path = _target_path(output_dir, info.filename)
      if path == os.path.normpath(output_dir):
        continue
      mode = _unix_mode(info)
      if info.is_dir() or stat.S_ISDIR(mode):
        os.makedirs(path, exist_ok=True)
      elif stat.S_ISLNK(mode):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _remove(path)
        os.symlink(zf.read(info).decode("utf-8"), path)
        continue
      else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _remove(path)
        with zf.open(info) as src, open(path, "wb") as dst:
          shutil.copyfileobj(src, dst)
      extracted.append((path, stat.S_IMODE(mode), info.date_time))

  # Directories are set last and deepest first, so that writing their content
  # doesn't change their timestamp and read-only directories can be filled.
  for path, mode, date_time in sorted(extracted, key=lambda e: e[0],
                                      reverse=True):
    if mode:
      os.chmod(path, mode)
    # Like `unzip`, interpret the timestamp as local time.
    mtime = time.mktime(date_time + (0, 0, -1))
    os.utime(path, (mtime, mtime))


def print_entry(archive, name, out):
  """Writes the content of the entry name of archive to out."""
  with zipfile.ZipFile(archive, "r") as zf:
    out.write(zf.read(name))


def main(argv):
  if len(argv) == 4 and argv[1] == "--print":
    try:
      print_entry(argv[2], argv[3], sys.stdout.buffer)
    except KeyError:
      sys.stderr.write("%s: no entry %s in %s\n" % (argv[0], argv[3], argv[2]))
      return 1
    return 0
  if len(argv) == 3 and not argv[1].startswith("-"):
    extract_zip(argv[1], argv[2])
    return 0
  sys.stderr.write(
      "Usage: %s archive output_dir\n"
      "       %s --print archive name\n" % (argv[0], argv[0])
  )
  return 1


if __name__ == "__main__":
  sys.exit(main(sys.argv))
