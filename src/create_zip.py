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
"""Creates a zip file with all its entries under a common directory prefix.

Usage: create_zip.py MODE directory_prefix output_zip [inputs]

"-" can be passed as directory_prefix if no top-level directory is required.

MODE is one of:
  --merge: the inputs are zip files whose content is merged.

For example, if we have the following zips and their content:
a.zip:
   dir1/a1.cc
   a2.cc
b.zip:
   dir2/b1.cc
   b2.cc

create_zip.py --merge src/main/cpp my_archive.zip a.zip b.zip
will create the archive my_archive.zip containing:
src/main/cpp/a2.cc
src/main/cpp/b2.cc
src/main/cpp/dir1/a1.cc
src/main/cpp/dir2/b1.cc

create_zip.py --merge - my_archive.zip a.zip b.zip
will create the archive my_archive.zip containing:
a2.cc
b2.cc
dir1/a1.cc
dir2/b1.cc

If the same file is present in several input zips, the last one wins. Unix
file permissions and timestamps of the entries are preserved. A directory entry
is written for the prefix and for every directory below it. The output entries
are sorted by name so that the result is deterministic.
"""

import os
import stat
import sys
import tempfile
import zipfile

# Timestamp used for directory entries that don't exist in any input zip.
_DEFAULT_DATE_TIME = (1980, 1, 1, 0, 0, 0)
_DIR_ATTR = ((stat.S_IFDIR | 0o755) << 16) | 0x10  # 0x10 is MS-DOS directory.
_FILE_ATTR = (stat.S_IFREG | 0o644) << 16
_UNIX = 3


def _join(prefix, name):
  return prefix + "/" + name if prefix else name


def _parent_dirs(name, prefix):
  """Yields all directories containing name, up to and including prefix."""
  parts = name.rstrip("/").split("/")[:-1]
  for i in range(len(parts), 0, -1):
    path = "/".join(parts[:i])
    if prefix and len(path) < len(prefix):
      return
    yield path + "/"


def create_zip(mode, directory_prefix, output, inputs):
  """Creates output from the inputs under directory_prefix.

  mode is one of the modes described in the module docstring.
  """
  if mode != "--merge":
    raise ValueError("Unknown mode: %s" % mode)
  prefix = "" if directory_prefix == "-" else directory_prefix.strip("/")

  # Maps entry name to (ZipInfo, data); data is None for directories.
  entries = {}
  for input_zip in inputs:
    with zipfile.ZipFile(input_zip, "r") as zf:
      for info in zf.infolist():
        name = info.filename.lstrip("/")
        if not name:
          continue
        name = _join(prefix, name)
        data = None if info.is_dir() else zf.read(info)
        entries[name] = (info, data)

  dirs = set([prefix + "/"] if prefix else [])
  for name in entries:
    dirs.update(_parent_dirs(name, prefix))
  for name in dirs:
    entries.setdefault(name, (None, None))

  # Write to a temporary file first so that a failure doesn't leave a
  # partially-written output behind.
  output_dir = os.path.dirname(os.path.abspath(output))
  fd, tmp_output = tempfile.mkstemp(dir=output_dir, suffix=".zip")
  os.close(fd)
  try:
    with zipfile.ZipFile(tmp_output, "w") as out:
      for name in sorted(entries):
        src, data = entries[name]
        info = zipfile.ZipInfo(
            name, src.date_time if src else _DEFAULT_DATE_TIME
        )
        info.create_system = _UNIX
        if data is None:
          if src and src.create_system == _UNIX and src.external_attr >> 16:
            info.external_attr = src.external_attr | 0x10
          else:
            info.external_attr = _DIR_ATTR
          out.writestr(info, b"")
        else:
          if src.create_system == _UNIX and src.external_attr >> 16:
            info.external_attr = src.external_attr
          else:
            info.external_attr = _FILE_ATTR
          info.compress_type = zipfile.ZIP_DEFLATED
          out.writestr(info, data, compresslevel=9)
    os.replace(tmp_output, output)
  except BaseException:
    os.remove(tmp_output)
    raise


def main(argv):
  if len(argv) < 4 or argv[1] != "--merge":
    sys.stderr.write(
        "Usage: %s MODE directory_prefix output_zip [inputs]\n" % argv[0]
    )
    return 1
  create_zip(argv[1], argv[2], argv[3], argv[4:])
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv))
