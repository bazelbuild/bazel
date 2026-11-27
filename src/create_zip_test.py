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
"""Tests for create_zip."""

import os
import stat
import tempfile
import unittest
import zipfile

from src.create_zip import main
from src.create_zip import create_zip


class CreateZipTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))

  def _path(self, name):
    return os.path.join(self._tmp, name)

  def _make_zip(self, name, entries):
    """Creates a zip from a dict of name -> content or (content, mode)."""
    path = self._path(name)
    with zipfile.ZipFile(path, "w") as zf:
      for entry_name, value in entries.items():
        content, mode = value if isinstance(value, tuple) else (value, 0o644)
        info = zipfile.ZipInfo(entry_name, (2020, 1, 2, 3, 4, 6))
        info.create_system = 3
        if entry_name.endswith("/"):
          info.external_attr = ((stat.S_IFDIR | mode) << 16) | 0x10
        else:
          info.external_attr = (stat.S_IFREG | mode) << 16
        zf.writestr(info, content)
    return path

  def _read_zip(self, path):
    with zipfile.ZipFile(path) as zf:
      return {
          i.filename: None if i.is_dir() else zf.read(i).decode()
          for i in zf.infolist()
      }

  def _names(self, path):
    with zipfile.ZipFile(path) as zf:
      return zf.namelist()

  def testNoPrefix(self):
    a = self._make_zip("a.zip", {"dir1/a1.cc": "a1", "a2.cc": "a2"})
    b = self._make_zip("b.zip", {"dir2/b1.cc": "b1", "b2.cc": "b2"})
    out = self._path("out.zip")
    create_zip("--merge", "-", out, [a, b])
    self.assertEqual(
        self._read_zip(out),
        {
            "a2.cc": "a2",
            "b2.cc": "b2",
            "dir1/": None,
            "dir1/a1.cc": "a1",
            "dir2/": None,
            "dir2/b1.cc": "b1",
        },
    )

  def testPrefix(self):
    a = self._make_zip("a.zip", {"dir1/a1.cc": "a1", "a2.cc": "a2"})
    b = self._make_zip("b.zip", {"dir2/b1.cc": "b1", "b2.cc": "b2"})
    out = self._path("out.zip")
    create_zip("--merge", "src/main/cpp", out, [a, b])
    # Like `zip -r src/main/cpp`, parents of the prefix get no entries.
    self.assertEqual(
        self._read_zip(out),
        {
            "src/main/cpp/": None,
            "src/main/cpp/a2.cc": "a2",
            "src/main/cpp/b2.cc": "b2",
            "src/main/cpp/dir1/": None,
            "src/main/cpp/dir1/a1.cc": "a1",
            "src/main/cpp/dir2/": None,
            "src/main/cpp/dir2/b1.cc": "b1",
        },
    )

  def testOutputIsSorted(self):
    a = self._make_zip("a.zip", {"z": "z", "a/b": "b", "m": "m"})
    out = self._path("out.zip")
    create_zip("--merge", "p", out, [a])
    self.assertEqual(self._names(out), ["p/", "p/a/", "p/a/b", "p/m", "p/z"])

  def testLastInputWins(self):
    a = self._make_zip("a.zip", {"f": "from a", "only_a": "a"})
    b = self._make_zip("b.zip", {"f": "from b"})
    out = self._path("out.zip")
    create_zip("--merge", "-", out, [a, b])
    self.assertEqual(self._read_zip(out), {"f": "from b", "only_a": "a"})

  def testExplicitDirectoryEntriesAreKept(self):
    a = self._make_zip("a.zip", {"empty/": ("", 0o700), "d/": "", "d/f": "f"})
    out = self._path("out.zip")
    create_zip("--merge", "-", out, [a])
    self.assertEqual(
        self._read_zip(out), {"d/": None, "d/f": "f", "empty/": None}
    )
    with zipfile.ZipFile(out) as zf:
      self.assertEqual(
          stat.S_IMODE(zf.getinfo("empty/").external_attr >> 16), 0o700
      )

  def testPermissionsAndTimestampsArePreserved(self):
    a = self._make_zip("a.zip", {"exe": ("x", 0o755), "data": ("d", 0o600)})
    out = self._path("out.zip")
    create_zip("--merge", "-", out, [a])
    with zipfile.ZipFile(out) as zf:
      exe = zf.getinfo("exe")
      data = zf.getinfo("data")
      self.assertEqual(exe.external_attr >> 16, stat.S_IFREG | 0o755)
      self.assertEqual(data.external_attr >> 16, stat.S_IFREG | 0o600)
      self.assertEqual(exe.date_time, (2020, 1, 2, 3, 4, 6))
      self.assertEqual(exe.compress_type, zipfile.ZIP_DEFLATED)

  def testNonUnixEntriesGetDefaultPermissions(self):
    path = self._path("a.zip")
    with zipfile.ZipFile(path, "w") as zf:
      info = zipfile.ZipInfo("d/f")
      info.create_system = 0  # MS-DOS
      zf.writestr(info, "f")
    out = self._path("out.zip")
    create_zip("--merge", "-", out, [path])
    with zipfile.ZipFile(out) as zf:
      self.assertEqual(
          zf.getinfo("d/f").external_attr >> 16, stat.S_IFREG | 0o644
      )
      self.assertEqual(
          zf.getinfo("d/").external_attr >> 16, stat.S_IFDIR | 0o755
      )

  def testIsDeterministic(self):
    a = self._make_zip("a.zip", {"d/f": "f", "g": "g"})
    out1 = self._path("out1.zip")
    out2 = self._path("out2.zip")
    create_zip("--merge", "p", out1, [a])
    create_zip("--merge", "p", out2, [a])
    with open(out1, "rb") as f1, open(out2, "rb") as f2:
      self.assertEqual(f1.read(), f2.read())

  def testOverwritesExistingOutput(self):
    a = self._make_zip("a.zip", {"f": "f"})
    out = self._path("out.zip")
    with open(out, "w") as f:
      f.write("garbage")
    create_zip("--merge", "-", out, [a])
    self.assertEqual(self._read_zip(out), {"f": "f"})
    self.assertEqual(sorted(os.listdir(self._tmp)), ["a.zip", "out.zip"])

  def testNoInputs(self):
    out = self._path("out.zip")
    create_zip("--merge", "-", out, [])
    self.assertEqual(self._read_zip(out), {})
    create_zip("--merge", "p", out, [])
    self.assertEqual(self._read_zip(out), {"p/": None})

  def testUnknownMode(self):
    out = self._path("out.zip")
    with self.assertRaisesRegex(ValueError, "Unknown mode: --unknown"):
      create_zip("--unknown", "-", out, [])
    self.assertFalse(os.path.exists(out))

  def testMain(self):
    a = self._make_zip("a.zip", {"f": "f"})
    out = self._path("out.zip")
    self.assertEqual(main(["create_zip", "--merge", "java_tools", out, a]), 0)
    self.assertEqual(
        self._read_zip(out), {"java_tools/": None, "java_tools/f": "f"}
    )
    self.assertEqual(main(["create_zip", "--merge", "-"]), 1)
    self.assertEqual(main(["create_zip", "java_tools", out, a]), 1)
    self.assertEqual(main(["create_zip", "--unknown", "-", out, a]), 1)


if __name__ == "__main__":
  unittest.main()
