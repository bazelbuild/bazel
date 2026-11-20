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
"""Tests for extract_zip."""

import io
import os
import stat
import tempfile
import time
import unittest
import zipfile

from src.extract_zip import extract_zip
from src.extract_zip import main
from src.extract_zip import print_entry


class ExtractZipTest(unittest.TestCase):

  def setUp(self):
    super().setUp()
    self._tmp = tempfile.mkdtemp(dir=os.environ.get("TEST_TMPDIR"))
    self._out = os.path.join(self._tmp, "out")

  def _make_zip(self, entries, unix=True):
    """Creates a zip from a list of (name, content, mode, date_time)."""
    path = os.path.join(self._tmp, "in.zip")
    with zipfile.ZipFile(path, "w") as zf:
      for name, content, mode, date_time in entries:
        info = zipfile.ZipInfo(name, date_time)
        if unix:
          info.create_system = 3
          info.external_attr = mode << 16
        else:
          info.create_system = 0
          info.external_attr = 0x10 if name.endswith("/") else 0
        zf.writestr(info, content)
    return path

  def _read(self, name):
    with open(os.path.join(self._out, name), "rb") as f:
      return f.read()

  def testExtractsFilesAndDirectories(self):
    archive = self._make_zip([
        ("a/", b"", stat.S_IFDIR | 0o755, (2020, 1, 1, 0, 0, 0)),
        ("a/b.txt", b"b", stat.S_IFREG | 0o644, (2020, 1, 1, 0, 0, 0)),
        ("c/d/e.txt", b"e", stat.S_IFREG | 0o644, (2020, 1, 1, 0, 0, 0)),
    ])
    extract_zip(archive, self._out)
    self.assertEqual(self._read("a/b.txt"), b"b")
    self.assertEqual(self._read("c/d/e.txt"), b"e")
    self.assertTrue(os.path.isdir(os.path.join(self._out, "a")))

  def testPreservesPermissions(self):
    archive = self._make_zip([
        ("bin/", b"", stat.S_IFDIR | 0o700, (1980, 1, 1, 0, 0, 0)),
        ("bin/tool", b"x", stat.S_IFREG | 0o755, (1980, 1, 1, 0, 0, 0)),
        ("ro.txt", b"r", stat.S_IFREG | 0o444, (1980, 1, 1, 0, 0, 0)),
    ])
    extract_zip(archive, self._out)

    def mode(name):
      return stat.S_IMODE(os.stat(os.path.join(self._out, name)).st_mode)

    self.assertEqual(mode("bin"), 0o700)
    self.assertEqual(mode("bin/tool"), 0o755)
    self.assertEqual(mode("ro.txt"), 0o444)

  def testPreservesTimestamps(self):
    date_time = (2010, 1, 1, 12, 30, 0)
    archive = self._make_zip([
        ("d/", b"", stat.S_IFDIR | 0o755, date_time),
        ("d/f.txt", b"f", stat.S_IFREG | 0o644, date_time),
    ])
    extract_zip(archive, self._out)
    expected = time.mktime(date_time + (0, 0, -1))
    for name in ["d", "d/f.txt"]:
      self.assertEqual(
          os.stat(os.path.join(self._out, name)).st_mtime, expected, name)

  def testExtractsSymlinks(self):
    archive = self._make_zip([
        ("target.txt", b"t", stat.S_IFREG | 0o644, (1980, 1, 1, 0, 0, 0)),
        ("link", b"target.txt", stat.S_IFLNK | 0o777, (1980, 1, 1, 0, 0, 0)),
    ])
    extract_zip(archive, self._out)
    link = os.path.join(self._out, "link")
    self.assertTrue(os.path.islink(link))
    self.assertEqual(os.readlink(link), "target.txt")
    self.assertEqual(self._read("link"), b"t")

  def testNonUnixEntriesGetDefaultPermissions(self):
    archive = self._make_zip(
        [
            ("dir/", b"", 0, (1980, 1, 1, 0, 0, 0)),
            ("dir/f.txt", b"f", 0, (1980, 1, 1, 0, 0, 0)),
        ],
        unix=False,
    )
    extract_zip(archive, self._out)
    path = os.path.join(self._out, "dir/f.txt")
    self.assertEqual(self._read("dir/f.txt"), b"f")
    self.assertTrue(os.access(path, os.R_OK | os.W_OK))
    self.assertTrue(os.path.isdir(os.path.join(self._out, "dir")))

  def testOverwritesExistingFiles(self):
    os.makedirs(self._out)
    existing = os.path.join(self._out, "f.txt")
    with open(existing, "wb") as f:
      f.write(b"old")
    os.chmod(existing, 0o444)
    archive = self._make_zip(
        [("f.txt", b"new", stat.S_IFREG | 0o644, (1980, 1, 1, 0, 0, 0))])
    extract_zip(archive, self._out)
    self.assertEqual(self._read("f.txt"), b"new")

  def testRejectsEntriesOutsideOutputDir(self):
    archive = self._make_zip(
        [("../evil.txt", b"x", stat.S_IFREG | 0o644, (1980, 1, 1, 0, 0, 0))])
    with self.assertRaises(ValueError):
      extract_zip(archive, self._out)
    self.assertFalse(os.path.exists(os.path.join(self._tmp, "evil.txt")))

  def testStripsLeadingSlash(self):
    archive = self._make_zip(
        [("/abs.txt", b"a", stat.S_IFREG | 0o644, (1980, 1, 1, 0, 0, 0))])
    extract_zip(archive, self._out)
    self.assertEqual(self._read("abs.txt"), b"a")

  def testPrintEntry(self):
    archive = self._make_zip([
        ("META-INF/MANIFEST.MF", b"Manifest-Version: 1.0\r\n",
         stat.S_IFREG | 0o644, (1980, 1, 1, 0, 0, 0)),
    ])
    out = io.BytesIO()
    print_entry(archive, "META-INF/MANIFEST.MF", out)
    self.assertEqual(out.getvalue(), b"Manifest-Version: 1.0\r\n")

  def testMainPrintMissingEntryFails(self):
    archive = self._make_zip(
        [("a.txt", b"a", stat.S_IFREG | 0o644, (1980, 1, 1, 0, 0, 0))])
    self.assertEqual(main(["extract_zip", "--print", archive, "b.txt"]), 1)

  def testMainExtracts(self):
    archive = self._make_zip(
        [("a.txt", b"a", stat.S_IFREG | 0o644, (1980, 1, 1, 0, 0, 0))])
    self.assertEqual(main(["extract_zip", archive, self._out]), 0)
    self.assertEqual(self._read("a.txt"), b"a")

  def testMainUsage(self):
    self.assertEqual(main(["extract_zip"]), 1)
    self.assertEqual(main(["extract_zip", "--print", "a.zip"]), 1)
    self.assertEqual(main(["extract_zip", "--bogus", "a.zip"]), 1)


if __name__ == "__main__":
  unittest.main()
