# Lint as: python3
# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""Testing for archive."""

import copy
import os
import tarfile
import unittest

from tools.mini_tar import mini_tar


class TarFileWriterTest(unittest.TestCase):
  """Testing for TarFileWriter class."""

  def assertTarFileContent(self, tar, content):
    """Assert that tarfile contains exactly the entry described by `content`.

    Args:
      tar: the path to the TAR file to test.
      content: an array describing the expected content of the TAR file. Each
               entry in that list should be a dictionary where each field is a
               field to test in the corresponding TarInfo. For testing the
               presence of a file "x", then the entry could simply be
               `{"name": "x"}`, the missing field will be ignored. To match the
               content of a file entry, use the key "data".
    """
    with tarfile.open(tar, "r:") as f:
      i = 0
      for current in f:
        error_msg = "Extraneous file at end of archive %s: %s" % (tar,
                                                                  current.name)
        self.assertLess(i, len(content), error_msg)
        for k, v in content[i].items():
          if k == "data":
            value = f.extractfile(current).read()
          else:
            value = getattr(current, k)
          error_msg = " ".join([
              "Value `%s` for key `%s` of file" % (value, k),
              "%s in archive %s does" % (current.name, tar),
              "not match expected value `%s`" % v
          ])
          self.assertEqual(value, v, error_msg)
        i += 1
      if i < len(content):
        self.fail("Missing file %s in archive %s" % (content[i], tar))

  def setUp(self):
    super(TarFileWriterTest, self).setUp()
    self.tempfile = os.path.join(os.environ["TEST_TMPDIR"], "test.tar")

  def tearDown(self):
    super(TarFileWriterTest, self).tearDown()
    if os.path.exists(self.tempfile):
      os.remove(self.tempfile)

  def test_empty_tar_file(self):
    with mini_tar.TarFileWriter(self.tempfile):
      pass
    self.assertTarFileContent(self.tempfile, [])

  def test_default_mtime_not_provided(self):
    with mini_tar.TarFileWriter(self.tempfile) as f:
      self.assertEqual(f.default_mtime, 0)

  def test_default_mtime_provided(self):
    with mini_tar.TarFileWriter(self.tempfile, default_mtime=1234) as f:
      self.assertEqual(f.default_mtime, 1234)

  def test_portable_mtime(self):
    with mini_tar.TarFileWriter(self.tempfile, default_mtime="portable") as f:
      self.assertEqual(f.default_mtime, 946684800)

  def test_files_with_dots(self):
    with mini_tar.TarFileWriter(self.tempfile) as f:
      f.add_file_and_parents("a")
      f.add_file_and_parents("b/.c")
      f.add_file_and_parents("..d")
      f.add_file_and_parents(".e")
    content = [
        {
            "name": "a"
        },
        {
            "name": "b"
        },
        {
            "name": "b/.c"
        },
        {
            "name": "..d"
        },
        {
            "name": ".e"
        },
    ]
    self.assertTarFileContent(self.tempfile, content)

  def test_add_parents(self):
    with mini_tar.TarFileWriter(self.tempfile) as f:
      f.add_parents("a/b/c/d/file")
      f.add_file_and_parents("a/b/foo")
      f.add_parents("a/b/e/file")
    content = [
        {
            "name": "a",
            "mode": 0o755
        },
        {
            "name": "a/b",
            "mode": 0o755
        },
        {
            "name": "a/b/c",
            "mode": 0o755
        },
        {
            "name": "a/b/c/d",
            "mode": 0o755
        },
        {
            "name": "a/b/foo",
            "mode": 0o644
        },
        {
            "name": "a/b/e",
            "mode": 0o755
        },
    ]
    self.assertTarFileContent(self.tempfile, content)

  def test_adding_tree(self):
    content = [
        {
            "name": "./a",
            "mode": 0o750
        },
        {
            "name": "./a/b",
            "data": b"ab",
            "mode": 0o640
        },
        {
            "name": "./a/c",
            "mode": 0o750
        },
        {
            "name": "./a/c/d",
            "data": b"acd",
            "mode": 0o640
        },
    ]
    tempdir = os.path.join(os.environ["TEST_TMPDIR"], "test_dir")
    # Iterate over the `content` array to create the directory
    # structure it describes.
    for c in content:
      if "data" in c:
        p = os.path.join(tempdir, c["name"])
        os.makedirs(os.path.dirname(p))
        with open(p, "wb") as f:
          f.write(c["data"])
    with mini_tar.TarFileWriter(self.tempfile) as f:
      f.add_file_at_dest(in_path=tempdir, dest_path=".", mode=0o640)
    self.assertTarFileContent(self.tempfile, content)

    # Try it again, but re-rooted
    with mini_tar.TarFileWriter(self.tempfile, root_directory="foo") as f:
      f.add_file_at_dest(in_path=tempdir, dest_path="x", mode=0o640)
    n_content = [
        {
            "name": "foo",
            "mode": 0o755
        },
        {
            "name": "foo/x",
            "mode": 0o750
        },
    ]
    for c in content:
      nc = copy.copy(c)
      nc["name"] = "foo/x/" + c["name"][2:]
      n_content.append(nc)
    self.assertTarFileContent(self.tempfile, n_content)


if __name__ == "__main__":
  unittest.main()
