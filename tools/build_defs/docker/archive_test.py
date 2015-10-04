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
"""Testing for archive."""

import os
import os.path
import tarfile
import unittest

from tools.build_defs.docker import archive
from tools.build_defs.docker import testenv


class SimpleArFileTest(unittest.TestCase):
  """Testing for SimpleArFile class."""

  def assertArFileContent(self, arfile, content):
    """Assert that arfile contains exactly the entry described by `content`.

    Args:
        arfile: the path to the AR file to test.
        content: an array describing the expected content of the AR file.
            Each entry in that list should be a dictionary where each field
            is a field to test in the corresponding SimpleArFileEntry. For
            testing the presence of a file "x", then the entry could simply
            be `{"filename": "x"}`, the missing field will be ignored.
    """
    with archive.SimpleArFile(arfile) as f:
      current = f.next()
      i = 0
      while current:
        error_msg = "Extraneous file at end of archive %s: %s" % (
            arfile,
            current.filename
            )
        self.assertTrue(i < len(content), error_msg)
        for k, v in content[i].items():
          value = getattr(current, k)
          error_msg = " ".join([
              "Value `%s` for key `%s` of file" % (value, k),
              "%s in archive %s does" % (current.filename, arfile),
              "not match expected value `%s`" % v
              ])
          self.assertEqual(value, v, error_msg)
        current = f.next()
        i += 1
      if i < len(content):
        self.fail("Missing file %s in archive %s" % (content[i], arfile))

  def testEmptyArFile(self):
    self.assertArFileContent(os.path.join(testenv.TESTDATA_PATH,
                                          "archive", "empty.ar"),
                             [])

  def assertSimpleFileContent(self, names):
    datafile = os.path.join(testenv.TESTDATA_PATH, "archive",
                            "_".join(names) + ".ar")
    content = [{"filename": n, "size": len(n), "data": n} for n in names]
    self.assertArFileContent(datafile, content)

  def testAFile(self):
    self.assertSimpleFileContent(["a"])

  def testBFile(self):
    self.assertSimpleFileContent(["b"])

  def testABFile(self):
    self.assertSimpleFileContent(["ab"])

  def testA_BFile(self):
    self.assertSimpleFileContent(["a", "b"])

  def testA_ABFile(self):
    self.assertSimpleFileContent(["a", "ab"])

  def testA_B_ABFile(self):
    self.assertSimpleFileContent(["a", "b", "ab"])


class TarFileWriterTest(unittest.TestCase):
  """Testing for TarFileWriter class."""

  def assertTarFileContent(self, tar, content):
    """Assert that tarfile contains exactly the entry described by `content`.

    Args:
        tar: the path to the TAR file to test.
        content: an array describing the expected content of the TAR file.
            Each entry in that list should be a dictionary where each field
            is a field to test in the corresponding TarInfo. For
            testing the presence of a file "x", then the entry could simply
            be `{"name": "x"}`, the missing field will be ignored. To match
            the content of a file entry, use the key "data".
    """
    with tarfile.open(tar, "r:") as f:
      i = 0
      for current in f:
        error_msg = "Extraneous file at end of archive %s: %s" % (
            tar,
            current.name
            )
        self.assertTrue(i < len(content), error_msg)
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
    self.tempfile = os.path.join(os.environ["TEST_TMPDIR"], "test.tar")

  def tearDown(self):
    if os.path.exists(self.tempfile):
      os.remove(self.tempfile)

  def testEmptyTarFile(self):
    with archive.TarFileWriter(self.tempfile):
      pass
    self.assertTarFileContent(self.tempfile, [])

  def assertSimpleFileContent(self, names):
    with archive.TarFileWriter(self.tempfile) as f:
      for n in names:
        f.add_file(n, content=n)
    content = [{"name": n, "size": len(n), "data": n} for n in names]
    self.assertTarFileContent(self.tempfile, content)

  def testAddFile(self):
    self.assertSimpleFileContent(["./a"])
    self.assertSimpleFileContent(["./b"])
    self.assertSimpleFileContent(["./ab"])
    self.assertSimpleFileContent(["./a", "./b"])
    self.assertSimpleFileContent(["./a", "./ab"])
    self.assertSimpleFileContent(["./a", "./b", "./ab"])

  def testDottedFiles(self):
    with archive.TarFileWriter(self.tempfile) as f:
      f.add_file("a")
      f.add_file("/b")
      f.add_file("./c")
      f.add_file("./.d")
      f.add_file("..e")
      f.add_file(".f")
    content = [
        {"name": "./a"},
        {"name": "/b"},
        {"name": "./c"},
        {"name": "./.d"},
        {"name": "./..e"},
        {"name": "./.f"}
        ]
    self.assertTarFileContent(self.tempfile, content)

  def testAddDir(self):
    # For some strange reason, ending slash is stripped by the test
    content = [
        {"name": ".", "mode": 0755},
        {"name": "./a", "mode": 0755},
        {"name": "./a/b", "data": "ab", "mode": 0644},
        {"name": "./a/c", "mode": 0755},
        {"name": "./a/c/d", "data": "acd", "mode": 0644},
        ]
    tempdir = os.path.join(os.environ["TEST_TMPDIR"], "test_dir")
    # Iterate over the `content` array to create the directory
    # structure it describes.
    for c in content:
      if "data" in c:
        p = os.path.join(tempdir, c["name"][2:])
        os.makedirs(os.path.dirname(p))
        with open(p, "w") as f:
          f.write(c["data"])
    with archive.TarFileWriter(self.tempfile) as f:
      f.add_dir("./", tempdir, mode=0644)
    self.assertTarFileContent(self.tempfile, content)

  def testMergeTar(self):
    content = [
        {"name": "./a", "data": "a"},
        {"name": "./ab", "data": "ab"},
        ]
    for ext in ["", ".gz", ".bz2", ".xz"]:
      with archive.TarFileWriter(self.tempfile) as f:
        f.add_tar(os.path.join(testenv.TESTDATA_PATH, "archive",
                               "tar_test.tar" + ext),
                  name_filter=lambda n: n != "./b")
      self.assertTarFileContent(self.tempfile, content)


if __name__ == "__main__":
  unittest.main()
