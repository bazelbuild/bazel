# pylint: disable=g-bad-file-header
# Copyright 2018 The Bazel Authors. All rights reserved.
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

import os
import tempfile
import unittest

from src.tools.runfiles import runfiles


class RunfilesTest(unittest.TestCase):
  # """Unit tests for `runfiles.Runfiles`."""

  def testRlocationArgumentValidation(self):
    r = runfiles.Create({"RUNFILES_DIR": "whatever"})
    self.assertRaises(ValueError, lambda: r.Rlocation(None))
    self.assertRaises(ValueError, lambda: r.Rlocation(""))
    self.assertRaises(TypeError, lambda: r.Rlocation(1))
    self.assertRaisesRegexp(ValueError, "contains uplevel",
                            lambda: r.Rlocation("foo/.."))
    if RunfilesTest.IsWindows():
      self.assertRaisesRegexp(ValueError, "is absolute",
                              lambda: r.Rlocation("\\foo"))
      self.assertRaisesRegexp(ValueError, "is absolute",
                              lambda: r.Rlocation("c:/foo"))
      self.assertRaisesRegexp(ValueError, "is absolute",
                              lambda: r.Rlocation("c:\\foo"))
    else:
      self.assertRaisesRegexp(ValueError, "is absolute",
                              lambda: r.Rlocation("/foo"))

  def testCreatesManifestBasedRunfiles(self):
    with _MockFile(["a/b c/d"]) as mf:
      r = runfiles.Create({
          "RUNFILES_MANIFEST_FILE": mf.Path(),
          "RUNFILES_DIR": "ignored when RUNFILES_MANIFEST_FILE has a value",
          "TEST_SRCDIR": "ignored when RUNFILES_MANIFEST_FILE has a value"
      })
      self.assertEqual(r.Rlocation("a/b"), "c/d")
      self.assertIsNone(r.Rlocation("foo"))
      self.assertDictEqual(r.EnvVar(), {"RUNFILES_MANIFEST_FILE": mf.Path()})

  def testCreatesDirectoryBasedRunfiles(self):
    r = runfiles.Create({
        "RUNFILES_DIR": "runfiles/dir",
        "TEST_SRCDIR": "ignored when RUNFILES_DIR is set"
    })
    self.assertEqual(r.Rlocation("a/b"), "runfiles/dir/a/b")
    self.assertEqual(r.Rlocation("foo"), "runfiles/dir/foo")
    self.assertDictEqual(r.EnvVar(), {"RUNFILES_DIR": "runfiles/dir"})

    r = runfiles.Create({"TEST_SRCDIR": "test/srcdir"})
    self.assertEqual(r.Rlocation("a/b"), "test/srcdir/a/b")
    self.assertEqual(r.Rlocation("foo"), "test/srcdir/foo")
    self.assertDictEqual(r.EnvVar(), {"RUNFILES_DIR": "test/srcdir"})

  def testFailsToCreateManifestBasedBecauseManifestDoesNotExist(self):

    def _Run():
      runfiles.Create({"RUNFILES_MANIFEST_FILE": "non-existing path"})

    self.assertRaisesRegexp(IOError, "non-existing path", _Run)

  def testFailsToCreateAnyRunfilesBecauseEnvvarsAreNotDefined(self):
    with _MockFile(["a b"]) as mf:
      runfiles.Create({
          "RUNFILES_MANIFEST_FILE": mf.Path(),
          "RUNFILES_DIR": "whatever",
          "TEST_SRCDIR": "whatever"
      })
    runfiles.Create({"RUNFILES_DIR": "whatever", "TEST_SRCDIR": "whatever"})
    runfiles.Create({"TEST_SRCDIR": "whatever"})
    self.assertIsNone(runfiles.Create({"FOO": "bar"}))

  def testManifestBasedRlocation(self):
    with _MockFile([
        "Foo/runfile1", "Foo/runfile2 C:/Actual Path\\runfile2",
        "Foo/Bar/runfile3 D:\\the path\\run file 3.txt"
    ]) as mf:
      r = runfiles.CreateManifestBased(mf.Path())
      self.assertEqual(r.Rlocation("Foo/runfile1"), "Foo/runfile1")
      self.assertEqual(r.Rlocation("Foo/runfile2"), "C:/Actual Path\\runfile2")
      self.assertEqual(
          r.Rlocation("Foo/Bar/runfile3"), "D:\\the path\\run file 3.txt")
      self.assertIsNone(r.Rlocation("unknown"))
      self.assertDictEqual(r.EnvVar(), {"RUNFILES_MANIFEST_FILE": mf.Path()})

  def testDirectoryBasedRlocation(self):
    # The _DirectoryBased strategy simply joins the runfiles directory and the
    # runfile's path on a "/". This strategy does not perform any normalization,
    # nor does it check that the path exists.
    r = runfiles.CreateDirectoryBased("foo/bar baz//qux/")
    self.assertEqual(r.Rlocation("arg"), "foo/bar baz//qux/arg")
    self.assertDictEqual(r.EnvVar(), {"RUNFILES_DIR": "foo/bar baz//qux/"})

  @staticmethod
  def IsWindows():
    return os.name == "nt"


class _MockFile(object):

  def __init__(self, contents):
    self._contents = contents
    self._path = None

  def __enter__(self):
    tmpdir = os.environ.get("TEST_TMPDIR")
    self._path = os.path.join(tempfile.mkdtemp(dir=tmpdir), "x")
    with open(self._path, "wt") as f:
      f.writelines(l + "\n" for l in self._contents)
    return self

  def __exit__(self, exc_type, exc_value, traceback):
    os.remove(self._path)
    os.rmdir(os.path.dirname(self._path))

  def Path(self):
    return self._path


if __name__ == "__main__":
  unittest.main()
