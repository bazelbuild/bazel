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

from tools.python.runfiles import runfiles


class RunfilesTest(unittest.TestCase):
  # """Unit tests for `runfiles.Runfiles`."""

  def testRlocationArgumentValidation(self):
    r = runfiles.Create({"RUNFILES_DIR": "whatever"})
    self.assertRaises(ValueError, lambda: r.Rlocation(None))
    self.assertRaises(ValueError, lambda: r.Rlocation(""))
    self.assertRaises(TypeError, lambda: r.Rlocation(1))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("../foo"))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("foo/.."))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("foo/../bar"))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("./foo"))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("foo/."))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("foo/./bar"))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("//foobar"))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("foo//"))
    self.assertRaisesRegexp(ValueError, "is not normalized",
                            lambda: r.Rlocation("foo//bar"))
    self.assertRaisesRegexp(ValueError, "is absolute without a drive letter",
                            lambda: r.Rlocation("\\foo"))

  def testCreatesManifestBasedRunfiles(self):
    with _MockFile(contents=["a/b c/d"]) as mf:
      r = runfiles.Create({
          "RUNFILES_MANIFEST_FILE": mf.Path(),
          "RUNFILES_DIR": "ignored when RUNFILES_MANIFEST_FILE has a value",
          "TEST_SRCDIR": "always ignored",
      })
      self.assertEqual(r.Rlocation("a/b"), "c/d")
      self.assertIsNone(r.Rlocation("foo"))

  def testManifestBasedRunfilesEnvVars(self):
    with _MockFile(name="MANIFEST") as mf:
      r = runfiles.Create({
          "RUNFILES_MANIFEST_FILE": mf.Path(),
          "TEST_SRCDIR": "always ignored",
      })
      self.assertDictEqual(
          r.EnvVars(), {
              "RUNFILES_MANIFEST_FILE": mf.Path(),
              "RUNFILES_DIR": mf.Path()[:-len("/MANIFEST")],
              "JAVA_RUNFILES": mf.Path()[:-len("/MANIFEST")],
          })

    with _MockFile(name="foo.runfiles_manifest") as mf:
      r = runfiles.Create({
          "RUNFILES_MANIFEST_FILE": mf.Path(),
          "TEST_SRCDIR": "always ignored",
      })
      self.assertDictEqual(
          r.EnvVars(), {
              "RUNFILES_MANIFEST_FILE":
                  mf.Path(),
              "RUNFILES_DIR": (
                  mf.Path()[:-len("foo.runfiles_manifest")] + "foo.runfiles"),
              "JAVA_RUNFILES": (
                  mf.Path()[:-len("foo.runfiles_manifest")] + "foo.runfiles"),
          })

    with _MockFile(name="x_manifest") as mf:
      r = runfiles.Create({
          "RUNFILES_MANIFEST_FILE": mf.Path(),
          "TEST_SRCDIR": "always ignored",
      })
      self.assertDictEqual(
          r.EnvVars(), {
              "RUNFILES_MANIFEST_FILE": mf.Path(),
              "RUNFILES_DIR": "",
              "JAVA_RUNFILES": "",
          })

  def testCreatesDirectoryBasedRunfiles(self):
    r = runfiles.Create({
        "RUNFILES_DIR": "runfiles/dir",
        "TEST_SRCDIR": "always ignored",
    })
    self.assertEqual(r.Rlocation("a/b"), "runfiles/dir/a/b")
    self.assertEqual(r.Rlocation("foo"), "runfiles/dir/foo")

  def testDirectoryBasedRunfilesEnvVars(self):
    r = runfiles.Create({
        "RUNFILES_DIR": "runfiles/dir",
        "TEST_SRCDIR": "always ignored",
    })
    self.assertDictEqual(r.EnvVars(), {
        "RUNFILES_DIR": "runfiles/dir",
        "JAVA_RUNFILES": "runfiles/dir",
    })

  def testFailsToCreateManifestBasedBecauseManifestDoesNotExist(self):

    def _Run():
      runfiles.Create({"RUNFILES_MANIFEST_FILE": "non-existing path"})

    self.assertRaisesRegexp(IOError, "non-existing path", _Run)

  def testFailsToCreateAnyRunfilesBecauseEnvvarsAreNotDefined(self):
    with _MockFile(contents=["a b"]) as mf:
      runfiles.Create({
          "RUNFILES_MANIFEST_FILE": mf.Path(),
          "RUNFILES_DIR": "whatever",
          "TEST_SRCDIR": "always ignored",
      })
    runfiles.Create({
        "RUNFILES_DIR": "whatever",
        "TEST_SRCDIR": "always ignored",
    })
    self.assertIsNone(runfiles.Create({"TEST_SRCDIR": "always ignored"}))
    self.assertIsNone(runfiles.Create({"FOO": "bar"}))

  def testManifestBasedRlocation(self):
    with _MockFile(contents=[
        "Foo/runfile1", "Foo/runfile2 C:/Actual Path\\runfile2",
        "Foo/Bar/runfile3 D:\\the path\\run file 3.txt"
    ]) as mf:
      r = runfiles.CreateManifestBased(mf.Path())
      self.assertEqual(r.Rlocation("Foo/runfile1"), "Foo/runfile1")
      self.assertEqual(r.Rlocation("Foo/runfile2"), "C:/Actual Path\\runfile2")
      self.assertEqual(
          r.Rlocation("Foo/Bar/runfile3"), "D:\\the path\\run file 3.txt")
      self.assertIsNone(r.Rlocation("unknown"))
      if RunfilesTest.IsWindows():
        self.assertEqual(r.Rlocation("c:/foo"), "c:/foo")
        self.assertEqual(r.Rlocation("c:\\foo"), "c:\\foo")
      else:
        self.assertEqual(r.Rlocation("/foo"), "/foo")

  def testDirectoryBasedRlocation(self):
    # The _DirectoryBased strategy simply joins the runfiles directory and the
    # runfile's path on a "/". This strategy does not perform any normalization,
    # nor does it check that the path exists.
    r = runfiles.CreateDirectoryBased("foo/bar baz//qux/")
    self.assertEqual(r.Rlocation("arg"), "foo/bar baz//qux/arg")
    if RunfilesTest.IsWindows():
      self.assertEqual(r.Rlocation("c:/foo"), "c:/foo")
      self.assertEqual(r.Rlocation("c:\\foo"), "c:\\foo")
    else:
      self.assertEqual(r.Rlocation("/foo"), "/foo")

  def testPathsFromEnvvars(self):
    # Both envvars have a valid value.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "mock1/MANIFEST",
                                 lambda path: path == "mock2")
    self.assertEqual(mf, "mock1/MANIFEST")
    self.assertEqual(dr, "mock2")

    # RUNFILES_MANIFEST_FILE is invalid but RUNFILES_DIR is good and there's a
    # runfiles manifest in the runfiles directory.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "mock2/MANIFEST",
                                 lambda path: path == "mock2")
    self.assertEqual(mf, "mock2/MANIFEST")
    self.assertEqual(dr, "mock2")

    # RUNFILES_MANIFEST_FILE is invalid but RUNFILES_DIR is good, but there's no
    # runfiles manifest in the runfiles directory.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: False,
                                 lambda path: path == "mock2")
    self.assertEqual(mf, "")
    self.assertEqual(dr, "mock2")

    # RUNFILES_DIR is invalid but RUNFILES_MANIFEST_FILE is good, and it is in
    # a valid-looking runfiles directory.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "mock1/MANIFEST",
                                 lambda path: path == "mock1")
    self.assertEqual(mf, "mock1/MANIFEST")
    self.assertEqual(dr, "mock1")

    # RUNFILES_DIR is invalid but RUNFILES_MANIFEST_FILE is good, but it is not
    # in any valid-looking runfiles directory.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "mock1/MANIFEST",
                                 lambda path: False)
    self.assertEqual(mf, "mock1/MANIFEST")
    self.assertEqual(dr, "")

    # Both envvars are invalid, but there's a manifest in a runfiles directory
    # next to argv0, however there's no other content in the runfiles directory.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "argv0.runfiles/MANIFEST",
                                 lambda path: False)
    self.assertEqual(mf, "argv0.runfiles/MANIFEST")
    self.assertEqual(dr, "")

    # Both envvars are invalid, but there's a manifest next to argv0. There's
    # no runfiles tree anywhere.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "argv0.runfiles_manifest",
                                 lambda path: False)
    self.assertEqual(mf, "argv0.runfiles_manifest")
    self.assertEqual(dr, "")

    # Both envvars are invalid, but there's a valid manifest next to argv0, and
    # a valid runfiles directory (without a manifest in it).
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "argv0.runfiles_manifest",
                                 lambda path: path == "argv0.runfiles")
    self.assertEqual(mf, "argv0.runfiles_manifest")
    self.assertEqual(dr, "argv0.runfiles")

    # Both envvars are invalid, but there's a valid runfiles directory next to
    # argv0, though no manifest in it.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: False,
                                 lambda path: path == "argv0.runfiles")
    self.assertEqual(mf, "")
    self.assertEqual(dr, "argv0.runfiles")

    # Both envvars are invalid, but there's a valid runfiles directory next to
    # argv0 with a valid manifest in it.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: path == "argv0.runfiles/MANIFEST",
                                 lambda path: path == "argv0.runfiles")
    self.assertEqual(mf, "argv0.runfiles/MANIFEST")
    self.assertEqual(dr, "argv0.runfiles")

    # Both envvars are invalid and there's no runfiles directory or manifest
    # next to the argv0.
    mf, dr = runfiles._PathsFrom("argv0", "mock1/MANIFEST", "mock2",
                                 lambda path: False, lambda path: False)
    self.assertEqual(mf, "")
    self.assertEqual(dr, "")

  @staticmethod
  def IsWindows():
    return os.name == "nt"


class _MockFile(object):

  def __init__(self, name=None, contents=None):
    self._contents = contents or []
    self._name = name or "x"
    self._path = None

  def __enter__(self):
    tmpdir = os.environ.get("TEST_TMPDIR")
    self._path = os.path.join(tempfile.mkdtemp(dir=tmpdir), self._name)
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
