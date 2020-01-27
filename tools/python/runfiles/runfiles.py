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
"""Runfiles lookup library for Bazel-built Python binaries and tests.

USAGE:

1.  Depend on this runfiles library from your build rule:

      py_binary(
          name = "my_binary",
          ...
          deps = ["@rules_python//python/runfiles"],
      )

2.  Import the runfiles library.

      from rules_python.python.runfiles import runfiles

3.  Create a Runfiles object and use rlocation to look up runfile paths:

      r = runfiles.Create()
      ...
      with open(r.Rlocation("my_workspace/path/to/my/data.txt"), "r") as f:
        contents = f.readlines()
        ...

    The code above creates a manifest- or directory-based implementations based
    on the environment variables in os.environ. See `Create()` for more info.

    If you want to explicitly create a manifest- or directory-based
    implementations, you can do so as follows:

      r1 = runfiles.CreateManifestBased("path/to/foo.runfiles_manifest")

      r2 = runfiles.CreateDirectoryBased("path/to/foo.runfiles/")

    If you want to start subprocesses that also need runfiles, you need to set
    the right environment variables for them:

      import subprocess
      from rules_python.python.runfiles import runfiles

      r = runfiles.Create()
      env = {}
      ...
      env.update(r.EnvVars())
      p = subprocess.Popen([r.Rlocation("path/to/binary")], env, ...)
"""

import os
import posixpath


def CreateManifestBased(manifest_path):
  return _Runfiles(_ManifestBased(manifest_path))


def CreateDirectoryBased(runfiles_dir_path):
  return _Runfiles(_DirectoryBased(runfiles_dir_path))


def Create(env=None):
  """Returns a new `Runfiles` instance.

  The returned object is either:
  - manifest-based, meaning it looks up runfile paths from a manifest file, or
  - directory-based, meaning it looks up runfile paths under a given directory
    path

  If `env` contains "RUNFILES_MANIFEST_FILE" with non-empty value, this method
  returns a manifest-based implementation. The object eagerly reads and caches
  the whole manifest file upon instantiation; this may be relevant for
  performance consideration.

  Otherwise, if `env` contains "RUNFILES_DIR" with non-empty value (checked in
  this priority order), this method returns a directory-based implementation.

  If neither cases apply, this method returns null.

  Args:
    env: {string: string}; optional; the map of environment variables. If None,
        this function uses the environment variable map of this process.
  Raises:
    IOError: if some IO error occurs.
  """
  env_map = os.environ if env is None else env
  manifest = env_map.get("RUNFILES_MANIFEST_FILE")
  if manifest:
    return CreateManifestBased(manifest)

  directory = env_map.get("RUNFILES_DIR")
  if directory:
    return CreateDirectoryBased(directory)

  return None


class _Runfiles(object):
  """Returns the runtime location of runfiles.

  Runfiles are data-dependencies of Bazel-built binaries and tests.
  """

  def __init__(self, strategy):
    self._strategy = strategy

  def Rlocation(self, path):
    """Returns the runtime path of a runfile.

    Runfiles are data-dependencies of Bazel-built binaries and tests.

    The returned path may not be valid. The caller should check the path's
    validity and that the path exists.

    The function may return None. In that case the caller can be sure that the
    rule does not know about this data-dependency.

    Args:
      path: string; runfiles-root-relative path of the runfile
    Returns:
      the path to the runfile, which the caller should check for existence, or
      None if the method doesn't know about this runfile
    Raises:
      TypeError: if `path` is not a string
      ValueError: if `path` is None or empty, or it's absolute or not normalized
    """
    if not path:
      raise ValueError()
    if not isinstance(path, str):
      raise TypeError()
    if (path.startswith("../") or "/.." in path or path.startswith("./") or
        "/./" in path or path.endswith("/.") or "//" in path):
      raise ValueError("path is not normalized: \"%s\"" % path)
    if path[0] == "\\":
      raise ValueError("path is absolute without a drive letter: \"%s\"" % path)
    if os.path.isabs(path):
      return path
    return self._strategy.RlocationChecked(path)

  def EnvVars(self):
    """Returns environment variables for subprocesses.

    The caller should set the returned key-value pairs in the environment of
    subprocesses in case those subprocesses are also Bazel-built binaries that
    need to use runfiles.

    Returns:
      {string: string}; a dict; keys are environment variable names, values are
      the values for these environment variables
    """
    return self._strategy.EnvVars()


class _ManifestBased(object):
  """`Runfiles` strategy that parses a runfiles-manifest to look up runfiles."""

  def __init__(self, path):
    if not path:
      raise ValueError()
    if not isinstance(path, str):
      raise TypeError()
    self._path = path
    self._runfiles = _ManifestBased._LoadRunfiles(path)

  def RlocationChecked(self, path):
    return self._runfiles.get(path)

  @staticmethod
  def _LoadRunfiles(path):
    """Loads the runfiles manifest."""
    result = {}
    with open(path, "r") as f:
      for line in f:
        line = line.strip()
        if line:
          tokens = line.split(" ", 1)
          if len(tokens) == 1:
            result[line] = line
          else:
            result[tokens[0]] = tokens[1]
    return result

  def _GetRunfilesDir(self):
    if self._path.endswith("/MANIFEST") or self._path.endswith("\\MANIFEST"):
      return self._path[:-len("/MANIFEST")]
    elif self._path.endswith(".runfiles_manifest"):
      return self._path[:-len("_manifest")]
    else:
      return ""

  def EnvVars(self):
    directory = self._GetRunfilesDir()
    return {
        "RUNFILES_MANIFEST_FILE": self._path,
        "RUNFILES_DIR": directory,
        # TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can
        # pick up RUNFILES_DIR.
        "JAVA_RUNFILES": directory,
    }


class _DirectoryBased(object):
  """`Runfiles` strategy that appends runfiles paths to the runfiles root."""

  def __init__(self, path):
    if not path:
      raise ValueError()
    if not isinstance(path, str):
      raise TypeError()
    self._runfiles_root = path

  def RlocationChecked(self, path):
    # Use posixpath instead of os.path, because Bazel only creates a runfiles
    # tree on Unix platforms, so `Create()` will only create a directory-based
    # runfiles strategy on those platforms.
    return posixpath.join(self._runfiles_root, path)

  def EnvVars(self):
    return {
        "RUNFILES_DIR": self._runfiles_root,
        # TODO(laszlocsomor): remove JAVA_RUNFILES once the Java launcher can
        # pick up RUNFILES_DIR.
        "JAVA_RUNFILES": self._runfiles_root,
    }


def _PathsFrom(argv0, runfiles_mf, runfiles_dir, is_runfiles_manifest,
               is_runfiles_directory):
  """Discover runfiles manifest and runfiles directory paths.

  Args:
    argv0: string; the value of sys.argv[0]
    runfiles_mf: string; the value of the RUNFILES_MANIFEST_FILE environment
      variable
    runfiles_dir: string; the value of the RUNFILES_DIR environment variable
    is_runfiles_manifest: lambda(string):bool; returns true if the argument is
      the path of a runfiles manifest file
    is_runfiles_directory: lambda(string):bool; returns true if the argument is
      the path of a runfiles directory

  Returns:
    (string, string) pair, first element is the path to the runfiles manifest,
    second element is the path to the runfiles directory. If the first element
    is non-empty, then is_runfiles_manifest returns true for it. Same goes for
    the second element and is_runfiles_directory respectively. If both elements
    are empty, then this function could not find a manifest or directory for
    which is_runfiles_manifest or is_runfiles_directory returns true.
  """
  mf_alid = is_runfiles_manifest(runfiles_mf)
  dir_valid = is_runfiles_directory(runfiles_dir)

  if not mf_alid and not dir_valid:
    runfiles_mf = argv0 + ".runfiles/MANIFEST"
    runfiles_dir = argv0 + ".runfiles"
    mf_alid = is_runfiles_manifest(runfiles_mf)
    dir_valid = is_runfiles_directory(runfiles_dir)
    if not mf_alid:
      runfiles_mf = argv0 + ".runfiles_manifest"
      mf_alid = is_runfiles_manifest(runfiles_mf)

  if not mf_alid and not dir_valid:
    return ("", "")

  if not mf_alid:
    runfiles_mf = runfiles_dir + "/MANIFEST"
    mf_alid = is_runfiles_manifest(runfiles_mf)
    if not mf_alid:
      runfiles_mf = runfiles_dir + "_manifest"
      mf_alid = is_runfiles_manifest(runfiles_mf)

  if not dir_valid:
    runfiles_dir = runfiles_mf[:-9]  # "_manifest" or "/MANIFEST"
    dir_valid = is_runfiles_directory(runfiles_dir)

  return (runfiles_mf if mf_alid else "", runfiles_dir if dir_valid else "")
