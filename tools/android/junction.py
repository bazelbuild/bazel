# pylint: disable=g-direct-third-party-import
# Copyright 2017 The Bazel Authors. All rights reserved.
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
"""A class that creates junctions in temp directories on Windows.

Only use this class on Windows, do not use on other platforms. Other platforms
support longer paths than Windows, and also support symlinks. Windows only
supports junctions (directory symlinks).

Junctions are useful if you need to shorten a long path. A long path is one that
is at least MAX_PATH (260) letters long. This is a constant in Windows, see
<windows.h> and API documentation for file-handling functions such as
CreateFileA.
"""

import os
import subprocess
import tempfile


class JunctionCreationError(Exception):
  """Raised when TempJunction fails to create an NTFS junction."""

  def __init__(self, path, target, stdout):
    Exception.__init__(
        self,
        "Could not create junction \"%s\" -> \"%s\"\nError from mklink:\n%s" %
        (path, target, stdout))


class TempJunction(object):
  r"""Junction in a temp directory.

  This object creates a temp directory and a junction under it. The junction
  points to a user-specified path.

  Use this object if you want to write files under long paths (absolute path at
  least MAX_PATH (260) chars long). Pass the directory you want to "shorten" as
  the initializer's argument. This object will create a junction under a shorter
  path, that points to the long directory. The combined path of the junction and
  files under it are more likely to be short than the original paths were.

  Usage example:
    with TempJunction("C:/some/long/path") as junc:
      # `junc` created a temp directory and a junction in it that points to
      # \\?\C:\some\long\path, and is itself shorter than that
      shortpath = os.path.join(junc, "file.txt")
      with open(shortpath, "w") as f:
        ...do something with f...
    # `junc` deleted the junction and its parent temp directory upon leaving
    # the `with` statement's body
    ...do something else...
  """

  def __init__(self,
               junction_target,
               testonly_mkdtemp=None,
               testonly_maxpath=None):
    """Initialize this object.

    Args:
      junction_target: string; an absolute Windows path; the __enter__ method
        creates a junction that points to this path
      testonly_mkdtemp: function(); for testing only; a custom function that
        returns a temp directory path, you can use it to mock out
        tempfile.mkdtemp
      testonly_maxpath: int; for testing oly; maximum path length before the
        path is a "long path" (typically MAX_PATH on Windows)
    """
    self._target = os.path.abspath(junction_target)
    self._junction = None
    self._mkdtemp = testonly_mkdtemp or tempfile.mkdtemp
    self._max_path = testonly_maxpath or 248

  @staticmethod
  def _Mklink(name, target):
    proc = subprocess.Popen(
        "cmd.exe /C mklink /J \"%s\" \"\\\\?\\%s\"" % (name, target),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT)
    exitcode = proc.wait()
    if exitcode != 0:
      stdout = proc.communicate()[0]
      raise JunctionCreationError(name, target, stdout)

  @staticmethod
  def _TryMkdir(path):
    try:
      os.mkdir(path)
    except OSError as e:
      # Another process may have already created this directory.
      if not os.path.isdir(path):
        raise IOError("Could not create directory at '%s': %s" % (path, str(e)))

  @staticmethod
  def _MakeLinks(target, mkdtemp, max_path):
    """Creates a temp directory and a junction in it, pointing to `target`.

    Creates all parent directories of `target` if they don't exist.

    Args:
      target: string; path to the directory that is the junction's target
      mkdtemp: function():string; creates a temp directory and returns its
        absolute path
      max_path: int; maximum path length before the path is a "long path"
        (typically MAX_PATH on Windows)
    Returns:
      The full path to the junction.
    Raises:
      JunctionCreationError: if `mklink` fails to create a junction
    """
    segments = []
    dirpath = target
    while not os.path.isdir(dirpath):
      dirpath, child = os.path.split(dirpath)
      if child:
        segments.append(child)
    tmp = mkdtemp()
    juncpath = os.path.join(tmp, "j")
    for child in reversed(segments):
      childpath = os.path.join(dirpath, child)
      if len(childpath) >= max_path:
        try:
          TempJunction._Mklink(juncpath, dirpath)
          TempJunction._TryMkdir(os.path.join(juncpath, child))
        finally:
          os.rmdir(juncpath)
      else:
        TempJunction._TryMkdir(childpath)
      dirpath = childpath
    TempJunction._Mklink(juncpath, target)
    return juncpath

  def __enter__(self):
    """Creates a temp directory and a junction in it, pointing to self._target.

    Creates all parent directories of self._target if they don't exist.

    This method is automatically called upon entering a `with` statement's body.

    Returns:
      The full path to the junction.
    Raises:
      JunctionCreationError: if `mklink` fails to create a junction
    """
    self._junction = TempJunction._MakeLinks(self._target, self._mkdtemp,
                                             self._max_path)
    return self._junction

  def __exit__(self, unused_type, unused_value, unused_traceback):
    """Deletes the junction and its parent directory.

    This method is automatically called upon leaving a `with` statement's body.

    Args:
      unused_type: unused
      unused_value: unused
      unused_traceback: unused
    """
    if self._junction:
      os.rmdir(self._junction)
      os.rmdir(os.path.dirname(self._junction))
