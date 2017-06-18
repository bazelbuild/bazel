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
"""Rules for manipulating paths."""


def dirname(path):
  """Returns the directory's name."""
  last_sep = path.rfind("/")
  if last_sep == -1:
    return ""  # The artifact is at the top level.
  return path[:last_sep]


def join(directory, path):
  """Compute the relative data path prefix from the data_path attribute."""
  if not path:
    return directory
  if path[0] == "/":
    return path[1:]
  if directory == "/":
    return path
  return directory + "/" + path


def canonicalize(path):
  """Canonicalize the input path."""
  if not path:
    return path
  # Strip ./ from the beginning if specified.
  # There is no way to handle .// correctly (no function that would make
  # that possible and Skylark is not turing complete) so just consider it
  # as an absolute path. A path of / should preserve the entire
  # path up to the repository root.
  if path == "/":
    return path
  if len(path) >= 2 and path[0:2] == "./":
    path = path[2:]
  if not path or path == ".":  # Relative to current package
    return ""
  elif path[0] == "/":  # Absolute path
    return path
  else:  # Relative to a sub-directory
    return path


def strip_prefix(path, prefix):
  """Returns the path with the specified prefix removed."""
  if path.startswith(prefix):
    return path[len(prefix):]
  return path


def runfile(ctx, f):
  """Return the runfiles relative path of f."""
  if ctx.workspace_name:
    return ctx.workspace_name + "/" + f.short_path
  else:
    return f.short_path
