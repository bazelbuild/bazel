# Copyright 2015 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#  http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""This package contains various functions used when building containers."""

import json
import tarfile


def ExtractValue(value):
  """Return the contents of a file point to by value if it starts with an @.

  Args:
    value: The possible filename to extract or a string.

  Returns:
    The content of the file if value starts with an @, or the passed value.
  """
  if value.startswith('@'):
    with open(value[1:], 'r') as f:
      value = f.read()
  return value


def GetTarFile(f, name):
  """Returns the content of a file inside a tar file.

  This method looks for ./f, /f and f file entry in a tar file and if found,
  return its content. This allows to read file with various path prefix.

  Args:
    f: The tar file to read.
    name: The name of the file inside the tar file.

  Returns:
    The content of the file, or None if not found.
  """
  with tarfile.open(f, 'r') as tar:
    members = [tarinfo.name for tarinfo in tar.getmembers()]
    for i in ['', './', '/']:
      if i + name in members:
        return tar.extractfile(i + name).read()
    return None


def GetManifestFromTar(f=None):
  """Returns the manifest array from a tar file.

  Args:
    f: The tar file to read.

  Returns:
    The content of the manifest file or an empty array if not found.
  """
  if f:
    raw_manifest_data = GetTarFile(f, 'manifest.json')
    if raw_manifest_data:
      return json.loads(raw_manifest_data)
  return []


def GetLatestManifestFromTar(f=None):
  """Returns the latest manifest entry from a tar file.

  The latest manifest entry is the one at the bottom.

  Args:
    f: The tar file to read.

  Returns:
    The latest manifest entry object, or None if not found.
  """
  manifest_data = GetManifestFromTar(f)
  return manifest_data[-1] if manifest_data else None
