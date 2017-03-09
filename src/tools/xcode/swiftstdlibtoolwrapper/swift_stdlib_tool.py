# pylint: disable=g-bad-file-header
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

"""A tool to find Swift runtime libraries required by a binary.

This tool is modeled after Xcode's swift-stdlib-tool. Given a binary, it
scans its transitive dylib dependencies to figure out the full set of Swift
runtime libraries (usually named libswift*.dylib) required to run the binary.
The libraries are then copied into the output directory.

This tool is used by the Apple packaging rules to properly construct macOS, iOS,
watchOS and tvOS app bundles.

Usage:
  swift-stdlib-tool.py BINARY_TO_SCAN PLATFORM_DIRECTORY OUTPUT_PATH
"""

import os
import shutil
import sys
from macholib.MachO import MachO


def dylib_full_path(platform_dir, relative_path):
  """Constructs an absolute path to a platform dylib.

  Args:
    platform_dir: A path to the platforms directory in the Swift toolchain.
    relative_path: A path to a dylib relative to the platforms directory.

  Returns:
    A normalized, absolute path to a dylib.
  """
  return os.path.abspath(os.path.join(platform_dir, relative_path))


def main():
  binary_path = sys.argv[1]
  platform_dir = sys.argv[2]
  out_path = sys.argv[3]

  # We want any dylib linked against which name starts with "libswift"
  seen = set()
  queue = [binary_path]
  while queue:
    path = queue.pop()
    m = MachO(path)
    for header in m.headers:
      for _, _, other in header.walkRelocatables():
        if other.startswith("@rpath/libswift"):
          full_path = dylib_full_path(platform_dir, other.lstrip("@rpath/"))
          if full_path not in seen:
            queue.append(full_path)
            seen.add(full_path)

  for dylib in seen:
    shutil.copy(dylib, out_path)

if __name__ == "__main__":
  main()
