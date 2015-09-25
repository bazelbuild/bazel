// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.skyframe;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.RootedPath;

/** Exception indicating that a cycle was found in the filesystem. */
class FileSymlinkCycleException extends FileSymlinkException {
  private final ImmutableList<RootedPath> pathToCycle;
  private final ImmutableList<RootedPath> cycle;

  FileSymlinkCycleException(ImmutableList<RootedPath> pathToCycle,
      ImmutableList<RootedPath> cycle) {
    // The cycle itself has already been reported by FileSymlinkCycleUniquenessValue, but we still
    // want to have a readable #getMessage.
    super("Symlink cycle");
    this.pathToCycle = pathToCycle;
    this.cycle = cycle;
  }

  /**
   * The symlink path to the symlink cycle. For example, suppose 'a' -> 'b' -> 'c' -> 'd' -> 'c'.
   * The path to the cycle is 'a', 'b'.
   */
  ImmutableList<RootedPath> getPathToCycle() {
    return pathToCycle;
  }

  /**
   * The symlink cycle. For example, suppose 'a' -> 'b' -> 'c' -> 'd' -> 'c'.
   * The cycle is 'c', 'd'.
   */
  ImmutableList<RootedPath> getCycle() {
    return cycle;
  }
}
