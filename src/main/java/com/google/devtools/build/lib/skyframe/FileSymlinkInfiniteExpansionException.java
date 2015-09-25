// Copyright 2015 The Bazel Authors. All rights reserved.
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

/** Exception indicating that a symlink has an unbounded expansion on resolution. */
public class FileSymlinkInfiniteExpansionException extends FileSymlinkException {
  private final ImmutableList<RootedPath> pathToChain;
  private final ImmutableList<RootedPath> chain;

  FileSymlinkInfiniteExpansionException(ImmutableList<RootedPath> pathToChain,
      ImmutableList<RootedPath> chain) {
    // The infinite expansion has already been reported by
    // FileSymlinkInfiniteExpansionUniquenessValue, but we still want to have a readable
    // #getMessage.
    super("Infinite symlink expansion");
    this.pathToChain = pathToChain;
    this.chain = chain;
  }

  /**
   * The symlink path to the symlink that is the root cause of the infinite expansion. For example,
   * suppose 'a' -> 'b' -> 'c' -> 'd' -> 'c/nope'. The path to the chain is 'a', 'b'.
   */
  ImmutableList<RootedPath> getPathToChain() {
    return pathToChain;
  }

  /**
   * The symlink chain that is the root cause of the infinite expansion. For example, suppose
   * 'a' -> 'b' -> 'c' -> 'd' -> 'c/nope'. The chain is 'c', 'd', 'c/nope'.
   */
  ImmutableList<RootedPath> getChain() {
    return chain;
  }
}

