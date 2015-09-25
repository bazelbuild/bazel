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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Optional;
import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Utility class for removing a prefix from an archive's path.
 */
@ThreadSafety.Immutable
public final class StripPrefixedPath {
  private final PathFragment pathFragment;
  private final boolean found;
  private final boolean skip;

  public static StripPrefixedPath maybeDeprefix(String entry, Optional<String> prefix) {
    boolean found = false;
    PathFragment entryPath = new PathFragment(entry);
    if (!prefix.isPresent()) {
      return new StripPrefixedPath(entryPath, false, false);
    }

    PathFragment prefixPath = new PathFragment(prefix.get());
    boolean skip = false;
    if (entryPath.startsWith(prefixPath)) {
      found = true;
      entryPath = entryPath.relativeTo(prefixPath);
      if (entryPath.getPathString().isEmpty()) {
        skip = true;
      }
    } else {
      skip = true;
    }
    return new StripPrefixedPath(entryPath, found, skip);
  }

  private StripPrefixedPath(PathFragment pathFragment, boolean found, boolean skip) {
    this.pathFragment = pathFragment;
    this.found = found;
    this.skip = skip;
  }

  public PathFragment getPathFragment() {
    return pathFragment;
  }

  public boolean foundPrefix() {
    return found;
  }

  public boolean skip() {
    return skip;
  }

}
