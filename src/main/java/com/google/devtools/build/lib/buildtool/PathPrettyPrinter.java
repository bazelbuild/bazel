// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.devtools.build.lib.vfs.PathFragment;
import java.util.Map;

/**
 * Uses information about the convenience symlinks to print shorter paths for output artifacts.
 *
 * <p>Instances cannot be reused across builds - they must be used for the build associated with the
 * provided symlinks. If instances are reused the pretty path may be incorrect, for example if the
 * symlinks end up pointing somewhere new.
 */
public final class PathPrettyPrinter {
  private static final String NO_CREATE_SYMLINKS_PREFIX = "/";

  private final String symlinkPrefix;
  private final Map<PathFragment, PathFragment> resolvedSymlinks;

  public PathPrettyPrinter(
      String symlinkPrefix, Map<PathFragment, PathFragment> convenienceSymlinks) {
    this.symlinkPrefix = symlinkPrefix;
    this.resolvedSymlinks = convenienceSymlinks;
  }

  /**
   * Returns a convenient path to the specified file, relativizing it and using convenience symlinks
   * if possible. Otherwise, return the original path.
   */
  public PathFragment getPrettyPath(PathFragment file) {
    if (NO_CREATE_SYMLINKS_PREFIX.equals(symlinkPrefix)) {
      return file;
    }

    for (Map.Entry<PathFragment, PathFragment> e : resolvedSymlinks.entrySet()) {
      PathFragment linkFragment = e.getKey();
      PathFragment linkTarget = e.getValue();
      if (file.startsWith(linkTarget)) {
        return linkFragment.getRelative(file.relativeTo(linkTarget));
      }
    }

    return file;
  }
}
