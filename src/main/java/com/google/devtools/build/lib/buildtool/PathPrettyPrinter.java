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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.config.SymlinkDefinition;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.util.LinkedHashMap;
import java.util.Map;

/** Uses information about the convenience symlinks to print shorter paths for output artifacts. */
public final class PathPrettyPrinter {
  private static final String NO_CREATE_SYMLINKS_PREFIX = "/";

  private final Map<PathFragment, PathFragment> resolvedSymlinks;
  private final String symlinkPrefix;
  private final String productName;
  private final Path workspaceDirectory;

  /**
   * Creates a path pretty printer, immediately resolving the symlink definitions by reading the
   * current symlinks _from disk_.
   */
  PathPrettyPrinter(
      ImmutableList<SymlinkDefinition> symlinkDefinitions,
      String symlinkPrefix,
      String productName,
      Path workspaceDirectory) {
    this.symlinkPrefix = symlinkPrefix;
    this.productName = productName;
    this.workspaceDirectory = workspaceDirectory;
    this.resolvedSymlinks = resolve(symlinkDefinitions);
  }

  private Map<PathFragment, PathFragment> resolve(
      ImmutableList<SymlinkDefinition> symlinkDefinitions) {
    Map<PathFragment, PathFragment> result = new LinkedHashMap<>();
    String workspaceBaseName = workspaceDirectory.getBaseName();
    for (SymlinkDefinition link : symlinkDefinitions) {
      String linkName = link.getLinkName(symlinkPrefix, productName, workspaceBaseName);
      PathFragment linkFragment = PathFragment.create(linkName);
      Path dir = workspaceDirectory.getRelative(linkFragment);
      try {
        PathFragment levelOneLinkTarget = dir.readSymbolicLink();
        if (levelOneLinkTarget.isAbsolute()) {
          result.put(linkFragment, dir.getRelative(levelOneLinkTarget).asFragment());
        }
      } catch (IOException ignored) {
        // We don't guarantee that the convenience symlinks exist - e.g., we might be running in a
        // readonly directory. We silently fall back to printing the full path in that case. As an
        // alternative, we could capture that information when we create the symlinks and pass that
        // here instead of reading files back from local disk.
      }
    }
    return result;
  }

  /**
   * Returns a convenient path to the specified file, relativizing it and using output-dir symlinks
   * if possible. Otherwise, return the absolute path.
   *
   * <p>This method must be called after the symlinks are created at the end of a build. If called
   * before, the pretty path may be incorrect if the symlinks end up pointing somewhere new.
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
