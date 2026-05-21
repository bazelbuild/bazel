// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static java.util.Objects.requireNonNull;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.lib.vfs.UnixGlob;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Pattern;
import javax.annotation.Nonnull;

/**
 * Contains information about the recursive digest of a directory tree, including all transitive
 * descendant files and their contents.
 */
public record DirectoryTreeDigestValue(String hexDigest) implements SkyValue {
  public DirectoryTreeDigestValue {
    requireNonNull(hexDigest, "hexDigest");
  }

  public static DirectoryTreeDigestValue of(String hexDigest) {
    return new DirectoryTreeDigestValue(hexDigest);
  }

  public static Key key(
      RootedPath rootedPath, RootedPath globBase, @Nonnull ImmutableList<String> excludes) {
    return new Key(rootedPath, globBase, excludes);
  }

  /**
   * Key type for {@link DirectoryTreeDigestValue}.
   *
   * <p>The {@code rootedPath} indicates the directory tree to compute a digest for.
   *
   * <p>To filter out files/directories, you can optionally provide a {@code globBase} and glob
   * {@code excludes} patterns. They are joined together to create the paths to be filtered out. For
   * example, if the given parameters are:
   *
   * <pre>
   * rootedPath = "/tmp"
   * globBase = "/tmp/path"
   * excludes = [".git/**", "cache/ignoreMe"]
   * </pre>
   *
   * Then the glob patterns that will be filtered/excluded from under <code>rootedPath</code> would
   * be:
   *
   * <pre>
   * /tmp/path/.git/**
   * /tmp/path/cache/ignoreMe
   * </pre>
   */
  public static class Key implements SkyKey {

    private final RootedPath rootedPath;
    private final RootedPath globBase;
    private final ImmutableList<String> excludes;

    private Key(
        RootedPath rootedPath, RootedPath globBase, @Nonnull ImmutableList<String> excludes) {
      this.rootedPath = rootedPath;
      this.globBase = globBase;
      this.excludes = excludes;
    }

    @Override
    public int hashCode() {
      return Objects.hash(rootedPath, globBase, excludes);
    }

    @Override
    public final boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Key other)) {
        return false;
      }
      return rootedPath.equals(other.getRootedPath())
          && globBase.equals(other.getGlobBase())
          && excludes.equals(other.getExcludes());
    }

    public RootedPath getRootedPath() {
      return rootedPath;
    }

    public RootedPath getGlobBase() {
      return globBase;
    }

    public ImmutableList<String> getExcludes() {
      return excludes;
    }

    @Override
    public SkyFunctionName functionName() {
      return SkyFunctions.DIRECTORY_TREE_DIGEST;
    }

    /** Returns if the given {@code rootedPath} would be filtered/excluded out. */
    public boolean excludes(RootedPath rootedPath, Map<String, Pattern> patternCache) {
      // Are we comparing the same roots?
      if (!rootedPath.getRoot().equals(globBase.getRoot())) {
        return false;
      }
      String path = rootedPath.getRootRelativePath().toString();
      return excludes(path, patternCache);
    }

    /** Returns if the given {@code path} would be filtered/excluded out. */
    public boolean excludes(String path, Map<String, Pattern> patternCache) {
      PathFragment baseExclude = globBase.getRootRelativePath();
      for (String exclude : excludes) {
        String excludePattern = baseExclude.getRelative(exclude).toString();
        if (UnixGlob.matches(excludePattern, path, patternCache)) {
          return true;
        }
      }
      return false;
    }
  }
}
