// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.vfs;

import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.Objects;

/**
 * A {@link PathFragment} relative to a root, which is an absolute {@link Path}. Typically the root
 * will be a package path entry.
 *
 * Two {@link RootedPath}s are considered equal iff they have equal roots and equal relative paths.
 *
 * TODO(bazel-team): refactor Artifact to use this instead of Root.
 * TODO(bazel-team): use an opaque root representation so as to not expose the absolute path to
 * clients via #asPath or #getRoot.
 */
public class RootedPath implements Serializable {

  private final Path root;
  private final PathFragment relativePath;
  private final Path path;

  /**
   * Constructs a {@link RootedPath} from an absolute root path and a non-absolute relative path.
   */
  private RootedPath(Path root, PathFragment relativePath) {
    Preconditions.checkState(!relativePath.isAbsolute(), "relativePath: %s root: %s", relativePath,
        root);
    this.root = root;
    this.relativePath = relativePath.normalize();
    this.path = root.getRelative(this.relativePath);
  }

  /**
   * Returns a rooted path representing {@code relativePath} relative to {@code root}.
   */
  public static RootedPath toRootedPath(Path root, PathFragment relativePath) {
    return new RootedPath(root, relativePath);
  }

  /**
   * Returns a rooted path representing {@code path} under the root {@code root}.
   */
  public static RootedPath toRootedPath(Path root, Path path) {
    Preconditions.checkState(path.startsWith(root), "path: %s root: %s", path, root);
    return new RootedPath(root, path.relativeTo(root));
  }

  /**
   * Returns a rooted path representing {@code path} under one of the package roots, or under the
   * filesystem root if it's not under any package root.
   */
  public static RootedPath toRootedPathMaybeUnderRoot(Path path, Iterable<Path> packagePathRoots) {
    for (Path root : packagePathRoots) {
      if (path.startsWith(root)) {
        return toRootedPath(root, path);
      }
    }
    return toRootedPath(path.getFileSystem().getRootDirectory(), path);
  }

  public Path asPath() {
    // Ideally, this helper method would not be needed. But Skyframe's FileFunction and
    // DirectoryListingFunction need to do filesystem operations on the absolute path and
    // Path#getRelative(relPath) is O(relPath.segmentCount()). Therefore we precompute the absolute
    // path represented by this relative path.
    return path;
  }

  public Path getRoot() {
    return root;
  }

  /**
   * Returns the (normalized) path relative to {@code #getRoot}.
   */
  public PathFragment getRelativePath() {
    return relativePath;
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RootedPath)) {
      return false;
    }
    RootedPath other = (RootedPath) obj;
    return Objects.equals(root, other.root) && Objects.equals(relativePath, other.relativePath);
  }

  @Override
  public int hashCode() {
    return Objects.hash(root, relativePath);
  }

  @Override
  public String toString() {
    return "[" + root + "]/[" + relativePath + "]";
  }
}
