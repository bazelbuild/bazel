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
package com.google.devtools.build.lib.vfs;

import static com.google.common.base.Preconditions.checkArgument;

import com.google.devtools.build.lib.skyframe.serialization.VisibleForSerialization;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Comparator;
import javax.annotation.Nullable;

/**
 * A {@link PathFragment} relative to a {@link Root}. Typically, the root is a package path entry.
 *
 * <p>Two {@link RootedPath}s are considered equal iff they have equal roots and equal relative
 * paths.
 *
 * <p>Instances are interned (except on Windows), which results in a large memory benefit (see
 * cl/516855266). In addition to being a {@link SkyKey} itself, {@link RootedPath} is used as a
 * field in several other common {@link SkyKey} types. Interning on the level of those keys does not
 * deduplicate referenced {@link RootedPath} instances which are also used as a {@link SkyKey}
 * directly.
 */
@AutoCodec
public final class RootedPath implements Comparable<RootedPath>, FileStateKey {

  // Interning on Windows (case-insensitive) surfaces a bug where paths that only differ in casing
  // use the same RootedPath instance.
  // TODO(#17904): Investigate this bug and add test coverage.
  @Nullable
  private static final SkyKeyInterner<RootedPath> interner =
      OsPathPolicy.getFilePathOs().isCaseSensitive() ? SkyKey.newInterner() : null;

  private final Root root;
  private final PathFragment rootRelativePath;

  // Cache the hash code: RootedPath is used in several of the most common SkyKeys, and we have a
  // free field to spend on it.
  private final transient int hashCode;

  /** Constructs a {@link RootedPath} from a {@link Root} and path fragment relative to the root. */
  @AutoCodec.Instantiator
  @VisibleForSerialization
  static RootedPath createInternal(Root root, PathFragment rootRelativePath) {
    checkArgument(
        rootRelativePath.isAbsolute() == root.isAbsolute(),
        "rootRelativePath: %s root: %s",
        rootRelativePath,
        root);
    var rootedPath = new RootedPath(root, rootRelativePath);
    return interner != null ? interner.intern(rootedPath) : rootedPath;
  }

  private RootedPath(Root root, PathFragment rootRelativePath) {
    this.root = root;
    this.rootRelativePath = rootRelativePath;
    this.hashCode = 31 * root.hashCode() + rootRelativePath.hashCode();
  }

  /** Returns a rooted path representing {@code rootRelativePath} relative to {@code root}. */
  public static RootedPath toRootedPath(Root root, PathFragment rootRelativePath) {
    if (rootRelativePath.isAbsolute() && !root.isAbsolute()) {
      checkArgument(
          root.contains(rootRelativePath),
          "rootRelativePath '%s' is absolute, but it's not under root '%s'",
          rootRelativePath,
          root);
      rootRelativePath = root.relativize(rootRelativePath);
    }
    return createInternal(root, rootRelativePath);
  }

  /** Returns a rooted path representing {@code path} under the root {@code root}. */
  public static RootedPath toRootedPath(Root root, Path path) {
    checkArgument(root.contains(path), "path: %s root: %s", path, root);
    return toRootedPath(root, path.asFragment());
  }

  /**
   * Returns a rooted path representing {@code path} under one of the specified roots, or under the
   * file system root if it's not under any of the roots in {@code packagePathRoots}.
   */
  public static RootedPath toRootedPathMaybeUnderRoot(Path path, Iterable<Root> packagePathRoots) {
    for (Root root : packagePathRoots) {
      if (root.contains(path)) {
        return toRootedPath(root, path);
      }
    }
    return toRootedPath(Root.absoluteRoot(path.getFileSystem()), path);
  }

  public Path asPath() {
    return root.getRelative(rootRelativePath);
  }

  public Root getRoot() {
    return root;
  }

  /** Returns the path fragment relative to {@code #getRoot}. */
  public PathFragment getRootRelativePath() {
    return rootRelativePath;
  }

  @Nullable
  public RootedPath getParentDirectory() {
    PathFragment rootRelativeParentDirectory = rootRelativePath.getParentDirectory();
    if (rootRelativeParentDirectory == null) {
      return null;
    }
    return createInternal(root, rootRelativeParentDirectory);
  }

  @Override
  public boolean equals(Object obj) {
    if (this == obj) {
      return true;
    }
    if (!(obj instanceof RootedPath other)) {
      return false;
    }
    return hashCode == other.hashCode
        && root.equals(other.root)
        && rootRelativePath.equals(other.rootRelativePath);
  }

  @Override
  public int hashCode() {
    return hashCode;
  }

  @Override
  public String toString() {
    return "[" + root + "]/[" + rootRelativePath + "]";
  }

  @Override
  public int compareTo(RootedPath o) {
    return COMPARATOR.compare(this, o);
  }

  private static final Comparator<RootedPath> COMPARATOR =
      Comparator.comparing(RootedPath::getRoot).thenComparing(RootedPath::getRootRelativePath);

  @Override
  public RootedPath argument() {
    return this;
  }

  @Override
  @Nullable
  public SkyKeyInterner<?> getSkyKeyInterner() {
    return interner;
  }
}
