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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
import com.google.devtools.build.lib.skylarkinterface.SkylarkValue;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.Serializable;
import java.util.Objects;
import javax.annotation.Nullable;

/**
 * A root for an artifact. The roots are the directories containing artifacts, and they are mapped
 * together into a single directory tree to form the execution environment. There are two kinds of
 * roots, source roots and derived roots. Source roots correspond to entries of the package path,
 * and they can be anywhere on disk. Derived roots correspond to output directories; there are
 * generally different output directories for different configurations, and different types of
 * output (bin, genfiles, includes, etc.).
 *
 * <p>When mapping the roots into a single directory tree, the source roots are merged, such that
 * each package is accessed in its entirety from a single source root. The package cache is
 * responsible for determining that mapping. The derived roots, on the other hand, have to be
 * distinct. (It is currently allowed to have a derived root that is the prefix of another one.)
 *
 * <p>The derived roots must have paths that point inside the exec root, i.e. below the directory
 * that is the root of the merged directory tree.
 */
@SkylarkModule(
  name = "root",
  category = SkylarkModuleCategory.BUILTIN,
  doc =
      "A root for files. The roots are the directories containing files, and they are mapped "
          + "together into a single directory tree to form the execution environment."
)
@Immutable
public final class ArtifactRoot implements Comparable<ArtifactRoot>, Serializable, SkylarkValue {

  // This must always be consistent with Package.getSourceRoot; otherwise computing source roots
  // from exec paths does not work, which can break the action cache for input-discovering actions.
  public static ArtifactRoot computeSourceRoot(Root packageRoot, RepositoryName repository) {
    if (repository.isMain()) {
      return asSourceRoot(packageRoot);
    } else {
      Path actualRootPath = packageRoot.asPath();
      for (int i = 0; i < repository.getSourceRoot().segmentCount(); i++) {
        actualRootPath = actualRootPath.getParentDirectory();
      }
      return asSourceRoot(Root.fromPath(actualRootPath));
    }
  }

  /** Returns the given path as a source root. The path may not be {@code null}. */
  public static ArtifactRoot asSourceRoot(Root path) {
    return new ArtifactRoot(null, PathFragment.EMPTY_FRAGMENT, path);
  }

  /**
   * Returns the given path as a derived root, relative to the given exec root. The root must be a
   * proper sub-directory of the exec root (i.e. not equal). Neither may be {@code null}.
   *
   * <p>Be careful with this method - all derived roots must be registered with the artifact factory
   * before the analysis phase.
   */
  public static ArtifactRoot asDerivedRoot(Path execRoot, Path root) {
    Preconditions.checkArgument(root.startsWith(execRoot));
    Preconditions.checkArgument(!root.equals(execRoot));
    PathFragment execPath = root.relativeTo(execRoot);
    return new ArtifactRoot(execRoot, execPath, Root.fromPath(root));
  }

  public static ArtifactRoot middlemanRoot(Path execRoot, Path outputDir) {
    Path root = outputDir.getRelative("internal");
    Preconditions.checkArgument(root.startsWith(execRoot));
    Preconditions.checkArgument(!root.equals(execRoot));
    PathFragment execPath = root.relativeTo(execRoot);
    return new ArtifactRoot(execRoot, execPath, Root.fromPath(root), true);
  }

  @Nullable private final Path execRoot;
  private final Root root;
  private final boolean isMiddlemanRoot;
  private final PathFragment execPath;

  private ArtifactRoot(
      @Nullable Path execRoot, PathFragment execPath, Root root, boolean isMiddlemanRoot) {
    this.execRoot = execRoot;
    this.root = Preconditions.checkNotNull(root);
    this.isMiddlemanRoot = isMiddlemanRoot;
    this.execPath = execPath;
  }

  private ArtifactRoot(@Nullable Path execRoot, PathFragment execPath, Root root) {
    this(execRoot, execPath, root, false);
  }

  public Root getRoot() {
    return root;
  }

  /**
   * Returns the path fragment from the exec root to the actual root. For source roots, this returns
   * the empty fragment.
   */
  public PathFragment getExecPath() {
    return execPath;
  }

  @SkylarkCallable(name = "path", structField = true,
      doc = "Returns the relative path from the exec root to the actual root.")
  public String getExecPathString() {
    return getExecPath().getPathString();
  }

  @Nullable
  public Path getExecRoot() {
    return execRoot;
  }

  public boolean isSourceRoot() {
    return execRoot == null;
  }

  boolean isMiddlemanRoot() {
    return isMiddlemanRoot;
  }

  @Override
  public int compareTo(ArtifactRoot o) {
    return root.compareTo(o.root);
  }

  @Override
  public int hashCode() {
    return Objects.hash(execRoot, root.hashCode());
  }

  @Override
  public boolean equals(Object o) {
    if (o == this) {
      return true;
    }
    if (!(o instanceof ArtifactRoot)) {
      return false;
    }
    ArtifactRoot r = (ArtifactRoot) o;
    return root.equals(r.root) && Objects.equals(execRoot, r.execRoot);
  }

  @Override
  public String toString() {
    return root + (isSourceRoot() ? "[source]" : "[derived]");
  }

  @Override
  public void repr(SkylarkPrinter printer) {
    printer.append(isSourceRoot() ? "<source root>" : "<derived root>");
  }
}
