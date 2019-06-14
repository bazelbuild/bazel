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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skylarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkPrinter;
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
@AutoCodec
@Immutable
public final class ArtifactRoot implements Comparable<ArtifactRoot>, Serializable, FileRootApi {
  private static final Interner<ArtifactRoot> INTERNER = Interners.newWeakInterner();
  /**
   * Do not use except in tests and in {@link
   * com.google.devtools.build.lib.skyframe.SkyframeExecutor}.
   *
   * <p>Returns the given path as a source root. The path may not be {@code null}.
   */
  public static ArtifactRoot asSourceRoot(Root root) {
    return new ArtifactRoot(root, PathFragment.EMPTY_FRAGMENT, RootType.Source);
  }

  /**
   * Constructs an ArtifactRoot given the output prefixes. (eg, "bin"), and (eg, "testlogs")
   * relative to the execRoot.
   *
   * <p>Be careful with this method - all derived roots must be registered with the artifact factory
   * before the analysis phase.
   */
  public static ArtifactRoot asDerivedRoot(Path execRoot, PathFragment... prefixes) {
    Path root = execRoot;
    for (PathFragment prefix : prefixes) {
      root = root.getRelative(prefix);
    }
    Preconditions.checkArgument(root.startsWith(execRoot));
    Preconditions.checkArgument(!root.equals(execRoot));
    PathFragment execPath = root.relativeTo(execRoot);
    return INTERNER.intern(
        new ArtifactRoot(
            Root.fromPath(root), execPath, RootType.Output, ImmutableList.copyOf(prefixes)));
  }

  /**
   * Returns the given path as a derived root, relative to the given exec root. The root must be a
   * proper sub-directory of the exec root (i.e. not equal). Neither may be {@code null}.
   *
   * <p>Be careful with this method - all derived roots must be registered with the artifact factory
   * before the analysis phase.
   */
  @VisibleForTesting
  public static ArtifactRoot asDerivedRoot(Path execRoot, Path root) {
    return asDerivedRoot(execRoot, root.relativeTo(execRoot));
  }

  public static ArtifactRoot middlemanRoot(Path execRoot, Path outputDir) {
    Path root = outputDir.getRelative("internal");
    Preconditions.checkArgument(root.startsWith(execRoot));
    Preconditions.checkArgument(!root.equals(execRoot));
    PathFragment execPath = root.relativeTo(execRoot);
    return INTERNER.intern(new ArtifactRoot(Root.fromPath(root), execPath, RootType.Middleman));
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static ArtifactRoot createForSerialization(
      Root root, PathFragment execPath, RootType rootType, ImmutableList<PathFragment> components) {
    return INTERNER.intern(new ArtifactRoot(root, execPath, rootType, components));
  }

  @AutoCodec.VisibleForSerialization
  enum RootType {
    Source,
    Output,
    Middleman
  }

  private final Root root;
  private final PathFragment execPath;
  private final RootType rootType;
  @Nullable private final ImmutableList<PathFragment> components;

  private ArtifactRoot(
      Root root, PathFragment execPath, RootType rootType, ImmutableList<PathFragment> components) {
    this.root = Preconditions.checkNotNull(root);
    this.execPath = execPath;
    this.rootType = rootType;
    this.components = components;
  }

  private ArtifactRoot(Root root, PathFragment execPath, RootType rootType) {
    this(root, execPath, rootType, /* components= */ null);
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

  @Override
  public String getExecPathString() {
    return getExecPath().getPathString();
  }

  public ImmutableList<PathFragment> getComponents() {
    return components;
  }

  public boolean isSourceRoot() {
    return rootType == RootType.Source;
  }

  boolean isMiddlemanRoot() {
    return rootType == RootType.Middleman;
  }

  @Override
  public int compareTo(ArtifactRoot o) {
    return root.compareTo(o.root);
  }

  @Override
  public int hashCode() {
    return Objects.hash(root, execPath, rootType, components);
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
    return root.equals(r.root)
        && execPath.equals(r.execPath)
        && rootType == r.rootType
        && Objects.equals(components, r.components);
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
