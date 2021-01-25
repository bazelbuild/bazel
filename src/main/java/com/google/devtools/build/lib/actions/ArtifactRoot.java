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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Interner;
import com.google.common.collect.Interners;
import com.google.devtools.build.lib.cmdline.LabelConstants;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.starlarkbuildapi.FileRootApi;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.io.Serializable;
import java.util.Objects;
import net.starlark.java.eval.Printer;

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
    return INTERNER.intern(new ArtifactRoot(root, PathFragment.EMPTY_FRAGMENT, RootType.Source));
  }

  /**
   * Do not use except in tests and in {@link
   * com.google.devtools.build.lib.skyframe.SkyframeExecutor}.
   *
   * <p>Returns the given path as the external source root. The path should end with {@link
   * LabelConstants.EXTERNAL_REPOSITORY_LOCATION} since the external repository root is always
   * $OUTPUT_BASE/external regardless of the layout of the exec root.
   */
  public static ArtifactRoot asExternalSourceRoot(Root root) {
    Preconditions.checkArgument(
        root.asPath()
            .asFragment()
            .getParentDirectory()
            .endsWith(LabelConstants.EXTERNAL_REPOSITORY_LOCATION));
    return INTERNER.intern(
        new ArtifactRoot(root, PathFragment.EMPTY_FRAGMENT, RootType.ExternalSource));
  }

  /**
   * Constructs an ArtifactRoot given the output prefixes. (eg, "bin"), and (eg, "testlogs")
   * relative to the execRoot.
   *
   * <p>Be careful with this method - all derived roots must be within the derived artifacts tree,
   * defined in ArtifactFactory (see {@link ArtifactFactory#isDerivedArtifact(PathFragment)}).
   */
  public static ArtifactRoot asDerivedRoot(
      Path execRoot,
      boolean isMiddleman,
      boolean isExternal,
      boolean siblingRepositoryLayout,
      String... prefixes) {
    PathFragment execPath = PathFragment.EMPTY_FRAGMENT;
    for (String prefix : prefixes) {
      // Tests can have empty segments here, be gentle to them.
      if (!prefix.isEmpty()) {
        execPath = execPath.getChild(prefix);
      }
    }
    return asDerivedRoot(execRoot, isMiddleman, isExternal, siblingRepositoryLayout, execPath);
  }

  /**
   * Constructs an {@link ArtifactRoot} given the execPath, relative to the execRoot.
   *
   * <p>Be careful with this method - all derived roots must be within the derived artifacts tree,
   * defined in ArtifactFactory (see {@link ArtifactFactory#isDerivedArtifact(PathFragment)}).
   */
  public static ArtifactRoot asDerivedRoot(
      Path execRoot,
      boolean isMiddleman,
      boolean isExternal,
      boolean siblingRepositoryLayout,
      PathFragment execPath) {
    // Make sure that we are not creating a derived artifact under the execRoot.
    Preconditions.checkArgument(!execPath.isEmpty(), "empty execPath");
    Preconditions.checkArgument(
        !execPath.getSegments().contains(".."),
        "execPath: %s contains parent directory reference (..)",
        execPath);
    Path root = execRoot.getRelative(execPath);
    RootType type;
    if (siblingRepositoryLayout) {
      type =
          isMiddleman
              ? (isExternal ? RootType.ExternalMiddleman : RootType.Middleman)
              : (isExternal ? RootType.ExternalOutput : RootType.Output);
    } else {
      Preconditions.checkArgument(
          !isExternal,
          "external derived roots unavailable without --experimental_sibling_repository_layout");
      type = isMiddleman ? RootType.LegacyMiddleman : RootType.LegacyOutput;
    }
    return INTERNER.intern(new ArtifactRoot(Root.fromPath(root), execPath, type));
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec.Instantiator
  static ArtifactRoot createForSerialization(
      Root rootForSerialization, PathFragment execPath, RootType rootType) {
    if (!isOutputRootType(rootType)) {
      return INTERNER.intern(new ArtifactRoot(rootForSerialization, execPath, rootType));
    }
    return asDerivedRoot(
        rootForSerialization.asPath(),
        false,
        rootType == RootType.ExternalOutput,
        rootType != RootType.LegacyOutput,
        execPath.getSegments().toArray(new String[0]));
  }

  @AutoCodec.VisibleForSerialization
  enum RootType {
    Source,
    Output,
    Middleman,
    // Root types for external repository artifacts. Note that ExternalOutput and ExternalMiddleman
    // are activated only if --experimental_sibling_repository_layout is true, which will become
    // the default value soon. The ExternalSource type is already in effect by default.
    ExternalSource,
    ExternalOutput,
    ExternalMiddleman,
    // Legacy root types for derived artifacts. Even though they're called legacy, they are still
    // the default, but soon to be deprecated when --experimental_sibling_repository_layout is set
    // true by default. In terms of the actual root paths, there are no differences between these
    // and Output and Middleman. Their sole purpose is to embed the
    // --experimental_sibling_repository_layout flag value information in Artifacts without
    // additional storage cost.
    LegacyOutput,
    LegacyMiddleman
  }

  private final Root root;
  private final PathFragment execPath;
  private final RootType rootType;

  private ArtifactRoot(Root root, PathFragment execPath, RootType rootType) {
    this.root = Preconditions.checkNotNull(root);
    this.execPath = execPath;
    this.rootType = rootType;
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
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

  public ImmutableList<String> getComponents() {
    return execPath.getSegments();
  }

  public boolean isSourceRoot() {
    return rootType == RootType.Source || rootType == RootType.ExternalSource;
  }

  private static boolean isOutputRootType(RootType rootType) {
    return rootType == RootType.Output
        || rootType == RootType.ExternalOutput
        || rootType == RootType.LegacyOutput;
  }

  boolean isMiddlemanRoot() {
    return rootType == RootType.Middleman
        || rootType == RootType.ExternalMiddleman
        || rootType == RootType.LegacyMiddleman;
  }

  public boolean isExternal() {
    return rootType == RootType.ExternalSource
        || rootType == RootType.ExternalOutput
        || rootType == RootType.ExternalMiddleman;
  }

  public boolean isLegacy() {
    return rootType == RootType.LegacyOutput || rootType == RootType.LegacyMiddleman;
  }

  @Override
  public int compareTo(ArtifactRoot o) {
    return root.compareTo(o.root);
  }

  @Override
  public int hashCode() {
    return Objects.hash(root, execPath, rootType);
  }

  /**
   * The Root of a derived ArtifactRoot contains the exec path. In order to avoid duplicating that
   * path, and enable the Root to be serialized as a constant, we return the "exec root" Root here,
   * by stripping the exec path. That Root is likely to be serialized as a constant by {@link
   * Root.RootCodec}, saving a lot of serialized bytes on the wire.
   */
  @SuppressWarnings("unused") // Used by @AutoCodec.
  Root getRootForSerialization() {
    if (!isOutputRootType(rootType)) {
      return root;
    }
    // Find fragment of root that does not include execPath and return just that root. It is likely
    // to be serialized as a constant by RootCodec. For instance, if the original exec root was
    // /execroot, and this root was /execroot/bazel-out/bin, with execPath bazel-out/bin, then we
    // just serialize /execroot and bazel-out/bin separately.
    // We just want to strip execPath from root, but I don't know a trivial way to do that.
    PathFragment rootFragment = root.asPath().asFragment();
    return Root.fromPath(
        root.asPath()
            .getFileSystem()
            .getPath(
                rootFragment.subFragment(
                    0, rootFragment.segmentCount() - execPath.segmentCount())));
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
    return root.equals(r.root) && execPath.equals(r.execPath) && rootType == r.rootType;
  }

  @Override
  public String toString() {
    return root + (isSourceRoot() ? "[source]" : "[derived]");
  }

  @Override
  public void repr(Printer printer) {
    printer.append(isSourceRoot() ? "<source root>" : "<derived root>");
  }
}
