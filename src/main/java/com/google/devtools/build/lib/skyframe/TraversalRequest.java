// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.MoreObjects;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.DirectTraversalRoot;
import com.google.devtools.build.lib.actions.FilesetTraversalParams.PackageBoundaryMode;
import com.google.devtools.build.lib.vfs.Root;
import com.google.devtools.build.skyframe.ExecutionPhaseSkyKey;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.errorprone.annotations.ForOverride;

/** A request for {@link RecursiveFilesystemTraversalFunction}. */
public abstract class TraversalRequest implements ExecutionPhaseSkyKey {

  /** The path to start the traversal from; may be a file, a directory or a symlink. */
  @VisibleForTesting
  public abstract DirectTraversalRoot root();

  /**
   * Whether the path is in the output tree.
   *
   * <p>Such paths and all their subdirectories are assumed not to define packages, so package
   * lookup for them is skipped.
   */
  protected abstract boolean isRootGenerated();

  /** Whether the traversal should descend into directories that are roots of subpackages. */
  protected abstract PackageBoundaryMode crossPkgBoundaries();

  /** Whether Fileset assumes that output artifacts are regular files. */
  protected abstract boolean strictOutputFiles();

  /**
   * Whether to skip checking if the root (if it's a directory) contains a BUILD file.
   *
   * <p>Such directories are not considered to be packages when this flag is true. This needs to be
   * true in order to traverse directories of packages, but should be false for <i>their</i>
   * subdirectories.
   */
  protected abstract boolean skipTestingForSubpackage();

  /**
   * Whether to emit nodes for empty directories.
   *
   * <p>If this returns false, empty directories will not be represented in the result of the
   * traversal.
   */
  protected abstract boolean emitEmptyDirectoryNodes();

  /**
   * Returns information to be attached to any error messages that may be reported.
   *
   * <p>This is purely informational and is not considered in equality.
   */
  protected abstract String errorInfo();

  /**
   * Creates a new traversal request identical to this one except with the given new values for
   * {@link #root} and {@link #skipTestingForSubpackage}.
   */
  @ForOverride
  protected abstract TraversalRequest duplicateWithOverrides(
      DirectTraversalRoot root, boolean skipTestingForSubpackage);

  /** Creates a new request to traverse a child element in the current directory (the root). */
  final TraversalRequest forChildEntry(String child) {
    DirectTraversalRoot newTraversalRoot =
        DirectTraversalRoot.forRootAndPath(
            root().getRootPart(), root().getRelativePart().getRelative(child));
    return duplicateWithOverrides(newTraversalRoot, /*skipTestingForSubpackage=*/ false);
  }

  /**
   * Creates a new request for a changed root.
   *
   * <p>This method can be used when a package is found out to be under a different root path than
   * originally assumed.
   */
  final TraversalRequest forChangedRootPath(Root newRoot) {
    DirectTraversalRoot newTraversalRoot =
        DirectTraversalRoot.forRootAndPath(newRoot, root().getRelativePart());
    return duplicateWithOverrides(newTraversalRoot, skipTestingForSubpackage());
  }

  @Override
  public final SkyFunctionName functionName() {
    return SkyFunctions.RECURSIVE_FILESYSTEM_TRAVERSAL;
  }

  @Override
  public final String toString() {
    return MoreObjects.toStringHelper(this)
        .add("root", root())
        .add("isRootGenerated", isRootGenerated())
        .add("crossPkgBoundaries", crossPkgBoundaries())
        .add("strictOutputFiles", strictOutputFiles())
        .add("skipTestingForSubpackage", skipTestingForSubpackage())
        .add("errorInfo", errorInfo())
        .toString();
  }
}
