// Copyright 2017 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Root;
import javax.annotation.Nullable;

/**
 * An interface that provides information about package's source roots, that is, the paths on disk
 * that their BUILD files can be found at. Usually this information is not needed except for when
 * planting the symlink forest in the exec root, and when resolving source exec paths to artifacts
 * in an {@link ArtifactResolver}.
 */
public interface PackageRoots {

  /**
   * Returns a map from {@link PackageIdentifier} to {@link Path}. Should only be needed for
   * {@linkplain com.google.devtools.build.lib.buildtool.SymlinkForest planting the symlink forest}.
   *
   * <p>If {@link PackageIdentifier#EMPTY_PACKAGE_ID} is present, then all top-level path entries
   * under the corresponding root are to be linked.
   */
  ImmutableMap<PackageIdentifier, Root> getPackageRootsMap();

  PackageRootLookup getPackageRootLookup();

  /** Interface for getting the source root of a package, given its {@link PackageIdentifier}. */
  interface PackageRootLookup {
    /**
     * Returns the {@link ArtifactRoot} of a package, given its {@link PackageIdentifier}. May be
     * null if the given {@code packageIdentifier} does not correspond to a package in this build.
     * However, if there is a unique source root for all packages, this may return that root even if
     * the {@code packageIdentifier} given does not correspond to any packages.
     */
    @Nullable
    Root getRootForPackage(PackageIdentifier packageIdentifier);
  }
}
