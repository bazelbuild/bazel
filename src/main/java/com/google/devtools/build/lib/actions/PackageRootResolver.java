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

import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.Map;

import javax.annotation.Nullable;

/**
 * Represents logic that evaluates the root of the package containing path.
 */
public interface PackageRootResolver {

  /**
   * Returns mapping from execPath to Root. Root will be null if the path has no containing
   * package.
   *
   * @param execPaths the paths to find {@link Root}s for. The search for a containing package will
   *    start with the path's parent directory, since the path is assumed to be a file.
   * @return mappings from {@code execPath} to {@link Root}, or null if for some reason we
   *    cannot determine the result at this time (such as when used within a SkyFunction)
   * @throws PackageRootResolutionException if unable to determine package roots or lack thereof,
   *    typically caused by exceptions encountered while attempting to locate BUILD files
   */
  @Nullable
  Map<PathFragment, Root> findPackageRootsForFiles(Iterable<PathFragment> execPaths)
      throws PackageRootResolutionException;

  /**
   * Returns mapping from execPath to Root. Root will be null if the path has no containing
   * package. Unlike {@link #findPackageRootsForFiles(Iterable)}, this function allows directories
   * in the list of exec paths.
   *
   * @param execPaths the paths to find {@link Root}s for. The search for a containing package will
   *    start with the path's parent directory, since the path is assumed to be a file.
   * @return mappings from {@code execPath} to {@link Root}, or null if for some reason we
   *    cannot determine the result at this time (such as when used within a SkyFunction)
   * @throws PackageRootResolutionException if unable to determine package roots or lack thereof,
   *    typically caused by exceptions encountered while attempting to locate BUILD files
   */
  // TODO(bazel-team): Remove this once we don't need to find package roots for directories.
  @Nullable
  Map<PathFragment, Root> findPackageRoots(Iterable<PathFragment> execPaths)
      throws PackageRootResolutionException;
}
