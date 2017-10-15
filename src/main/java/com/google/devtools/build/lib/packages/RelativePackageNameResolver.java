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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Resolves relative package names to absolute ones. Handles the absolute
 * package path marker ("//") and uplevel references ("..").
 */
public class RelativePackageNameResolver {
  private final PathFragment offset;
  private final boolean discardBuild;

  /**
   * @param offset the base package path used to resolve relative paths
   * @param discardBuild if true, discards the last package path segment if
   *        it is called "BUILD"
   */
  public RelativePackageNameResolver(PathFragment offset, boolean discardBuild) {
    Preconditions.checkArgument(!offset.containsUplevelReferences(),
        "offset should not contain uplevel references");

    this.offset = offset;
    this.discardBuild = discardBuild;
  }

  /**
   * Resolves the given package name with respect to the offset given in the
   * constructor.
   *
   * @param pkg the relative package name to be resolved
   * @return the absolute package name
   * @throws InvalidPackageNameException if the package name cannot be resolved
   *         (only syntactic checks are done -- it is not checked if the package
   *         really exists or not)
   */
  public String resolve(String pkg) throws InvalidPackageNameException {
    boolean isAbsolute;
    String relativePkg;

    if (pkg.startsWith("//")) {
      isAbsolute = true;
      relativePkg = pkg.substring(2);
    } else if (pkg.startsWith("/")) {
      throw new InvalidPackageNameException(
          PackageIdentifier.createInMainRepo(pkg),
          "package name cannot start with a single slash");
    } else {
      isAbsolute = false;
      relativePkg = pkg;
    }

    PathFragment relative = PathFragment.create(relativePkg);

    if (discardBuild && relative.getBaseName().equals("BUILD")) {
      relative = relative.getParentDirectory();
    }

    PathFragment result = isAbsolute ? relative : offset.getRelative(relative);
    result = result.normalize();
    if (result.containsUplevelReferences()) {
      throw new InvalidPackageNameException(
          PackageIdentifier.createInMainRepo(pkg),
          "package name contains too many '..' segments");
    }

    return result.getPathString();
  }
}
