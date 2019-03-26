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
package com.google.devtools.build.lib.pkgcache;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.Map;

/**
 * Support for resolving {@code package/...} target patterns.
 */
public interface RecursivePackageProvider extends PackageProvider {

  /**
   * Returns the names of all the packages under a given directory.
   *
   * <p>Packages returned by this method and passed into {@link #bulkGetPackages(Iterable)} are
   * expected to return successful {@link Package} values.
   *
   * @param eventHandler any errors emitted during package lookup and loading for {@code directory}
   *     and non-excluded directories beneath it will be reported here
   * @param directory a {@link RootedPath} specifying the directory to search
   * @param blacklistedSubdirectories a set of {@link PathFragment}s specifying transitive
   *     subdirectories that have been blacklisted
   * @param excludedSubdirectories a set of {@link PathFragment}s specifying transitive
   *     subdirectories to exclude
   */
  Iterable<PathFragment> getPackagesUnderDirectory(
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      ImmutableSet<PathFragment> blacklistedSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException;

  /**
   * Returns the {@link Package} corresponding to each Package in "pkgIds". If any of the packages
   * does not exist (e.g. {@code isPackage(pkgIds)} returns false), throws a {@link
   * NoSuchPackageException}.
   *
   * <p>The returned package may contain lexical/grammatical errors, in which case <code>
   * pkg.containsErrors() == true</code>. Such packages may be missing some rules. Any rules that
   * are present may soundly be used for builds, though.
   *
   * @param pkgIds an Iterable of PackageIdentifier objects.
   * @throws NoSuchPackageException if any package could not be found.
   * @throws InterruptedException if the package loading was interrupted.
   */
  Map<PackageIdentifier, Package> bulkGetPackages(Iterable<PackageIdentifier> pkgIds)
      throws NoSuchPackageException, InterruptedException;
}
