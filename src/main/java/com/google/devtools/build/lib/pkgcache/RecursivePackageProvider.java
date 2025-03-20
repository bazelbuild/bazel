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
import com.google.devtools.build.lib.cmdline.BatchCallback.SafeBatchCallback;
import com.google.devtools.build.lib.cmdline.IgnoredSubdirectories;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.io.ProcessPackageDirectoryException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.util.Map;

/** Support for resolving {@code package/...} target patterns. */
public interface RecursivePackageProvider extends PackageProvider {

  /**
   * Calls the supplied callback with the name of each package under a given directory, as soon as
   * that package is identified.
   *
   * <p>Packages yielded by this method and passed into {@link #bulkGetPackages(Iterable)} are
   * expected to return successful {@link Package} values.
   *
   * @param results callback invoked <em>from a single thread</em> for every eligible, loaded
   *     package as it is discovered
   * @param eventHandler any errors emitted during package lookup and loading for {@code directory}
   *     and non-excluded directories beneath it will be reported here
   * @param directory a {@link RootedPath} specifying the directory to search
   * @param ignoredSubdirectories a set of {@link PathFragment}s specifying transitive
   *     subdirectories that are ignored. {@code directory} must not be a subdirectory of any of
   *     these
   * @param excludedSubdirectories a set of {@link PathFragment}s specifying transitive
   *     subdirectories that are excluded from this traversal. Different from {@code
   *     ignoredSubdirectories} only in that these directories should not be embedded in any {@code
   *     SkyKey}s that are created during the traversal, instead filtered out later
   */
  void streamPackagesUnderDirectory(
      SafeBatchCallback<PackageIdentifier> results,
      ExtendedEventHandler eventHandler,
      RepositoryName repository,
      PathFragment directory,
      IgnoredSubdirectories ignoredSubdirectories,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws InterruptedException,
          QueryException,
          NoSuchPackageException,
          ProcessPackageDirectoryException;

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

  /**
   * A {@link RecursivePackageProvider} in terms of a map of pre-fetched packages.
   *
   * <p>Note that this class implements neither {@link #streamPackagesUnderDirectory} nor {@link
   * #bulkGetPackages}, so it can only be used for use cases that do not call either of these
   * methods. When used for target pattern resolution, it can be used to resolve SINGLE_TARGET and
   * TARGETS_IN_PACKAGE patterns by pre-fetching the corresponding packages. It can also be used to
   * resolve PATH_AS_TARGET patterns either by finding the outermost package or by pre-fetching all
   * possible packages.
   *
   * @see com.google.devtools.build.lib.cmdline.TargetPattern.Type
   */
  class PackageBackedRecursivePackageProvider implements RecursivePackageProvider {
    private final Map<PackageIdentifier, Package> packages;

    public PackageBackedRecursivePackageProvider(Map<PackageIdentifier, Package> packages) {
      this.packages = packages;
    }

    @Override
    public Package getPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName)
        throws NoSuchPackageException {
      Package pkg = packages.get(packageName);
      if (pkg == null) {
        throw new NoSuchPackageException(packageName, "");
      }
      return pkg;
    }

    @Override
    public boolean isPackage(ExtendedEventHandler eventHandler, PackageIdentifier packageName) {
      return packages.containsKey(packageName);
    }

    @Override
    public Target getTarget(ExtendedEventHandler eventHandler, Label label)
        throws NoSuchPackageException, NoSuchTargetException {
      return getPackage(eventHandler, label.getPackageIdentifier()).getTarget(label.getName());
    }

    @Override
    public void streamPackagesUnderDirectory(
        SafeBatchCallback<PackageIdentifier> results,
        ExtendedEventHandler eventHandler,
        RepositoryName repository,
        PathFragment directory,
        IgnoredSubdirectories ignoredSubdirectories,
        ImmutableSet<PathFragment> excludedSubdirectories) {
      throw new UnsupportedOperationException();
    }

    @Override
    public Map<PackageIdentifier, Package> bulkGetPackages(Iterable<PackageIdentifier> pkgIds) {
      throw new UnsupportedOperationException();
    }
  }
}
