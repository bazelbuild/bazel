// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;

/**
 * A {@link RecursivePackageProvider} backed by an {@link Environment}. Its methods
 * may throw {@link MissingDepException} if the package values this depends on haven't been
 * calculated and added to its environment.
 */
public final class EnvironmentBackedRecursivePackageProvider implements RecursivePackageProvider {

  private final Environment env;

  public EnvironmentBackedRecursivePackageProvider(Environment env) {
    this.env = env;
  }

  @Override
  public Package getPackage(EventHandler eventHandler, PackageIdentifier packageName)
      throws NoSuchPackageException, MissingDepException {
    SkyKey pkgKey = PackageValue.key(packageName);
    Package pkg;
    try {
      PackageValue pkgValue =
          (PackageValue) env.getValueOrThrow(pkgKey, NoSuchPackageException.class);
      if (pkgValue == null) {
        throw new MissingDepException();
      }
      pkg = pkgValue.getPackage();
    } catch (NoSuchPackageException e) {
      pkg = e.getPackage();
      if (pkg == null) {
        throw e;
      }
    }
    return pkg;
  }

  @Override
  public boolean isPackage(EventHandler eventHandler, String packageName)
      throws MissingDepException {
    SkyKey packageLookupKey = PackageLookupValue.key(new PathFragment(packageName));
    try {
      PackageLookupValue packageLookupValue =
          (PackageLookupValue) env.getValueOrThrow(packageLookupKey, NoSuchPackageException.class,
              InconsistentFilesystemException.class);
      if (packageLookupValue == null) {
        throw new MissingDepException();
      }
      return packageLookupValue.packageExists();
    } catch (NoSuchPackageException | InconsistentFilesystemException e) {
      eventHandler.handle(Event.error(e.getMessage()));
      return false;
    }
  }

  @Override
  public Iterable<PathFragment> getPackagesUnderDirectory(RootedPath directory,
      ImmutableSet<PathFragment> excludedSubdirectories)
      throws MissingDepException {
    PathFragment rootedPathFragment = directory.getRelativePath();
    PathFragment.checkAllPathsAreUnder(excludedSubdirectories, rootedPathFragment);
    RecursivePkgValue lookup = (RecursivePkgValue) env.getValue(
        RecursivePkgValue.key(directory, excludedSubdirectories));
    if (lookup == null) {
      // Typically a null value from Environment.getValue(k) means that either the key k is missing
      // a dependency or an exception was thrown during evaluation of k. Here, if this getValue
      // call returns null in a keep_going build, it can only mean a missing dependency, because
      // RecursivePkgFunction#compute never throws.
      // In a nokeep_going build, a lower-level exception that RecursivePkgFunction ignored may
      // bubble up to here, but we ignore it and depend on the top-level caller to be flexible in
      // the exception types it can accept.
      throw new MissingDepException();
    }
    // TODO(bazel-team): Make RecursivePkgValue return NestedSet<PathFragment> so this transform is
    // unnecessary.
    return Iterables.transform(lookup.getPackages(), PathFragment.TO_PATH_FRAGMENT);
  }

  @Override
  public Target getTarget(EventHandler eventHandler, Label label) throws NoSuchPackageException,
      NoSuchTargetException, MissingDepException {
    return getPackage(eventHandler, label.getPackageIdentifier()).getTarget(label.getName());
  }

  /**
   * Indicates that a missing dependency is needed before target parsing can proceed. Currently
   * used only in skyframe to notify the framework of missing dependencies. Caught by the compute
   * method in {@link com.google.devtools.build.lib.skyframe.TargetPatternFunction}, which then
   * returns null in accordance with the skyframe missing dependency policy.
   */
  static class MissingDepException extends RuntimeException {
  }
}
