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

import com.google.common.base.Preconditions;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.PackageIdentifier;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.RecursivePackageProvider;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.WalkableGraph;

import java.util.Collections;

/** A {@link RecursivePackageProvider} backed by a {@link WalkableGraph}. */
public final class GraphBackedRecursivePackageProvider implements RecursivePackageProvider {

  private final WalkableGraph graph;

  public GraphBackedRecursivePackageProvider(WalkableGraph graph) {
    this.graph = graph;
  }

  @Override
  public Package getPackage(EventHandler eventHandler, PackageIdentifier packageName)
      throws NoSuchPackageException {
    SkyKey pkgKey = PackageValue.key(packageName);

    PackageValue pkgValue;
    if (graph.exists(pkgKey)) {
      pkgValue = (PackageValue) graph.getValue(pkgKey);
      if (pkgValue == null) {
        throw (NoSuchPackageException) Preconditions.checkNotNull(graph.getException(pkgKey));
      }
    } else {
      // If the package key does not exist in the graph, then it must not correspond to any package,
      // because the SkyQuery environment has already loaded the universe.
      throw new BuildFileNotFoundException(packageName.toString(),
          "BUILD file not found on package path");
    }
    return pkgValue.getPackage();
  }

  @Override
  public boolean isPackage(EventHandler eventHandler, String packageName) {
    SkyKey packageLookupKey = PackageLookupValue.key(new PathFragment(packageName));
    if (!graph.exists(packageLookupKey)) {
      // If the package lookup key does not exist in the graph, then it must not correspond to any
      // package, because the SkyQuery environment has already loaded the universe.
      return false;
    }
    PackageLookupValue packageLookupValue = (PackageLookupValue) graph.getValue(packageLookupKey);
    if (packageLookupValue == null) {
      Exception exception = Preconditions.checkNotNull(graph.getException(packageLookupKey),
          "During package lookup for '%s', got null for exception", packageName);
      if (exception instanceof NoSuchPackageException
          || exception instanceof InconsistentFilesystemException) {
        eventHandler.handle(Event.error(exception.getMessage()));
        return false;
      } else {
        throw new IllegalStateException("During package lookup for '" + packageName
            + "', got unexpected exception type", exception);
      }
    }
    return packageLookupValue.packageExists();
  }

  @Override
  public Iterable<PathFragment> getPackagesUnderDirectory(RootedPath directory) {
    SkyKey recursivePackageKey = RecursivePkgValue.key(directory);
    if (!graph.exists(recursivePackageKey)) {
      // If the recursive package key does not exist in the graph, then it must not correspond to
      // any directory transitively containing packages, because the SkyQuery environment has
      // already loaded the universe.
      return Collections.emptyList();
    }
    // If the recursive package key exists in the graph, then it must have a value and must not
    // have an exception, because RecursivePkgFunction#compute never throws.
    RecursivePkgValue lookup =
        (RecursivePkgValue) Preconditions.checkNotNull(graph.getValue(recursivePackageKey));
    // TODO(bazel-team): Make RecursivePkgValue return NestedSet<PathFragment> so this transform is
    // unnecessary.
    return Iterables.transform(lookup.getPackages(), PathFragment.TO_PATH_FRAGMENT);
  }

  @Override
  public Target getTarget(EventHandler eventHandler, Label label)
      throws NoSuchPackageException, NoSuchTargetException {
    return getPackage(eventHandler, label.getPackageIdentifier()).getTarget(label.getName());
  }
}
