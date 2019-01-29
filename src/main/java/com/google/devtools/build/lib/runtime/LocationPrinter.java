// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.Root;
import java.util.concurrent.atomic.AtomicReference;
import javax.annotation.Nullable;

class LocationPrinter {
  private final boolean attemptToPrintRelativePaths;
  @Nullable private final PathFragment workspacePathFragment;
  private final AtomicReference<ImmutableList<Root>> packagePathRootsRef =
      new AtomicReference<>(ImmutableList.of());

  LocationPrinter(
      boolean attemptToPrintRelativePaths,
      @Nullable PathFragment workspacePathFragment) {
    this.attemptToPrintRelativePaths = attemptToPrintRelativePaths;
    this.workspacePathFragment = workspacePathFragment;
  }

  void packageLocatorCreated(PathPackageLocator packageLocator) {
    packagePathRootsRef.set(packageLocator.getPathEntries());
  }

  String getLocationString(Location location) {
    return attemptToPrintRelativePaths
        ? getRelativeLocationString(location, workspacePathFragment, packagePathRootsRef.get())
        : location.toString();
  }

  @VisibleForTesting
  static String getRelativeLocationString(
      Location location,
      @Nullable PathFragment workspacePathFragment,
      ImmutableList<Root> packagePathRoots) {
    PathFragment relativePathToUse = null;
    PathFragment locationPathFragment = location.getPath();
    if (locationPathFragment.isAbsolute()) {
      if (workspacePathFragment != null && locationPathFragment.startsWith(workspacePathFragment)) {
        relativePathToUse = locationPathFragment.relativeTo(workspacePathFragment);
      } else {
        for (Root packagePathRoot : packagePathRoots) {
          if (packagePathRoot.contains(locationPathFragment)) {
            relativePathToUse = packagePathRoot.relativize(locationPathFragment);
            break;
          }
        }
      }
    }
    return relativePathToUse == null ? location.print() : location.printWithPath(relativePathToUse);
  }
}
