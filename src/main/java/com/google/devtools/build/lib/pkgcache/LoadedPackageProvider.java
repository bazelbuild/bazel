// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.syntax.Label;

/**
 * Read-only API for retrieving packages, i.e., calling this API should not result in packages being
 * loaded.
 *
 * <p><b>Concurrency</b>: Implementations should be thread-safe.
 */
// TODO(bazel-team): Skyframe doesn't really implement this - can we remove it?
public interface LoadedPackageProvider {

  /**
   * Returns a package if it was recently loaded, i.e., since the most recent cache sync. This
   * throws an exception if the package was not loaded, even if it exists on disk. Returns the
   * package even if it is in error.
   */
  Package getLoadedPackage(PackageIdentifier packageIdentifier) throws NoSuchPackageException;

  /**
   * Returns a target if it was recently loaded, i.e., since the most recent cache sync. This
   * throws an exception if the target was not loaded or not validated, even if it exists in the
   * surrounding package. If the surrounding package is in error, still attempts to retrieve the
   * target.
   */
  Target getLoadedTarget(Label label) throws NoSuchPackageException, NoSuchTargetException;

  /**
   * Returns true iff the specified target is current, i.e. a request for its label using {@link
   * #getLoadedTarget} would return the same target instance.
   */
  boolean isTargetCurrent(Target target);
}
