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
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.vfs.Path;

/**
 * CachingPackageLocator implementations locate a package by its name.
 *
 * <p> Similar to #pkgcache.PathPackageLocator, but implementations are required
 * to cache results and handle deleted packages.
 *
 * <p> This interface exists for two reasons: (1) to avoid creating a bad dependency edge from the
 * PythonPreprocessor to lib.pkgcache ("dependency injection") and (2) to allow Skyframe to use
 * pieces of legacy code while still updating the Skyframe node graph.
 */
public interface CachingPackageLocator {

  /**
   * Returns path of BUILD file for specified package iff the specified package exists, null
   * otherwise (e.g. invalid package name, no build file, or package has been deleted via
   * --deleted_packages)..
   *
   * <p> The package's root directory may be computed by calling getParentFile().
   *
   * <p> Instances of this interface are required to cache the results.
   *
   * <p> This method must be thread-safe.
   */
  @ThreadSafe
  Path getBuildFileForPackage(PackageIdentifier packageName);

}
