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

import com.google.devtools.build.lib.concurrent.ThreadSafety;
import com.google.devtools.build.lib.packages.CachingPackageLocator;
import java.io.PrintStream;

/**
 * A PackageManager keeps state about loaded packages around for quick lookup, and provides
 * related functionality: Recursive package finding, loaded package checking, etc.
 */
public interface PackageManager extends PackageProvider, CachingPackageLocator {
  /**
   * Returns the package cache statistics.
   */
  PackageManagerStatistics getStatistics();

  /**
   * Dump the contents of the package manager in human-readable form.
   * Used by 'bazel dump' and the BuildTool's unexpected exception handler.
   */
  void dump(PrintStream printStream);

  /**
   * Returns the package locator used by this package manager.
   *
   * <p>If you are tempted to call {@code getPackagePath().getPathEntries().get(0)}, be warned that
   * this is probably not the value you are looking for!  Look at the methods of {@code
   * BazelRuntime} instead.
   */
  @ThreadSafety.ThreadSafe
  PathPackageLocator getPackagePath();

  /**
   * Collects statistics of the package manager since the last sync.
   */
  interface PackageManagerStatistics {
    public static final PackageManagerStatistics ZERO = new PackageManagerStatistics() {
        @Override public int getPackagesLoaded() {
          return 0;
        }

        @Override public int getPackagesLookedUp() {
          return 0;
        }

        @Override public int getCacheSize() {
          return 0;
        }
    };

    /**
     * Returns the number of packages loaded since the last sync. I.e. the cache
     * misses.
     */
    int getPackagesLoaded();

    /**
     * Returns the number of packages looked up since the last sync.
     */
    int getPackagesLookedUp();

    /**
     * Returns the number of all the packages currently loaded.
     *
     * <p>
     * Note that this method is not affected by sync(), and the packages it
     * returns are not guaranteed to be up-to-date.
     */
    int getCacheSize();
  }

  /**
   * Retrieve a target pattern parser that works with this package manager.
   */
  TargetPatternEvaluator newTargetPatternEvaluator();

  /**
   * Construct a new {@link TransitivePackageLoader}.
   */
  TransitivePackageLoader newTransitiveLoader();
}
