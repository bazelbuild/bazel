// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.packages;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.Package;
import javax.annotation.Nullable;

/** A standalone library for performing Bazel package loading. */
public interface PackageLoader {
  /**
   * Loads and returns a single package. This method is a simplified shorthand for {@link
   * #loadPackages} when just a single {@link Package} and nothing else is desired.
   */
  Package loadPackage(PackageIdentifier pkgId) throws NoSuchPackageException, InterruptedException;

  /**
   * Returns the result of loading a collection of packages. Note that the returned {@link Package}s
   * may contain errors - see {@link Package#containsErrors()} for details.
   */
  Result loadPackages(Iterable<PackageIdentifier> pkgIds) throws InterruptedException;

  /** Contains the result of package loading. */
  class Result {
    private final ImmutableMap<PackageIdentifier, PackageOrException> loadedPackages;

    Result(ImmutableMap<PackageIdentifier, PackageOrException> loadedPackages) {
      this.loadedPackages = loadedPackages;
    }

    public ImmutableMap<PackageIdentifier, PackageOrException> getLoadedPackages() {
      return loadedPackages;
    }
  }

  /** Contains a {@link Package} or the exception produced while loading it. */
  class PackageOrException {
    private final Package pkg;
    private final NoSuchPackageException exception;

    PackageOrException(@Nullable Package pkg, @Nullable NoSuchPackageException exception) {
      checkState((pkg == null) != (exception == null));
      this.pkg = pkg;
      this.exception = exception;
    }

    /**
     * @throws NoSuchPackageException if there was a non-recoverable error loading the package, e.g.
     *     an io error reading the BUILD file.
     */
    public Package get() throws NoSuchPackageException {
      if (pkg != null) {
        return pkg;
      }
      throw exception;
    }
  }
}
