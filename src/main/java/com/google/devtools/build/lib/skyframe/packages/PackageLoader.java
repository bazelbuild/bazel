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
   * Returns a {@link Package} instance, if any, representing the Blaze package specified by {@code
   * pkgId}. Note that the returned {@link Package} instance may be in error (see {@link
   * Package#containsErrors}), e.g. if there was syntax error in the package's BUILD file.
   *
   * @throws InterruptedException if the package loading was interrupted.
   * @throws NoSuchPackageException if there was a non-recoverable error loading the package, e.g.
   *     an io error reading the BUILD file.
   */
  Package loadPackage(PackageIdentifier pkgId) throws NoSuchPackageException, InterruptedException;

  /**
   * Returns {@link Package} instances, if any, representing Blaze packages specified by {@code
   * pkgIds}. Note that returned {@link Package} instances may be in error (see {@link
   * Package#containsErrors}), e.g. if there was syntax error in the package's BUILD file.
   */
  ImmutableMap<PackageIdentifier, PackageOrException> loadPackages(
      Iterable<? extends PackageIdentifier> pkgIds) throws InterruptedException;

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
