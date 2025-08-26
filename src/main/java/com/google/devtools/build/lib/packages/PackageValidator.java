// Copyright 2020 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.PackageLoadingListener.Metrics;
import com.google.devtools.build.lib.util.DetailedExitCode;

/** Provides loaded-package validation functionality. */
public interface PackageValidator {

  /** No-op implementation of {@link PackageValidator}. */
  PackageValidator NOOP_VALIDATOR = (pkg, metrics, eventHandler) -> {};

  /** Thrown when a package is deemed invalid. */
  class InvalidPackageException extends NoSuchPackageException {
    public InvalidPackageException(PackageIdentifier pkgId, String message) {
      super(pkgId, message);
    }

    public InvalidPackageException(
        PackageIdentifier pkgId, String message, DetailedExitCode detailedExitCode) {
      super(pkgId, message, detailedExitCode);
    }
  }

  /** Thrown when a package piece is deemed invalid. */
  class InvalidPackagePieceException extends NoSuchPackagePieceException {
    public InvalidPackagePieceException(PackagePieceIdentifier packagePieceId, String message) {
      super(packagePieceId, message);
    }

    public InvalidPackagePieceException(
        PackagePieceIdentifier packagePieceId, String message, DetailedExitCode detailedExitCode) {
      super(packagePieceId, message, detailedExitCode);
    }
  }

  default Package.Builder.PackageLimits getPackageLimits() {
    return Package.Builder.PackageLimits.DEFAULTS;
  }

  /**
   * Validates a loaded package. Throws {@link InvalidPackageException} if the package is deemed
   * invalid.
   */
  void validate(Package pkg, Metrics metrics, ExtendedEventHandler eventHandler)
      throws InvalidPackageException;
}
