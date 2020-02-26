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

/** Provides loaded-package validation functionality. */
public interface PackageValidator {

  /** No-op implementation of {@link PackageValidator}. */
  PackageValidator NOOP_VALIDATOR = pkg -> {};

  /** Thrown when a package is deemed invalid. */
  class InvalidPackageException extends NoSuchPackageException {
    public InvalidPackageException(PackageIdentifier pkgId, String message) {
      super(pkgId, message);
    }
  }

  /**
   * Validates a loaded package. Throws {@link InvalidPackageException} if the package is deemed
   * invalid.
   */
  void validate(Package pkg) throws InvalidPackageException;
}
