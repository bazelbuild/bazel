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
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.io.IOException;

/**
 * Exception indicating a failed attempt to access a package that could not be read or had syntax
 * errors.
 */
public class BuildFileContainsErrorsException extends NoSuchPackageException {

  public BuildFileContainsErrorsException(PackageIdentifier packageIdentifier) {
    super(
        packageIdentifier,
        String.format(
            "Package '%s' contains errors",
            packageIdentifier.getPackageFragment().getPathString()));
  }

  public BuildFileContainsErrorsException(PackageIdentifier packageIdentifier, String message) {
    super(packageIdentifier, message);
  }

  public BuildFileContainsErrorsException(
      PackageIdentifier packageIdentifier, String message, IOException cause) {
    super(packageIdentifier, message, cause);
  }

  public BuildFileContainsErrorsException(
      PackageIdentifier packageIdentifier, String message, DetailedExitCode detailedExitCode) {
    super(packageIdentifier, message, detailedExitCode);
  }

  public BuildFileContainsErrorsException(
      PackageIdentifier packageIdentifier,
      String message,
      IOException cause,
      DetailedExitCode detailedExitCode) {
    super(packageIdentifier, message, cause, detailedExitCode);
  }

  @Override
  public String getMessage() {
    return String.format("error loading package '%s': %s", getPackageId(), getRawMessage());
  }
}
