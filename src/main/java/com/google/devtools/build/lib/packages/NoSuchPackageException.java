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
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/**
 * Exception indicating an attempt to access a package which is not found, does not exist, or can't
 * be parsed into a package.
 *
 * <p>Prefer using more-specific subclasses, when appropriate.
 */
public class NoSuchPackageException extends NoSuchThingException implements DetailedException {

  // TODO(b/138456686): Remove Nullable and add Precondition#checkNotNull in constructor when all
  //  subclasses are instantiated with DetailedExitCode.
  @Nullable private final DetailedExitCode detailedExitCode;
  private final PackageIdentifier packageId;

  public NoSuchPackageException(PackageIdentifier packageId, String message) {
    super(message);
    this.packageId = packageId;
    this.detailedExitCode = null;
  }

  public NoSuchPackageException(PackageIdentifier packageId, String message, Exception cause) {
    super(message, cause);
    this.packageId = packageId;
    this.detailedExitCode = null;
  }

  public NoSuchPackageException(
      PackageIdentifier packageId, String message, DetailedExitCode detailedExitCode) {
    super(message);
    this.packageId = packageId;
    this.detailedExitCode = detailedExitCode;
  }

  public NoSuchPackageException(
      PackageIdentifier packageId,
      String message,
      Exception cause,
      DetailedExitCode detailedExitCode) {
    super(message, cause);
    this.packageId = packageId;
    this.detailedExitCode = detailedExitCode;
  }

  public PackageIdentifier getPackageId() {
    return packageId;
  }

  public String getRawMessage() {
    return super.getMessage();
  }

  @Override
  public String getMessage() {
    return String.format("%s '%s': %s", "no such package", packageId, getRawMessage());
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
