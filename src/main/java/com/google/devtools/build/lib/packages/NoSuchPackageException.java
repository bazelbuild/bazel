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

package com.google.devtools.build.lib.packages;

import com.google.devtools.build.lib.cmdline.PackageIdentifier;

/**
 * Exception indicating an attempt to access a package which is not found, does
 * not exist, or can't be parsed into a package.
 */
public abstract class NoSuchPackageException extends NoSuchThingException {

  private final com.google.devtools.build.lib.cmdline.PackageIdentifier packageId;

  public NoSuchPackageException(PackageIdentifier packageId, String message) {
    this(packageId, "no such package", message);
  }

  public NoSuchPackageException(PackageIdentifier packageId, String message,
      Throwable cause) {
    this(packageId, "no such package", message, cause);
  }

  protected NoSuchPackageException(
      PackageIdentifier packageId, String messagePrefix, String message) {
    super(messagePrefix + " '" + packageId + "': " + message);
    this.packageId = packageId;
  }

  protected NoSuchPackageException(PackageIdentifier packageId, String messagePrefix,
      String message, Throwable cause) {
    super(messagePrefix + " '" + packageId + "': " + message, cause);
    this.packageId = packageId;
  }

  public PackageIdentifier getPackageId() {
    return packageId;
  }
}
