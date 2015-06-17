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

import javax.annotation.Nullable;

/**
 * Exception indicating a failed attempt to access a package that could not
 * be read or had syntax errors.
 */
public class BuildFileContainsErrorsException extends NoSuchPackageException {

  private Package pkg;

  public BuildFileContainsErrorsException(PackageIdentifier packageIdentifier, String message) {
    super(packageIdentifier, "error loading package", message);
  }

  public BuildFileContainsErrorsException(PackageIdentifier packageIdentifier, String message,
      Throwable cause) {
    super(packageIdentifier, "error loading package", message, cause);
  }

  public BuildFileContainsErrorsException(Package pkg, String msg) {
    this(pkg.getPackageIdentifier(), msg);
    this.pkg = pkg;
  }

  @Override
  @Nullable
  public Package getPackage() {
    return pkg;
  }
}
