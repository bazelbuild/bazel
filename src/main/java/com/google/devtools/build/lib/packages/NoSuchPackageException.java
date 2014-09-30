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
 * Exception indicating an attempt to access a package which is not found, does
 * not exist, or can't be parsed into a package.
 */
public abstract class NoSuchPackageException extends NoSuchThingException {

  private final String packageName;

  public NoSuchPackageException(String packageName, String message) {
    this(packageName, "no such package", message);
  }

  public NoSuchPackageException(String packageName, String message,
      Throwable cause) {
    this(packageName, "no such package", message, cause);
  }

  protected NoSuchPackageException(String packageName, String messagePrefix, String message) {
    super(messagePrefix + " '" + packageName + "': " + message);
    this.packageName = packageName;
  }

  protected NoSuchPackageException(String packageName, String messagePrefix, String message,
      Throwable cause) {
    super(messagePrefix + " '" + packageName + "': " + message, cause);
    this.packageName = packageName;
  }

  public String getPackageName() {
    return packageName;
  }

  /**
   * Return the package if parsing completed enough to construct it. May return null.
   */
  @Nullable
  public Package getPackage() {
    return null;
  }
}
