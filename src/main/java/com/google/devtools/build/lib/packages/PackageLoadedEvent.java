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

/**
 * An event that is fired after a package is loaded.
 */
public final class PackageLoadedEvent {
  private final String packageName;
  private final long timeInMillis;
  private final boolean reloading;
  private final boolean successful;

  public PackageLoadedEvent(String packageName, long timeInMillis, boolean reloading,
      boolean successful) {
    this.packageName = packageName;
    this.timeInMillis = timeInMillis;
    this.reloading = reloading;
    this.successful = successful;
  }

  /**
   * Returns the package name.
   */
  public String getPackageName() {
    return packageName;
  }

  /**
   * Returns time which was spent to load a package.
   */
  public long getTimeInMillis() {
    return timeInMillis;
  }

  /**
   * Returns true if package had been loaded before.
   */
  public boolean isReloading() {
    return reloading;
  }

  /**
   * Returns true if package was loaded successfully.
   */
  public boolean isSuccessful() {
    return successful;
  }
}
