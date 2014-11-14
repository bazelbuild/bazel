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
package com.google.devtools.build.lib.util;

/**
 * An operating system.
 */
public enum OS {
  DARWIN,
  LINUX,
  WINDOWS,
  UNKNOWN;

  /**
   * The current operating system.
   */
  public static OS getCurrent() {
    return HOST_SYSTEM;
  }
  // We inject a the OS name through blaze.os, so we can have
  // some coverage for Windows specific code on Linux.
  private static String getOsName() {
    String override = System.getProperty("blaze.os");
    return override == null ? System.getProperty("os.name") : override;
  }

  private static final OS HOST_SYSTEM =
      "Mac OS X".equals(getOsName()) ? OS.DARWIN : (
      "Linux".equals(getOsName()) ? OS.LINUX : (
          getOsName().contains("Windows") ? OS.WINDOWS : OS.UNKNOWN));
}

