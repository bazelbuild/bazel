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

package com.google.devtools.build.lib.util;

/**
 * Operating system-specific utilities.
 */
public final class OsUtils {

  private static final String EXECUTABLE_EXTENSION = executableExtension(OS.getCurrent());

  // Utility class.
  private OsUtils() {
  }

  public static String executableExtension(OS os) {
    return os == OS.WINDOWS ? ".exe" : "";
  }

  /**
   * Returns the extension used for executables on the current platform (.exe
   * for Windows, empty string for others).
   */
  public static String executableExtension() {
    return EXECUTABLE_EXTENSION;
  }
}
