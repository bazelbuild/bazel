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

import com.google.devtools.build.lib.vfs.PathFragment;

/**
 * Operating system-specific utilities.
 */
public final class OsUtils {

  private static final String EXECUTABLE_EXTENSION = OS.getCurrent() == OS.WINDOWS ? ".exe" : "";

  // Utility class.
  private OsUtils() {
  }

  /**
   * Returns the extension used for executables on the current platform (.exe
   * for Windows, empty string for others).
   */
  public static String executableExtension() {
    return EXECUTABLE_EXTENSION;
  }

  /**
   * Loads JNI libraries, if necessary under the current platform.
   */
  public static void maybeForceJNI(PathFragment installBase) {
    if (jniLibsAvailable()) {
      forceJNI(installBase);
    }
  }

  private static boolean jniLibsAvailable() {
    if ("0".equals(System.getProperty("io.bazel.UnixFileSystem"))) {
      return false;
    }
    // JNI libraries work fine on Windows, but at the moment we are not using any.
    return OS.getCurrent() != OS.WINDOWS;
  }

  // Force JNI linking at a moment when we have 'installBase' handy, and print
  // an informative error if it fails.
  private static void forceJNI(PathFragment installBase) {
    try {
      ProcessUtils.getpid(); // force JNI initialization
    } catch (UnsatisfiedLinkError t) {
      System.err.println("JNI initialization failed: " + t.getMessage() + ".  "
          + "Possibly your installation has been corrupted; "
          + "if this problem persists, try 'rm -fr " + installBase + "'.");
      throw t;
    }
  }

  /**
   * Returns the PID of the current process, or -1 if not available.
   */
  public static int getpid() {
    if (jniLibsAvailable()) {
      return ProcessUtils.getpid();
    }
    return -1;
  }
}
