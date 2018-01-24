// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.windows.jni;

import com.google.devtools.build.lib.windows.runfiles.WindowsRunfiles;
import java.io.IOException;

/** Loads native code under Windows. */
public class WindowsJniLoader {
  private static final String[] SEARCH_PATHS = {
    "io_bazel/src/main/native/windows/windows_jni.dll",
    "io_bazel/external/bazel_tools/src/main/native/windows/windows_jni.dll",
    "bazel_tools/src/main/native/windows/windows_jni.dll",
  };

  private static boolean jniLoaded = false;

  public static synchronized void loadJni() {
    if (jniLoaded) {
      return;
    }

    try {
      System.loadLibrary("windows_jni");
    } catch (UnsatisfiedLinkError ex) {
      // Try to find the library in the runfiles.
      loadFromRunfileOrThrow(ex);
    }
    jniLoaded = true;
  }

  private static void loadFromRunfileOrThrow(UnsatisfiedLinkError ex) {
    for (String path : SEARCH_PATHS) {
      if (loadFromRunfileOrThrow(path, ex)) {
        return;
      }
    }

    // We throw the UnsatisfiedLinkError if we cannot find the DLL under any known location.
    throw ex;
  }

  private static boolean loadFromRunfileOrThrow(String runfile, UnsatisfiedLinkError ex) {
    // Try to find the library in the runfiles.
    String path;
    try {
      path = WindowsRunfiles.getRunfile(runfile);
      if (path == null) {
        // Just return false if the runfile path was not found. Maybe it's under a different path.
        return false;
      }
      System.load(path);
      return true;
    } catch (IOException e) {
      // We throw the UnsatisfiedLinkError if we cannot find the runfiles
      throw ex;
    }
  }
}
