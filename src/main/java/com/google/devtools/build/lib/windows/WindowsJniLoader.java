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

package com.google.devtools.build.lib.windows;

import java.io.IOException;

/**
 * Loads native code under Windows.
 */
public class WindowsJniLoader {
  private static boolean jniLoaded = false;
  public static synchronized void loadJni() {
    if (jniLoaded) {
      return;
    }

    try {
      System.loadLibrary("windows_jni");
    } catch (UnsatisfiedLinkError ex) {
      // We are probably in tests, let's try to find the library in the runfiles
      try {
        System.load(WindowsRunfiles.getRunfile("io_bazel/src/main/native/windows_jni.dll"));
      } catch (IOException e) {
        // We throw the UnsatisfiedLinkError if we cannot find the runfiles
        throw ex;
      }
    }
    jniLoaded = true;
  }
}
