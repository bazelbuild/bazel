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

/**
 * Process management on Windows.
 */
public class WindowsProcesses {
  private static boolean jniLoaded = false;
  private WindowsProcesses() {
    // Prevent construction
  }

  private static native String helloWorld(int arg, String fruit);
  private static native int nativeGetpid();

  public static int getpid() {
    ensureJni();
    return nativeGetpid();
  }

  private static synchronized void ensureJni() {
    if (jniLoaded) {
      return;
    }

    System.loadLibrary("windows_jni");
    jniLoaded = true;
  }
}
