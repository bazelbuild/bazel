// Copyright 2014 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib;

import com.google.devtools.build.lib.util.OS;

import java.io.File;

/**
 * A class to load JNI dependencies for Bazel.
 */
public class UnixJniLoader {
  public static void loadJni() {
    try {
      if (OS.getCurrent() != OS.WINDOWS) {
        System.loadLibrary("unix");
      }
    } catch (UnsatisfiedLinkError ex) {
      // We are probably in tests, let's try to find the library relative to where we are.
      File cwd = new File(System.getProperty("user.dir"));
      String libunix = "src" + File.separator + "main" + File.separator + "native" + File.separator
          + System.mapLibraryName("unix");
      File toTest = new File(cwd, libunix);
      if (toTest.exists()) {
        System.load(toTest.toString());
      } else {
        throw ex;
      }
    }
  }
}
