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

/** File operations on Windows. */
public class WindowsFileOperations {

  private WindowsFileOperations() {
    // Prevent construction
  }

  // Keep IS_JUNCTION_* values in sync with src/main/native/windows_file_operations.cc.
  private static final int IS_JUNCTION_YES = 0;
  private static final int IS_JUNCTION_NO = 1;
  private static final int IS_JUNCTION_ERROR = 2;

  static native int nativeIsJunction(String path, String[] error);

  /** Determines whether `path` is a junction point or directory symlink. */
  public static boolean isJunction(String path) throws IOException {
    WindowsJniLoader.loadJni();
    String[] error = new String[] {null};
    switch (nativeIsJunction(path, error)) {
      case IS_JUNCTION_YES:
        return true;
      case IS_JUNCTION_NO:
        return false;
      default:
        throw new IOException(error[0]);
    }
  }
}
