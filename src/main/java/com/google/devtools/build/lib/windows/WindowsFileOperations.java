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

  // A note about UNC paths and path prefixes on Windows. The prefixes can be:
  // - "\\?\", meaning it's a UNC path that is passed to user mode unicode WinAPI functions
  //   (e.g. CreateFileW) or a return value of theirs (e.g. GetLongPathNameW); this is the
  //   prefix we'll most often see
  // - "\??\", meaning it's Device Object path; it's mostly only used by kernel/driver functions
  //   but we may come across it when resolving junction targets, as the target's path is
  //   specified with this prefix, see usages of DeviceIoControl with FSCTL_GET_REPARSE_POINT
  // - "\\.\", meaning it's a Device Object path again; both "\??\" and "\\.\" are shorthands
  //   for the "\DosDevices\" Object Directory, so "\\.\C:" and "\??\C:" and "\DosDevices\C:"
  //   and "C:\" all mean the same thing, but functions like CreateFileW don't understand the
  //   fully qualified device path, only the shorthand versions; the difference between "\\.\"
  //   is "\??\" is not entirely clear (one is not available while Windows is booting, but
  //   that only concerns device drivers) but we most likely won't come across them anyway
  // Some of this is documented here:
  // - https://msdn.microsoft.com/en-us/library/windows/hardware/ff557762(v=vs.85).aspx
  // - https://msdn.microsoft.com/en-us/library/windows/hardware/ff565384(v=vs.85).aspx
  // - http://stackoverflow.com/questions/23041983
  // - http://stackoverflow.com/questions/14482421

  private WindowsFileOperations() {
    // Prevent construction
  }

  // Keep IS_JUNCTION_* values in sync with src/main/native/windows_file_operations.cc.
  private static final int IS_JUNCTION_YES = 0;
  private static final int IS_JUNCTION_NO = 1;
  private static final int IS_JUNCTION_ERROR = 2;

  static native int nativeIsJunction(String path, String[] error);

  static native boolean nativeGetLongPath(String path, String[] result, String[] error);

  /** Determines whether `path` is a junction point or directory symlink. */
  public static boolean isJunction(String path) throws IOException {
    WindowsJniLoader.loadJni();
    String[] error = new String[] {null};
    switch (nativeIsJunction(asLongPath(path), error)) {
      case IS_JUNCTION_YES:
        return true;
      case IS_JUNCTION_NO:
        return false;
      default:
        throw new IOException(error[0]);
    }
  }

  /**
   * Returns the long path associated with the input `path`.
   *
   * <p>This method resolves all 8dot3 style components of the path and returns the long format. For
   * example, if the input is "C:/progra~1/micros~1" the result may be "C:\Program Files\Microsoft
   * Visual Studio 14.0". The returned path is Windows-style in that it uses backslashes, even if
   * the input uses forward slashes.
   *
   * <p>May return an UNC path if `path` or its resolution is sufficiently long.
   *
   * @throws IOException if the `path` is not found or some other I/O error occurs
   */
  public static String getLongPath(String path) throws IOException {
    WindowsJniLoader.loadJni();
    String[] result = new String[] {null};
    String[] error = new String[] {null};
    if (nativeGetLongPath(asLongPath(path), result, error)) {
      return result[0];
    } else {
      throw new IOException(error[0]);
    }
  }

  /**
   * Returns a Windows-style path suitable to pass to unicode WinAPI functions.
   *
   * <p>Returns an UNC path if `path` is longer than `MAX_PATH` (in <windows.h>). If it's shorter or
   * is already an UNC path, then this method returns `path` itself.
   */
  static String asLongPath(String path) {
    return path.length() > 260 && !path.startsWith("\\\\?\\")
        ? ("\\\\?\\" + path.replace('/', '\\'))
        : path;
  }
}
