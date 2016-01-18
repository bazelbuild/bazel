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
package com.google.devtools.build.lib.unix;

import com.google.devtools.build.lib.UnixJniLoader;


/**
 * Various utilities related to UNIX processes.
 */
public final class ProcessUtils {

  private ProcessUtils() {}

  static {
    if (!"0".equals(System.getProperty("io.bazel.UnixFileSystem"))) {
      UnixJniLoader.loadJni();
    }
  }

  /**
   * Native wrapper around POSIX getgid(2).
   *
   * @return the real group ID of the current process.
   */
  public static native int getgid();

  /**
   * Native wrapper around POSIX getpid(2) syscall.
   *
   * @return the process ID of this process.
   */
  public static native int getpid();

  /**
   * Native wrapper around POSIX getuid(2).
   *
   * @return the real user ID of the current process.
   */
  public static native int getuid();
}
