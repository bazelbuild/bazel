// Copyright 2016 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.unix;

import com.google.devtools.build.lib.UnixJniLoader;

import java.io.IOException;

/**
 * Utility methods for access to UNIX system calls not exposed by the Java
 * SDK. Exception messages are selected to be consistent with those generated
 * by the java.io package where appropriate--see package javadoc for details.
 */
public class NativePosixSystem {

  private NativePosixSystem() {}

  static {
    if (!"0".equals(System.getProperty("io.bazel.EnableJni"))) {
      UnixJniLoader.loadJni();
    }
  }

  /**
   * Native wrapper around POSIX sysctlbyname(3) syscall.
   *
   * @param name the name for value to get from sysctl
   * @throws IOException iff the sysctlbyname() syscall failed.
   */
  public static native long sysctlbynameGetLong(String name) throws IOException;
}
