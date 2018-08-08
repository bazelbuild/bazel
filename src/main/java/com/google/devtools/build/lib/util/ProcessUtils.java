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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.windows.jni.WindowsProcesses;

/**
 * OS Process related utilities.
 */
@ThreadSafe
public final class ProcessUtils {
  private ProcessUtils() {
    // prevent construction.
  }

  /**
   * @return the real group ID of the current process.
   * @throws UnsatisfiedLinkError when JNI is not available.
   * @throws UnsupportedOperationException on operating systems where this call is not implemented.
   */
  public static int getgid() {
    if (OS.getCurrent() == OS.WINDOWS) {
      throw new UnsupportedOperationException();
    } else {
      return com.google.devtools.build.lib.unix.ProcessUtils.getgid();
    }
  }

  /**
   * @return the process ID of this process.
   * @throws UnsatisfiedLinkError when JNI is not available.
   */
  public static int getpid() {
    // TODO(ulfjack): Use ProcessHandle.current().getPid() here.
    if (OS.getCurrent() == OS.WINDOWS) {
      return WindowsProcesses.getpid();
    } else {
      return com.google.devtools.build.lib.unix.ProcessUtils.getpid();
    }
  }

  /**
   * @return the real user ID of the current process.
   * @throws UnsatisfiedLinkError when JNI is not available.
   * @throws UnsupportedOperationException on operating systems where this call is not implemented.
   */
  public static int getuid() {
    if (OS.getCurrent() == OS.WINDOWS) {
      throw new UnsupportedOperationException();
    } else {
      return com.google.devtools.build.lib.unix.ProcessUtils.getuid();
    }
  }
}
