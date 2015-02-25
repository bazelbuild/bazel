// Copyright 2014 Google Inc. All rights reserved.
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

/**
 * OS Process related utilities.
 *
 * <p>Default implementation forwards all requests to
 * {@link com.google.devtools.build.lib.unix.ProcessUtils}. The default implementation
 * can be overridden by {@code #setImplementation(ProcessUtilsImpl)} method.
 */
@ThreadSafe
public final class ProcessUtils {

  /**
   * Describes implementation to which all {@code ProcessUtils} requests are
   * forwarded.
   */
  public interface ProcessUtilsImpl {
    /** @see ProcessUtils#getgid() */
    int getgid();

    /** @see ProcessUtils#getpid() */
    int getpid();

    /** @see ProcessUtils#getuid() */
    int getuid();
  }

  private volatile static ProcessUtilsImpl implementation = new ProcessUtilsImpl() {

    @Override
    public int getgid() {
      return com.google.devtools.build.lib.unix.ProcessUtils.getgid();
    }

    @Override
    public int getpid() {
      return com.google.devtools.build.lib.unix.ProcessUtils.getpid();
    }

    @Override
    public int getuid() {
      return com.google.devtools.build.lib.unix.ProcessUtils.getuid();
    }
  };

  private ProcessUtils() {
    // prevent construction.
  }

  /**
   * @return the real group ID of the current process.
   */
  public static int getgid() {
    return implementation.getgid();
  }

  /**
   * @return the process ID of this process.
   */
  public static int getpid() {
    return implementation.getpid();
  }

  /**
   * @return the real user ID of the current process.
   */
  public static int getuid() {
    return implementation.getuid();
  }
}
