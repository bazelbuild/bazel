// Copyright 2025 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.util.OS;
import com.sun.security.auth.module.UnixSystem;
import javax.annotation.Nullable;

/** UNIX process utilities. */
public final class ProcessUtils {
  @Nullable
  private static final UnixSystem UNIX_SYSTEM =
      OS.getCurrent().isPosixCompatible() ? new UnixSystem() : null;

  /**
   * Returns the real user ID of the current process.
   *
   * @throws UnsupportedOperationException on operating systems where this call is not supported.
   */
  public static long getUid() {
    if (UNIX_SYSTEM == null) {
      throw new UnsupportedOperationException();
    }
    return UNIX_SYSTEM.getUid();
  }

  /**
   * Returns the real group ID of the current process.
   *
   * @throws UnsupportedOperationException on operating systems where this call is not supported.
   */
  public static long getGid() {
    if (UNIX_SYSTEM == null) {
      throw new UnsupportedOperationException();
    }
    return UNIX_SYSTEM.getGid();
  }

  private ProcessUtils() {}
}
