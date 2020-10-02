// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.jni;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.unix.jni.UnixJniLoader;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.jni.WindowsJniLoader;

/** Generic code to interact with the platform-specific JNI code bundle. */
public final class JniLoader {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final boolean JNI_AVAILABLE;

  static {
    boolean jniAvailable;
    try {
      switch (OS.getCurrent()) {
        case LINUX:
        case FREEBSD:
        case OPENBSD:
        case UNKNOWN:
        case DARWIN:
          UnixJniLoader.loadJni();
          break;
        case WINDOWS:
          WindowsJniLoader.loadJni();
          break;
        default:
          throw new AssertionError("switch statement out of sync with OS values");
      }
      jniAvailable = true;
    } catch (UnsatisfiedLinkError e) {
      logger.atWarning().withCause(e).log("Failed to load JNI library");
      jniAvailable = false;
    }
    JNI_AVAILABLE = jniAvailable;
  }

  protected JniLoader() {}

  /**
   * Triggers the load of the JNI bundle in a platform-independent basis.
   *
   * <p>This does <b>not</b> fail if the JNI bundle cannot be loaded because there are scenarios in
   * which we want to run Bazel without JNI (e.g. during bootstrapping). We rely on the fact that
   * any calls to native code will fail anyway and with a more descriptive error message if we
   * failed to load the JNI bundle.
   *
   * <p>Callers can check if the JNI bundle load succeeded by calling {@link #isJniAvailable()}.
   */
  public static void loadJni() {}

  /** Checks whether the JNI bundle was successfully loaded or not. */
  public static boolean isJniAvailable() {
    return JNI_AVAILABLE;
  }
}
