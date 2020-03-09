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

package com.google.devtools.build.lib.platform;

import com.google.devtools.build.lib.unix.jni.UnixJniLoader;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.windows.jni.WindowsJniLoader;

/** All classes that extend this class depend on being able to load jni. */
class JniLoader {
  private static final boolean JNI_ENABLED;

  static {
    JNI_ENABLED = !"0".equals(System.getProperty("io.bazel.EnableJni"));
    if (JNI_ENABLED) {
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
    }
  }

  protected JniLoader() {}

  public static boolean jniEnabled() {
    return JNI_ENABLED;
  }
}
