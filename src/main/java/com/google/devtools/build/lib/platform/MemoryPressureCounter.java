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

import com.google.devtools.build.lib.jni.JniLoader;

/** Native methods for dealing with memory pressure events. */
public final class MemoryPressureCounter {

  static {
    JniLoader.loadJni();
  }

  private MemoryPressureCounter() {}

  static native int warningCountJNI();

  static native int criticalCountJNI();

  /** The number of times that a memory pressure warning notification has been seen. */
  public static int warningCount() {
    return JniLoader.isJniAvailable() ? warningCountJNI() : 0;
  }

  /** The number of times that a memory pressure critical notification has been seen. */
  public static int criticalCount() {
    return JniLoader.isJniAvailable() ? criticalCountJNI() : 0;
  }
}
