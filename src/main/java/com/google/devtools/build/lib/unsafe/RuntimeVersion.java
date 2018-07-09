// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.unsafe;

import java.lang.reflect.Method;

/**
 * JDK version string utilities.
 *
 * <p>Code in this class is copied from Java Platform Team's JDK version string utility class,
 * in order to avoid exporting an extra dependence from internal repository.
 */
final class RuntimeVersion {

  private static final int MAJOR = getMajorJdkVersion();

  private static int getMajorJdkVersion() {
    try {
      Method versionMethod = Runtime.class.getMethod("version");
      Object version = versionMethod.invoke(null);
      return (int) version.getClass().getMethod("major").invoke(version);
    } catch (Exception e) {
      // continue below
    }

    int version = (int) Double.parseDouble(System.getProperty("java.class.version"));
    if (49 <= version && version <= 52) {
      return version - (49 - 5);
    }
    throw new IllegalStateException(
        "Unknown Java version: " + System.getProperty("java.specification.version"));
  }

  /** Returns true if the current runtime is JDK 9 or newer. */
  static boolean isAtLeast9() {
    return MAJOR >= 9;
  }
}
