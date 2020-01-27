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

package com.google.devtools.build.runfiles;

/**
 * Utilities for the other classes in this package.
 *
 * <p>These functions are implementations of some basic utilities in the Guava library. We
 * reimplement these functions instead of depending on Guava, so that the Runfiles library has no
 * third-party dependencies, thus any Java project can depend on it without the risk of pulling
 * unwanted or conflicting dependencies (for example if the project already depends on Guava, or
 * wishes not to depend on it at all).
 */
class Util {
  private Util() {}

  /** Returns true when {@code s} is null or an empty string. */
  public static boolean isNullOrEmpty(String s) {
    return s == null || s.isEmpty();
  }

  /** Throws an {@code IllegalArgumentException} if {@code condition} is false. */
  public static void checkArgument(boolean condition) {
    checkArgument(condition, null, null);
  }

  /** Throws an {@code IllegalArgumentException} if {@code condition} is false. */
  public static void checkArgument(boolean condition, String error, Object arg1) {
    if (!condition) {
      if (isNullOrEmpty(error)) {
        throw new IllegalArgumentException("argument validation failed");
      } else {
        throw new IllegalArgumentException(String.format(error, arg1));
      }
    }
  }
}
