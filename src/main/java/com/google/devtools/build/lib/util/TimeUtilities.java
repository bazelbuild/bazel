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

/** Various utility methods operating on time values. */
public abstract class TimeUtilities {
  private TimeUtilities() {}

  /**
   * Converts time to the user-friendly string representation.
   *
   * @param timeInNs The length of time in nanoseconds.
   */
  public static String prettyTime(double timeInNs) {
    double ms = timeInNs / 1000000.0;
    if (ms < 10.0) {
      return String.format("%.2f ms", ms);
    } else if (ms < 100.0) {
      return String.format("%.1f ms", ms);
    } else if (ms < 1000.0) {
      return String.format("%.0f ms", ms);
    }
    return String.format("%.3f s", ms / 1000.0);
  }
}
