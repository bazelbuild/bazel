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
package com.google.devtools.build.android.desugar.runtime;

/**
 * Conversions for hidden Android APIs that use desugared primitives (see b/79121791).  These are
 * grouped into a separate class to simplify building with them, since they use APIs that are
 * omitted in the android.jar.
 */
@SuppressWarnings("AndroidApiChecker")
public final class HiddenConversions {
  private HiddenConversions() {}  // static methods only

  public static j$.time.LocalDate getLocalDate(
      android.hardware.display.AmbientBrightnessDayStats stats) {
    return fromLocalDate(stats.getLocalDate());
  }

  private static j$.time.LocalDate fromLocalDate(java.time.LocalDate date) {
    if (date == null) {
      return null;
    }
    return j$.time.LocalDate.of(date.getYear(), date.getMonthValue(), date.getDayOfMonth());
  }
}
