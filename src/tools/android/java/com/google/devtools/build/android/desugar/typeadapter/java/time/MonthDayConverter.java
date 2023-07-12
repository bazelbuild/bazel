/*
 * Copyright 2020 The Bazel Authors. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.google.devtools.build.android.desugar.typeadapter.java.time;

/** Converts types between the desugar-mirrored and desugar-shadowed {@link java.time.MonthDay}. */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class MonthDayConverter {

  private MonthDayConverter() {}

  public static j$.time.MonthDay from(java.time.MonthDay monthDay) {
    return monthDay == null
        ? null
        : j$.time.MonthDay.of(monthDay.getMonthValue(), monthDay.getDayOfMonth());
  }

  public static java.time.MonthDay to(j$.time.MonthDay monthDay) {
    return monthDay == null
        ? null
        : java.time.MonthDay.of(monthDay.getMonthValue(), monthDay.getDayOfMonth());
  }
}
