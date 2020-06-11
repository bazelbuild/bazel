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

/**
 * Converts types between the desugar-mirrored and desugar-shadowed {@link java.time.ZonedDateTime}.
 */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class ZonedDateTimeConverter {

  private ZonedDateTimeConverter() {}

  public static j$.time.ZonedDateTime from(java.time.ZonedDateTime zonedDateTime) {
    return zonedDateTime == null
        ? null
        : j$.time.ZonedDateTime.of(
            zonedDateTime.getYear(),
            zonedDateTime.getMonthValue(),
            zonedDateTime.getDayOfMonth(),
            zonedDateTime.getHour(),
            zonedDateTime.getMinute(),
            zonedDateTime.getSecond(),
            zonedDateTime.getNano(),
            j$.time.ZoneId.of(zonedDateTime.getZone().getId()));
  }

  public static java.time.ZonedDateTime to(j$.time.ZonedDateTime zonedDateTime) {
    return zonedDateTime == null
        ? null
        : java.time.ZonedDateTime.of(
            zonedDateTime.getYear(),
            zonedDateTime.getMonthValue(),
            zonedDateTime.getDayOfMonth(),
            zonedDateTime.getHour(),
            zonedDateTime.getMinute(),
            zonedDateTime.getSecond(),
            zonedDateTime.getNano(),
            java.time.ZoneId.of(zonedDateTime.getZone().getId()));
  }
}
