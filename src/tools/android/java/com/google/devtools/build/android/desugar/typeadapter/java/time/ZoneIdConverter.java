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

/** Converts types between the desugar-mirrored and desugar-shadowed {@link java.time.ZoneId}. */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class ZoneIdConverter {

  private ZoneIdConverter() {}

  public static j$.time.ZoneId from(java.time.ZoneId zoneId) {
    return zoneId == null ? null : j$.time.ZoneId.of(zoneId.getId());
  }

  public static java.time.ZoneId to(j$.time.ZoneId zoneId) {
    return zoneId == null ? null : java.time.ZoneId.of(zoneId.getId());
  }
}
