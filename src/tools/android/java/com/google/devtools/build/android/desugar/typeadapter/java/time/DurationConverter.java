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

/** Converts types between the desugar-mirrored and desugar-shadowed {@link java.time.Duration}. */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class DurationConverter {

  private DurationConverter() {}

  public static j$.time.Duration from(java.time.Duration duration) {
    return duration == null
        ? null
        : j$.time.Duration.ofSeconds(duration.getSeconds(), duration.getNano());
  }

  public static java.time.Duration to(j$.time.Duration duration) {
    return duration == null
        ? null
        : java.time.Duration.ofSeconds(duration.getSeconds(), duration.getNano());
  }
}
