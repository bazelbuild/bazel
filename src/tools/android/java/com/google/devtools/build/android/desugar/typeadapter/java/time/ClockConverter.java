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

/** Converts types between the desugar-mirrored and desugar-shadowed {@link java.time.Clock}. */
@SuppressWarnings("AndroidJdkLibsChecker")
public abstract class ClockConverter {

  private ClockConverter() {}

  public static j$.time.Clock from(final java.time.Clock clock) {
    return clock == null
        ? null
        : new j$.time.Clock() {
          @Override
          public j$.time.ZoneId getZone() {
            return ZoneIdConverter.from(clock.getZone());
          }

          @Override
          public j$.time.Clock withZone(j$.time.ZoneId zone) {
            return from(clock.withZone(ZoneIdConverter.to(zone)));
          }

          @Override
          public j$.time.Instant instant() {
            return InstantConverter.from(clock.instant());
          }
        };
  }

  public static java.time.Clock to(final j$.time.Clock clock) {
    return clock == null
        ? null
        : new java.time.Clock() {
          @Override
          public java.time.ZoneId getZone() {
            return ZoneIdConverter.to(clock.getZone());
          }

          @Override
          public java.time.Clock withZone(java.time.ZoneId zone) {
            return to(clock.withZone(ZoneIdConverter.from(zone)));
          }

          @Override
          public java.time.Instant instant() {
            return InstantConverter.to(clock.instant());
          }
        };
  }
}
