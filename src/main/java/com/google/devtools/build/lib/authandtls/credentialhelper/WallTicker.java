// Copyright 2023 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.authandtls.credentialhelper;

import com.github.benmanes.caffeine.cache.Ticker;
import com.google.devtools.build.lib.clock.Clock;
import java.time.Duration;

/**
 * WallTicker is a Ticker which reports wall time since the unix epoch.
 *
 * <p>We use this instead of com.github.benmanes.caffeine.cache.Ticker.SystemTicker because the
 * latter uses monotonic time (which doesn't increment the time source while the system is asleep)
 * with an unspecified reference point (which is unhelpful when computing the cache duration for
 * credentials whose expiry is a fixed point in time, not a fixed duration).
 */
final class WallTicker implements Ticker {
  private final Clock clock;

  WallTicker(Clock clock) {
    this.clock = clock;
  }

  @Override
  public long read() {
    // Documented to return a value in nanoseconds.
    return Duration.ofMillis(clock.currentTimeMillis()).toNanos();
  }
}
