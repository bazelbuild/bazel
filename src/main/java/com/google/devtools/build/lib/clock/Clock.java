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
package com.google.devtools.build.lib.clock;

import java.time.Instant;

/** Provides an interface for a pluggable clock. */
public interface Clock {

  /**
   * Returns the current time in milliseconds. The milliseconds are counted from midnight Jan 1,
   * 1970.
   */
  long currentTimeMillis();

  /**
   * Returns the current time in nanoseconds. The nanoseconds are measured relative to some unknown,
   * but fixed event. Unfortunately, a sequence of calls to this method is *not* guaranteed to
   * return non-decreasing values, so callers should be tolerant to this behavior.
   */
  long nanoTime();

  /** Returns {@link #currentTimeMillis} as an {@link Instant}. */
  default Instant now() {
    return Instant.ofEpochMilli(currentTimeMillis());
  }
}
