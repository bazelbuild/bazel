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

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;

/**
 * Provides the clock implementation used by Blaze, which is {@link JavaClock}
 * by default, but can be overridden at runtime. Note that if you set this
 * clock, you also have to set the clock used by the Profiler.
 */
@ThreadSafe
public abstract class BlazeClock {

  private BlazeClock() {
  }

  private static volatile Clock instance = new JavaClock();

  /**
   * Returns singleton instance of the clock
   */
  public static Clock instance() {
    return instance;
  }

  /**
   * Overrides default clock instance.
   */
  public static synchronized void setClock(Clock clock) {
    instance = clock;
  }

  public static long nanoTime() {
    return instance().nanoTime();
  }
}
