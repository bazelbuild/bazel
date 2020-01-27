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
package com.google.devtools.build.lib.runtime;

import com.google.devtools.build.lib.clock.BlazeClock;
import java.lang.management.GarbageCollectorMXBean;
import java.lang.management.ManagementFactory;
import java.util.Date;

/**
 * Base class for Command events that includes some resource fields.
 */
public abstract class CommandEvent {

  private final long eventTimeInNanos;
  private final long eventTimeInEpochTime;
  private final long gcTimeInMillis;

  protected CommandEvent() {
    eventTimeInNanos = BlazeClock.nanoTime();
    eventTimeInEpochTime = new Date().getTime();
    gcTimeInMillis = collectGcTimeInMillis();
  }

  /**
   * Returns time spent in garbage collection since the start of the JVM process.
   */
  private static long collectGcTimeInMillis() {
    long gcTime = 0;
    for (GarbageCollectorMXBean gcBean : ManagementFactory.getGarbageCollectorMXBeans()) {
      gcTime += gcBean.getCollectionTime();
    }
    return gcTime;
  }

  /**
   * Get the time-stamp in ns for the event.
   */
  public long getEventTimeInNanos() {
    return eventTimeInNanos;
  }

  /**
   * Get the time-stamp as epoch-time for the event.
   */
  public long getEventTimeInEpochTime() {
    return eventTimeInEpochTime;
  }

  /**
   * Get the cumulative GC time for the event.
   */
  public long getGCTimeInMillis() {
    return gcTimeInMillis;
  }
}
