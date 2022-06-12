// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.platform;

import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.jni.JniLoader;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/** A singleton that is the java side interface for dealing with memory pressure events. */
public final class SystemMemoryPressureMonitor {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private static final SystemMemoryPressureMonitor singleton = new SystemMemoryPressureMonitor();

  @GuardedBy("this")
  @Nullable private Reporter reporter;

  @GuardedBy("this")
  private int eventCount = 0;

  public static SystemMemoryPressureMonitor getInstance() {
    return singleton;
  }

  private SystemMemoryPressureMonitor() {
    JniLoader.loadJni();
    if (JniLoader.isJniAvailable()) {
      registerJNI();
    }
  }

  private native void registerJNI();

  /** The number of times that a memory pressure notification has been seen. */
  public synchronized int eventCount() {
    return eventCount;
  }

  public synchronized void setReporter(@Nullable Reporter reporter) {
    this.reporter = reporter;
  }

  synchronized void memoryPressureCallback(int value) {
    SystemMemoryPressureEvent event = new SystemMemoryPressureEvent(value);
    if (reporter != null) {
      reporter.post(event);
    }
    logger.atInfo().log(event.logString());
    eventCount += 1;
  }
}
