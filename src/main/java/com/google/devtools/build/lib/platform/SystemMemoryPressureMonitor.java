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
  @Nullable
  private Reporter reporter;

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

  private native int systemMemoryPressure();

  /** The possible memory pressure levels. */
  public enum Level {
    NORMAL("Normal"),
    WARNING("Warning"),
    CRITICAL("Critical");

    private final String logString;

    Level(String logString) {
      this.logString = logString;
    }

    public String logString() {
      return logString;
    }

    /** These constants are mapped to enum in third_party/bazel/src/main/native/unix_jni.h. */
    static Level fromInt(int number) {
      return switch (number) {
        case 0 -> NORMAL;
        case 1 -> WARNING;
        case 2 -> CRITICAL;
        default -> throw new IllegalStateException("Unknown memory pressure level: " + number);
      };
    }
  };

  /** Return current memory pressure */
  public Level level() {
    return Level.fromInt(systemMemoryPressure());
  }

  public synchronized void setReporter(@Nullable Reporter reporter) {
    this.reporter = reporter;
    int pressure = systemMemoryPressure();
    if (Level.fromInt(pressure) != Level.NORMAL) {
      memoryPressureCallback(pressure);
    }
  }

  synchronized void memoryPressureCallback(int value) {
    SystemMemoryPressureEvent event = new SystemMemoryPressureEvent(Level.fromInt(value));
    if (reporter != null) {
      reporter.post(event);
    }
    logger.atInfo().log("%s", event.logString());
  }
}
