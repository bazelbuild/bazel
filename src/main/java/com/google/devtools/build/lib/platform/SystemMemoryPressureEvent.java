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

import com.google.devtools.build.lib.events.ExtendedEventHandler;

/**
 * This event is fired from {@link
 * com.google.devtools.build.lib.platform.SystemMemoryPressureModule#memoryPressureCallback} to
 * indicate that a memory pressure event has occurred.
 */
public class SystemMemoryPressureEvent implements ExtendedEventHandler.Postable {

  /** The possible reasons a system could be suspended. */
  public enum Level {
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
      switch (number) {
        case 0:
          return WARNING;
        case 1:
          return CRITICAL;
        default:
          throw new IllegalStateException("Unknown memory pressure level: " + number);
      }
    }
  };

  private final Level level;

  public SystemMemoryPressureEvent(int level) {
    this.level = Level.fromInt(level);
  }

  public Level level() {
    return level;
  }

  public String logString() {
    return "SystemMemoryPressureEvent: " + level.logString();
  }
}
