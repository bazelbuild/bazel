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
 * com.google.devtools.build.lib.platform.SystemDiskSpaceModule#diskSpaceCallback} to indicate that
 * a disk space event has occurred.
 */
public class SystemDiskSpaceEvent implements ExtendedEventHandler.Postable {

  /** Rough description of why the disk space event fired. */
  public enum Level {
    LOW("Low"),
    VERY_LOW("Very Low");

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
        case 0 -> LOW;
        case 1 -> VERY_LOW;
        default -> throw new IllegalStateException("Unknown disk space level: " + number);
      };
    }
  };

  private final Level level;

  public SystemDiskSpaceEvent(int level) {
    this.level = Level.fromInt(level);
  }

  public Level level() {
    return level;
  }

  public String logString() {
    return "SystemDiskSpaceEvent: " + level.logString();
  }
}
