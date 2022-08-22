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
import com.google.devtools.build.lib.platform.SystemMemoryPressureMonitor.Level;

/**
 * This event is fired from {@link
 * com.google.devtools.build.lib.platform.SystemMemoryPressureModule#memoryPressureCallback} to
 * indicate that a memory pressure event has occurred.
 */
public class SystemMemoryPressureEvent implements ExtendedEventHandler.Postable {

  private final Level level;

  public SystemMemoryPressureEvent(Level level) {
    this.level = level;
  }

  public Level level() {
    return level;
  }

  public String logString() {
    return "SystemMemoryPressureEvent: " + level.logString();
  }
}
