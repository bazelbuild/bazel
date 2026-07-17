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
 * com.google.devtools.build.lib.platform.SystemLoadAdvisoryEventModule#systemLoadAdvisoryCallback}
 * to indicate that an advisory event has occurred.
 */
public class SystemLoadAdvisoryEvent implements ExtendedEventHandler.Postable {
  private final int value;
  private final String osDescription;

  /**
   * SystemLoadAdvisoryEvent Constructor.
   *
   * @param value The actual load advisory value (0-100).
   * @param osDescription A description string that may be OS specific.
   */
  public SystemLoadAdvisoryEvent(int value, String osDescription) {
    this.value = value;
    this.osDescription = osDescription;
  }

  public int value() {
    return value;
  }

  public String osDescription() {
    return osDescription;
  }

  public String logString() {
    return String.format("SystemLoadAdvisoryEvent: %d (%s)", value, osDescription);
  }
}
