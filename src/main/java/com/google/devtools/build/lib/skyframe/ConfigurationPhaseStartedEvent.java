// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.events.ExtendedEventHandler;

/** This event is fired at the beginning of the target configuration phase. */
public final class ConfigurationPhaseStartedEvent implements ExtendedEventHandler.Postable {

  final ConfiguredTargetProgressReceiver configuredTargetProgress;

  /**
   * Construct the event.
   *
   * @param configuredTargetProgress a receiver that gets updated about the progress of target
   *     configuration.
   */
  public ConfigurationPhaseStartedEvent(ConfiguredTargetProgressReceiver configuredTargetProgress) {
    this.configuredTargetProgress = configuredTargetProgress;
  }

  public ConfiguredTargetProgressReceiver getConfiguredTargetProgressReceiver() {
    return configuredTargetProgress;
  }
}
