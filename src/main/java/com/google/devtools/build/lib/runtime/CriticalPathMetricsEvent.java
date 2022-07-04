// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.events.ExtendedEventHandler;

import java.time.Duration;

/**
 * Event transporting information about the critical path. Emitted after the critical path has
 * been aggregated.
**/
public class CriticalPathMetricsEvent implements ExtendedEventHandler.Postable {
  private final Duration totalTime;

  public CriticalPathMetricsEvent(Duration totalTime) {
    this.totalTime = totalTime;
  }

  public Duration getTotalTime() {
    return totalTime;
  }
}
