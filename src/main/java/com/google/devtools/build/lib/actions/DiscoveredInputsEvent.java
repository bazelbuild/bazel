// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.actions;

import com.google.devtools.build.lib.events.ExtendedEventHandler;

/** Event for passing SpawnMetrics for the discover inputs phase. */
public class DiscoveredInputsEvent implements ExtendedEventHandler.ProgressLike {
  private final SpawnMetrics metrics;
  private final Action action;
  private final long startTimeNanos;

  public DiscoveredInputsEvent(SpawnMetrics metrics, Action action, long startTimeNanos) {
    this.metrics = metrics;
    this.action = action;
    this.startTimeNanos = startTimeNanos;
  }

  public SpawnMetrics getMetrics() {
    return metrics;
  }

  public Action getAction() {
    return action;
  }

  public long getStartTimeNanos() {
    return startTimeNanos;
  }
}
