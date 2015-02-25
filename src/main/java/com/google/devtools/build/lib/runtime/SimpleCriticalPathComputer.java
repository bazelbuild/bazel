// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.util.Clock;

/**
 * Computes the critical path during a build.
 */
public class SimpleCriticalPathComputer
    extends CriticalPathComputer<SimpleCriticalPathComponent,
        AggregatedCriticalPath<SimpleCriticalPathComponent>> {

  public SimpleCriticalPathComputer(Clock clock) {
    super(clock);
  }

  @Override
  public SimpleCriticalPathComponent createComponent(Action action, long relativeStartNanos) {
    return new SimpleCriticalPathComponent(action, relativeStartNanos);
  }

  /**
   * Return the critical path stats for the current command execution.
   *
   * <p>This method allow us to calculate lazily the aggregate statistics of the critical path,
   * avoiding the memory and cpu penalty for doing it for all the actions executed.
   */
  @Override
  public AggregatedCriticalPath<SimpleCriticalPathComponent> aggregate() {
    ImmutableList.Builder<SimpleCriticalPathComponent> components = ImmutableList.builder();
    SimpleCriticalPathComponent maxCriticalPath = getMaxCriticalPath();
    if (maxCriticalPath == null) {
      return new AggregatedCriticalPath<>(0, components.build());
    }
    SimpleCriticalPathComponent child = maxCriticalPath;
    while (child != null) {
      components.add(child);
      child = child.getChild();
    }
    return new AggregatedCriticalPath<>(maxCriticalPath.getAggregatedElapsedTimeMillis(),
        components.build());
  }
}

