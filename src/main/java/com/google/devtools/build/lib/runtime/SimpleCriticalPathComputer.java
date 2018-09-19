// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.clock.Clock;
import java.time.Duration;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Computes the critical path during a build.
 */
public class SimpleCriticalPathComputer extends CriticalPathComputer {
  private final AtomicInteger idGenerator = new AtomicInteger();

  SimpleCriticalPathComputer(
      ActionKeyContext actionKeyContext, Clock clock, boolean discardActions) {
    super(actionKeyContext, clock, discardActions);
  }

  @Override
  public CriticalPathComponent createComponent(Action action, long relativeStartNanos) {
    int id = idGenerator.getAndIncrement();
    return discardActions
        ? new ActionDiscardingCriticalPathComponent(id, action, relativeStartNanos)
        : new CriticalPathComponent(id, action, relativeStartNanos);
  }

  /**
   * Return the critical path stats for the current command execution.
   *
   * <p>This method allow us to calculate lazily the aggregate statistics of the critical path,
   * avoiding the memory and cpu penalty for doing it for all the actions executed.
   */
  @Override
  public AggregatedCriticalPath aggregate() {
    ImmutableList.Builder<CriticalPathComponent> components = ImmutableList.builder();
    CriticalPathComponent maxCriticalPath = getMaxCriticalPath();
    if (maxCriticalPath == null) {
      return new AggregatedCriticalPath(Duration.ZERO, SpawnMetrics.EMPTY, components.build());
    }
    CriticalPathComponent child = maxCriticalPath;
    while (child != null) {
      components.add(child);
      child = child.getChild();
    }
    return new AggregatedCriticalPath(
        maxCriticalPath.getAggregatedElapsedTime(), SpawnMetrics.EMPTY, components.build());
  }
}

