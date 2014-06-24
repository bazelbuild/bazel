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

package com.google.devtools.build.lib.blaze;

import com.google.common.base.Preconditions;
import com.google.common.collect.Maps;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionMetadata;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.util.Clock;

import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.TimeUnit;

/**
 * Computes the critical path in the action graph based on events published to the event bus.
 *
 * <p>After instantiation, this object needs to be registered on the event bus to work.
 */
public abstract class CriticalPathComputer<C extends AbstractCriticalPathComponent<C>,
                                           A extends AggregatedCriticalPath<C>> {

  // outputArtifactToStats is accessed from multiple event handlers.
  protected final ConcurrentMap<Artifact, C> outputArtifactToStats = Maps.newConcurrentMap();

  /** Maximum critical path found. */
  private C maxCriticalPath;
  private final Clock clock;

  public CriticalPathComputer(Clock clock) {
    this.clock = clock;
    maxCriticalPath = null;
  }

  public abstract C createComponent(Action action, long nanoTimeStart);

  @Subscribe
  public void logActionStarted(ActionStartedEvent event) {
    Action action = event.getAction();
    C component = createComponent(action, TimeUnit.NANOSECONDS.toMillis(event.getNanoTimeStart()));
    for (Artifact output : action.getOutputs()) {
      C old = outputArtifactToStats.put(output, component);
      Preconditions.checkState(old == null, "Previous event registered for action %s", action);
    }
  }

  private long getTime() {
    return TimeUnit.NANOSECONDS.toMillis(clock.nanoTime());
  }


  /**
   * Record an action that was not executed because it was in the (disk) cache. This is needed so
   * that we can calculate correctly the dependencies tree if we have some cached actions in the
   * middle of the critical path.
   */
  @Subscribe
  public void actionCached(CachedActionEvent event) {
    Action action = event.getAction();
    C stat = createComponent(action, TimeUnit.NANOSECONDS.toMillis(event.getNanoTimeStart()));
    for (Artifact output : action.getOutputs()) {
      outputArtifactToStats.put(output, stat);
    }
    finalizeActionStat(action, stat);
  }

  /**
   * Records the elapsed time stats for the action. For each input artifact, it finds the real
   * dependent artifacts and records the critical path stats.
   */
  @Subscribe
  public void actionComplete(ActionCompletionEvent event) {
    ActionMetadata action = event.getActionMetadata();
    C stats = Preconditions.checkNotNull(
        outputArtifactToStats.get(action.getPrimaryOutput()));
    finalizeActionStat(action, stats);
  }

  private void finalizeActionStat(ActionMetadata action, C stats) {
    stats.setFinishTime(getTime());
    for (Artifact input : action.getInputs()) {
      addArtifactDependency(stats, input);
    }
    if (isBiggestCriticalPath(stats)) {
      maxCriticalPath = stats;
    }
  }

  private boolean isBiggestCriticalPath(C newCriticalPath) {
    return maxCriticalPath == null
        || maxCriticalPath.getAggregatedWallTime() < newCriticalPath.getAggregatedWallTime();
  }

  /**
   * If "input" is a generated artifact, link its critical path to the one we're building.
   */
  private void addArtifactDependency(C actionStats, Artifact input) {
    C depStats = outputArtifactToStats.get(input);
    if (depStats != null) {
      actionStats.addDepInfo(depStats);
    }
  }

  /**
   * Return the critical path stats for the current command execution.
   *
   * <p>This method allow us to calculate lazily the aggregate statistics of the critical path,
   * avoiding the memory and cpu penalty for doing it for all the actions executed.
   */
  public abstract A aggregate();

  /** Maximum critical path component found during the build. */
  protected C getMaxCriticalPath() {
    return maxCriticalPath;
  }
}
