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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionMetadata;
import com.google.devtools.build.lib.actions.ActionMiddlemanEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.util.Clock;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.PriorityQueue;
import java.util.concurrent.ConcurrentMap;

import javax.annotation.concurrent.ThreadSafe;

/**
 * Computes the critical path in the action graph based on events published to the event bus.
 *
 * <p>After instantiation, this object needs to be registered on the event bus to work.
 */
@ThreadSafe
public abstract class CriticalPathComputer<C extends AbstractCriticalPathComponent<C>,
                                           A extends AggregatedCriticalPath<C>> {

  /** Number of top actions to record. */
  static final int SLOWEST_COMPONENTS_SIZE = 30;
  // outputArtifactToComponent is accessed from multiple event handlers.
  protected final ConcurrentMap<Artifact, C> outputArtifactToComponent = Maps.newConcurrentMap();

  /** Maximum critical path found. */
  private C maxCriticalPath;
  private final Clock clock;

  /**
   * The list of slowest individual components, ignoring the time to build dependencies.
   *
   * <p>This data is a useful metric when running non highly incremental builds, where multiple
   * tasks could run un parallel and critical path would only record the longest path.
   */
  private final PriorityQueue<C> slowestComponents = new PriorityQueue<>(SLOWEST_COMPONENTS_SIZE,
      new Comparator<C>() {
        @Override
        public int compare(C o1, C o2) {
          return Long.compare(o1.getElapsedTimeNanos(), o2.getElapsedTimeNanos());
        }
      }
  );

  private final Object lock = new Object();

  protected CriticalPathComputer(Clock clock) {
    this.clock = clock;
    maxCriticalPath = null;
  }

  /**
   * Creates a critical path component for an action.
   * @param action the action for the critical path component
   * @param relativeStartNanos time when the action started to run in nanos. Only mean to be used
   * for computing time differences.
   */
  protected abstract C createComponent(Action action, long relativeStartNanos);

  /**
   * Return the critical path stats for the current command execution.
   *
   * <p>This method allows us to calculate lazily the aggregate statistics of the critical path,
   * avoiding the memory and cpu penalty for doing it for all the actions executed.
   */
  public abstract A aggregate();

  /**
   * Record an action that has started to run.
   *
   * @param event information about the started action
   */
  @Subscribe
  public void actionStarted(ActionStartedEvent event) {
    Action action = event.getAction();
    C component = createComponent(action, event.getNanoTimeStart());
    for (Artifact output : action.getOutputs()) {
      C old = outputArtifactToComponent.put(output, component);
      Preconditions.checkState(old == null, "Duplicate output artifact found. This could happen"
          + " if a previous event registered the action %s. Artifact: %s", action, output);
    }
  }

  /**
   * Record a middleman action execution. Even if middleman are almost instant, we record them
   * because they depend on other actions and we need them for constructing the critical path.
   *
   * <p>For some rules with incorrect configuration transitions we might get notified several times
   * for the same middleman. This should only happen if the actions are shared.
   */
  @Subscribe
  public void middlemanAction(ActionMiddlemanEvent event) {
    Action action = event.getAction();
    C component = createComponent(action, event.getNanoTimeStart());
    boolean duplicate = false;
    for (Artifact output : action.getOutputs()) {
      C old = outputArtifactToComponent.putIfAbsent(output, component);
      if (old != null) {
        if (!Actions.canBeShared(action, old.getAction())) {
          throw new IllegalStateException("Duplicate output artifact found for middleman."
              + "This could happen  if a previous event registered the action.\n"
              + "Old action: " + old.getAction() + "\n\n"
              + "New action: " + action + "\n\n"
              + "Artifact: " + output + "\n");
        }
        duplicate = true;
      }
    }
    if (!duplicate) {
      finalizeActionStat(action, component);
    }
  }

  /**
   * Record an action that was not executed because it was in the (disk) cache. This is needed so
   * that we can calculate correctly the dependencies tree if we have some cached actions in the
   * middle of the critical path.
   */
  @Subscribe
  public void actionCached(CachedActionEvent event) {
    Action action = event.getAction();
    C component = createComponent(action, event.getNanoTimeStart());
    for (Artifact output : action.getOutputs()) {
      outputArtifactToComponent.put(output, component);
    }
    finalizeActionStat(action, component);
  }

  /**
   * Records the elapsed time stats for the action. For each input artifact, it finds the real
   * dependent artifacts and records the critical path stats.
   */
  @Subscribe
  public void actionComplete(ActionCompletionEvent event) {
    ActionMetadata action = event.getActionMetadata();
    C component = Preconditions.checkNotNull(
        outputArtifactToComponent.get(action.getPrimaryOutput()));
    finalizeActionStat(action, component);
  }

  /** Maximum critical path component found during the build. */
  protected C getMaxCriticalPath() {
    synchronized (lock) {
      return maxCriticalPath;
    }
  }

  /**
   * The list of slowest individual components, ignoring the time to build dependencies.
   */
  public ImmutableList<C> getSlowestComponents() {
    ArrayList<C> list;
    synchronized (lock) {
      list = new ArrayList<>(slowestComponents);
      Collections.sort(list, slowestComponents.comparator());
    }
    return ImmutableList.copyOf(list).reverse();
  }

  private void finalizeActionStat(ActionMetadata action, C component) {
    component.setRelativeStartNanos(clock.nanoTime());
    for (Artifact input : action.getInputs()) {
      addArtifactDependency(component, input);
    }

    synchronized (lock) {
      if (isBiggestCriticalPath(component)) {
        maxCriticalPath = component;
      }

      if (slowestComponents.size() == SLOWEST_COMPONENTS_SIZE) {
        // The new component is faster than any of the slow components, avoid insertion.
        if (slowestComponents.peek().getElapsedTimeNanos() >= component.getElapsedTimeNanos()) {
          return;
        }
        // Remove the head element to make space (The fastest component in the queue).
        slowestComponents.remove();
      }
      slowestComponents.add(component);
    }
  }

  private boolean isBiggestCriticalPath(C newCriticalPath) {
    synchronized (lock) {
      return maxCriticalPath == null
          || maxCriticalPath.getAggregatedElapsedTimeMillis()
          < newCriticalPath.getAggregatedElapsedTimeMillis();
    }
  }

  /**
   * If "input" is a generated artifact, link its critical path to the one we're building.
   */
  private void addArtifactDependency(C actionStats, Artifact input) {
    C depComponent = outputArtifactToComponent.get(input);
    if (depComponent != null) {
      actionStats.addDepInfo(depComponent);
    }
  }
}

