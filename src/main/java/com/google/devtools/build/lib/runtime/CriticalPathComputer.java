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

import static java.util.Comparator.comparingLong;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionMiddlemanEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Objects;
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
  protected final boolean discardActions;

  /**
   * The list of slowest individual components, ignoring the time to build dependencies.
   *
   * <p>This data is a useful metric when running non highly incremental builds, where multiple
   * tasks could run un parallel and critical path would only record the longest path.
   */
  private final PriorityQueue<C> slowestComponents =
      new PriorityQueue<>(SLOWEST_COMPONENTS_SIZE, comparingLong(C::getElapsedTimeNanos));

  private final Object lock = new Object();

  protected CriticalPathComputer(Clock clock, boolean discardActions) {
    this.clock = clock;
    this.discardActions = discardActions;
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
    tryAddComponent(createComponent(action, event.getNanoTimeStart()));
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
    C component = tryAddComponent(createComponent(action, event.getNanoTimeStart()));
    finalizeActionStat(event.getNanoTimeStart(), action, component);
  }

  /**
   * Try to add the component to the map of critical path components. If there is an existing
   * component for its primary output it uses that to update the rest of the outputs.
   *
   * @return The component to be used for updating the time stats.
   */
  private C tryAddComponent(C newComponent) {
    Action newAction = Preconditions.checkNotNull(newComponent.maybeGetAction(), newComponent);
    Artifact primaryOutput = newAction.getPrimaryOutput();
    C storedComponent = outputArtifactToComponent.putIfAbsent(primaryOutput, newComponent);

    if (storedComponent != null) {
      Action oldAction = storedComponent.maybeGetAction();
      if (oldAction != null) {
        if (!Actions.canBeShared(newAction, oldAction)) {
          throw new IllegalStateException(
              "Duplicate output artifact found for unsharable actions."
                  + "This can happen if a previous event registered the action.\n"
                  + "Old action: "
                  + oldAction
                  + "\n\nNew action: "
                  + newAction
                  + "\n\nArtifact: "
                  + primaryOutput
                  + "\n");
        }
      } else {
        String mnemonic = storedComponent.getMnemonic();
        String prettyPrint = storedComponent.prettyPrintAction();
        if (!newAction.getMnemonic().equals(mnemonic)
            || !newAction.prettyPrint().equals(prettyPrint)) {
          throw new IllegalStateException(
              "Duplicate output artifact found for unsharable actions."
                  + "This can happen if a previous event registered the action.\n"
                  + "Old action mnemonic and prettyPrint: "
                  + mnemonic
                  + ", "
                  + prettyPrint
                  + "\n\nNew action: "
                  + newAction
                  + "\n\nArtifact: "
                  + primaryOutput
                  + "\n");
        }
      }
    } else {
      storedComponent = newComponent;
    }
    // Try to insert the existing component for the rest of the outputs even if we failed to be
    // the ones inserting the component so that at the end of this method we guarantee that all the
    // outputs have a component.
    for (Artifact output : newAction.getOutputs()) {
      if (output == primaryOutput) {
        continue;
      }
      C old = outputArtifactToComponent.putIfAbsent(output, storedComponent);
      // If two actions run concurrently maybe we find a component by primary output but we are
      // the first updating the rest of the outputs.
      Preconditions.checkState(old == null || old == storedComponent,
          "Inconsistent state for %s", newAction);
    }
    return storedComponent;
  }

  /**
   * Record an action that was not executed because it was in the (disk) cache. This is needed so
   * that we can calculate correctly the dependencies tree if we have some cached actions in the
   * middle of the critical path.
   */
  @Subscribe
  public void actionCached(CachedActionEvent event) {
    Action action = event.getAction();
    C component = tryAddComponent(createComponent(action, event.getNanoTimeStart()));
    finalizeActionStat(event.getNanoTimeStart(), action, component);
  }

  /**
   * Records the elapsed time stats for the action. For each input artifact, it finds the real
   * dependent artifacts and records the critical path stats.
   */
  @Subscribe
  public void actionComplete(ActionCompletionEvent event) {
    Action action = event.getAction();
    C component = Preconditions.checkNotNull(
        outputArtifactToComponent.get(action.getPrimaryOutput()));
    finalizeActionStat(event.getRelativeActionStartTime(), action, component);
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

  private void finalizeActionStat(long startTimeNanos, Action action, C component) {

    for (Artifact input : action.getInputs()) {
      addArtifactDependency(component, input);
    }

    boolean updated = component.finishActionExecution(startTimeNanos, clock.nanoTime());
    synchronized (lock) {
      if (isBiggestCriticalPath(component)) {
        maxCriticalPath = component;
      }

      // We do not want to fill slow components list with the same component.
      //
      // This might still insert a second copy of the component but only if the new self elapsed
      // time is greater than the old time. That said, in practice this is not important, since
      // this would happen when we have two concurrent shared actions and one is a cache hit
      // because of the other one. In this case, the cache hit would not appear in the 30 slowest
      // actions or we had a very fast build, so we do not care :).
      if (updated) {
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
      Action action = depComponent.maybeGetAction();
      if (depComponent.isRunning && action != null) {
        // Rare case that an action depending on a previously-cached shared action sees a different
        // shared action that is in the midst of being an action cache hit.
        for (Artifact actionOutput : action.getOutputs()) {
          if (input.equals(actionOutput)
              && Objects.equals(input.getArtifactOwner(), actionOutput.getArtifactOwner())) {
            // As far as we can tell, this (currently running) action is the same action that
            // produced input, not another shared action. This should be impossible.
            throw new IllegalStateException(
                String.format(
                    "Cannot add critical path stats when the action is not finished. %s. %s. %s",
                    input, actionStats.prettyPrintAction(), action));
          }
        }
        return;
      }
      actionStats.addDepInfo(depComponent);
    }
  }
}

