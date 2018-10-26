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

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.util.Comparator.comparingLong;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Maps;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionMiddlemanEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Actions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CachedActionEvent;
import com.google.devtools.build.lib.actions.SpawnExecutedEvent;
import com.google.devtools.build.lib.actions.SpawnMetrics;
import com.google.devtools.build.lib.actions.SpawnResult;
import com.google.devtools.build.lib.clock.Clock;
import java.time.Duration;
import java.util.Objects;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BinaryOperator;
import javax.annotation.concurrent.ThreadSafe;

/**
 * Computes the critical path in the action graph based on events published to the event bus.
 *
 * <p>After instantiation, this object needs to be registered on the event bus to work.
 */
@ThreadSafe
public class CriticalPathComputer {
  /** Number of top actions to record. */
  static final int SLOWEST_COMPONENTS_SIZE = 30;
  private static final int LARGEST_MEMORY_COMPONENTS_SIZE = 20;
  private static final int LARGEST_INPUT_SIZE_COMPONENTS_SIZE = 20;

  /** Selects and returns the longer of two components (the first may be {@code null}). */
  private static final BinaryOperator<CriticalPathComponent> SELECT_LONGER_COMPONENT =
      (a, b) ->
          a == null || a.getAggregatedElapsedTime().compareTo(b.getAggregatedElapsedTime()) < 0
              ? b
              : a;

  private final AtomicInteger idGenerator = new AtomicInteger();
  // outputArtifactToComponent is accessed from multiple event handlers.
  protected final ConcurrentMap<Artifact, CriticalPathComponent> outputArtifactToComponent =
      Maps.newConcurrentMap();
  private final ActionKeyContext actionKeyContext;

  /** Maximum critical path found. */
  private final AtomicReference<CriticalPathComponent> maxCriticalPath;
  private final Clock clock;
  protected final boolean discardActions;

  protected CriticalPathComputer(
      ActionKeyContext actionKeyContext, Clock clock, boolean discardActions) {
    this.actionKeyContext = actionKeyContext;
    this.clock = clock;
    this.discardActions = discardActions;
    maxCriticalPath = new AtomicReference<>();
  }

  /**
   * Creates a critical path component for an action.
   * @param action the action for the critical path component
   * @param relativeStartNanos time when the action started to run in nanos. Only mean to be used
   * for computing time differences.
   */
  public CriticalPathComponent createComponent(Action action, long relativeStartNanos) {
    int id = idGenerator.getAndIncrement();
    return discardActions
        ? new ActionDiscardingCriticalPathComponent(id, action, relativeStartNanos)
        : new CriticalPathComponent(id, action, relativeStartNanos);
  }

  /**
   * Return the critical path stats for the current command execution.
   *
   * <p>This method allows us to calculate lazily the aggregate statistics of the critical path,
   * avoiding the memory and cpu penalty for doing it for all the actions executed.
   */
  public AggregatedCriticalPath aggregate() {
    CriticalPathComponent criticalPath = getMaxCriticalPath();
    Duration totalTime = Duration.ZERO;
    Duration parseTime = Duration.ZERO;
    Duration networkTime = Duration.ZERO;
    Duration fetchTime = Duration.ZERO;
    Duration remoteQueueTime = Duration.ZERO;
    Duration uploadTime = Duration.ZERO;
    Duration setupTime = Duration.ZERO;
    Duration executionWallTime = Duration.ZERO;
    Duration retryTime = Duration.ZERO;
    long inputFiles = 0L;
    long inputBytes = 0L;
    long memoryEstimate = 0L;
    ImmutableList.Builder<CriticalPathComponent> components = ImmutableList.builder();
    if (criticalPath == null) {
      return AggregatedCriticalPath.EMPTY;
    }
    CriticalPathComponent child = criticalPath;
    while (child != null) {
      SpawnMetrics childSpawnMetrics = child.getSpawnMetrics();
      if (childSpawnMetrics != null) {
        totalTime = totalTime.plus(childSpawnMetrics.totalTime());
        parseTime = parseTime.plus(childSpawnMetrics.parseTime());
        networkTime = networkTime.plus(childSpawnMetrics.networkTime());
        fetchTime = fetchTime.plus(childSpawnMetrics.fetchTime());
        remoteQueueTime = remoteQueueTime.plus(childSpawnMetrics.remoteQueueTime());
        uploadTime = uploadTime.plus(childSpawnMetrics.uploadTime());
        setupTime = setupTime.plus(childSpawnMetrics.setupTime());
        executionWallTime = executionWallTime.plus(childSpawnMetrics.executionWallTime());
        retryTime = retryTime.plus(childSpawnMetrics.retryTime());
        inputBytes += childSpawnMetrics.inputBytes();
        inputFiles += childSpawnMetrics.inputFiles();
        memoryEstimate += childSpawnMetrics.memoryEstimate();
      }
      components.add(child);
      child = child.getChild();
    }

    return new AggregatedCriticalPath(
        criticalPath.getAggregatedElapsedTime(),
        new SpawnMetrics(
            totalTime,
            parseTime,
            networkTime,
            fetchTime,
            remoteQueueTime,
            setupTime,
            uploadTime,
            executionWallTime,
            retryTime,
            inputBytes,
            inputFiles,
            memoryEstimate),
        components.build());
  }

  /** Adds spawn metrics to the action stats. */
  @Subscribe
  @AllowConcurrentEvents
  public void spawnExecuted(SpawnExecutedEvent event) {
    ActionAnalysisMetadata action = event.getActionMetadata();
    Artifact primaryOutput = action.getPrimaryOutput();
    if (primaryOutput == null) {
      // Despite the documentation to the contrary, the SpawnIncludeScanner creates an
      // ActionExecutionMetadata instance that returns a null primary output. That said, this
      // class is incorrect wrt. multiple Spawns in a single action. See b/111583707.
      return;
    }
    CriticalPathComponent stats =
        Preconditions.checkNotNull(outputArtifactToComponent.get(primaryOutput));

    SpawnResult spawnResult = event.getSpawnResult();
    stats.addSpawnMetrics(spawnResult.getMetrics());
  }

  /** Returns the list of components using the most memory. */
  public ImmutableList<CriticalPathComponent> getLargestMemoryComponents() {
    return outputArtifactToComponent
        .values()
        .stream()
        .sorted(
            comparingLong((CriticalPathComponent a) -> a.getSpawnMetrics().memoryEstimate())
                .reversed())
        .limit(LARGEST_MEMORY_COMPONENTS_SIZE)
        .collect(toImmutableList());
  }

  /** Returns the list of components with the largest input sizes. */
  public ImmutableList<CriticalPathComponent> getLargestInputSizeComponents() {
    return outputArtifactToComponent
        .values()
        .stream()
        .sorted(
            comparingLong((CriticalPathComponent a) -> a.getSpawnMetrics().inputBytes())
                .reversed())
        .limit(LARGEST_INPUT_SIZE_COMPONENTS_SIZE)
        .collect(toImmutableList());
  }

  /** Returns the list of slowest components. */
  public ImmutableList<CriticalPathComponent> getSlowestComponents() {
    return outputArtifactToComponent
        .values()
        .stream()
        .sorted(
            comparingLong((CriticalPathComponent a) -> a.getElapsedTimeNanos())
                .reversed())
        .limit(SLOWEST_COMPONENTS_SIZE)
        .collect(toImmutableList());
  }

  /**
   * Record an action that has started to run.
   *
   * @param event information about the started action
   */
  @Subscribe
  @AllowConcurrentEvents
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
  @AllowConcurrentEvents
  public void middlemanAction(ActionMiddlemanEvent event) {
    Action action = event.getAction();
    CriticalPathComponent component =
        tryAddComponent(createComponent(action, event.getNanoTimeStart()));
    finalizeActionStat(event.getNanoTimeStart(), action, component);
  }

  /**
   * Try to add the component to the map of critical path components. If there is an existing
   * component for its primary output it uses that to update the rest of the outputs.
   *
   * @return The component to be used for updating the time stats.
   */
  private CriticalPathComponent tryAddComponent(CriticalPathComponent newComponent) {
    Action newAction = Preconditions.checkNotNull(newComponent.maybeGetAction(), newComponent);
    Artifact primaryOutput = newAction.getPrimaryOutput();
    CriticalPathComponent storedComponent =
        outputArtifactToComponent.putIfAbsent(primaryOutput, newComponent);

    if (storedComponent != null) {
      Action oldAction = storedComponent.maybeGetAction();
      if (oldAction != null) {
        if (!Actions.canBeShared(actionKeyContext, newAction, oldAction)) {
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
      CriticalPathComponent old = outputArtifactToComponent.putIfAbsent(output, storedComponent);
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
  @AllowConcurrentEvents
  public void actionCached(CachedActionEvent event) {
    Action action = event.getAction();
    CriticalPathComponent component
        = tryAddComponent(createComponent(action, event.getNanoTimeStart()));
    finalizeActionStat(event.getNanoTimeStart(), action, component);
  }

  /**
   * Records the elapsed time stats for the action. For each input artifact, it finds the real
   * dependent artifacts and records the critical path stats.
   */
  @Subscribe
  @AllowConcurrentEvents
  public void actionComplete(ActionCompletionEvent event) {
    Action action = event.getAction();
    CriticalPathComponent component = Preconditions.checkNotNull(
        outputArtifactToComponent.get(action.getPrimaryOutput()));
    finalizeActionStat(event.getRelativeActionStartTime(), action, component);
  }

  /** Maximum critical path component found during the build. */
  protected CriticalPathComponent getMaxCriticalPath() {
    return maxCriticalPath.get();
  }

  private void finalizeActionStat(
      long startTimeNanos, Action action, CriticalPathComponent component) {
    for (Artifact input : action.getInputs()) {
      addArtifactDependency(component, input);
    }

    component.finishActionExecution(startTimeNanos, clock.nanoTime());
    maxCriticalPath.accumulateAndGet(component, SELECT_LONGER_COMPONENT);
  }

  /**
   * If "input" is a generated artifact, link its critical path to the one we're building.
   */
  private void addArtifactDependency(CriticalPathComponent actionStats, Artifact input) {
    CriticalPathComponent depComponent = outputArtifactToComponent.get(input);
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

