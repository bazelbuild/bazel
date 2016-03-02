// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import com.google.common.base.Supplier;
import com.google.common.collect.Sets;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.analysis.AspectCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetCompleteEvent;
import com.google.devtools.build.lib.skyframe.ActionExecutionInactivityWatchdog;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue;
import com.google.devtools.build.lib.skyframe.SkyFunctions;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor;
import com.google.devtools.build.lib.skyframe.TargetCompletionValue;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;

import java.text.NumberFormat;
import java.util.Collections;
import java.util.Locale;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Listener for executed actions and built artifacts. We use a listener so that we have an
 * accurate set of successfully run actions and built artifacts, even if the build is interrupted.
 */
public final class ExecutionProgressReceiver
    implements EvaluationProgressReceiver, SkyframeActionExecutor.ProgressSupplier,
        SkyframeActionExecutor.ActionCompletedReceiver {
  private static final NumberFormat PROGRESS_MESSAGE_NUMBER_FORMATTER;

  // Must be thread-safe!
  private final Set<ConfiguredTarget> builtTargets;
  private final Set<SkyKey> enqueuedActions = Sets.newConcurrentHashSet();
  private final Set<Action> completedActions = Sets.newConcurrentHashSet();
  private final Object activityIndicator = new Object();
  /** Number of exclusive tests. To be accounted for in progress messages. */
  private final int exclusiveTestsCount;
  private final EventBus eventBus;

  static {
    PROGRESS_MESSAGE_NUMBER_FORMATTER = NumberFormat.getIntegerInstance(Locale.ENGLISH);
    PROGRESS_MESSAGE_NUMBER_FORMATTER.setGroupingUsed(true);
  }

  /**
   * {@code builtTargets} is accessed through a synchronized set, and so no other access to it
   * is permitted while this receiver is active.
   */
  ExecutionProgressReceiver(
      Set<ConfiguredTarget> builtTargets, int exclusiveTestsCount, EventBus eventBus) {
    this.builtTargets = Collections.synchronizedSet(builtTargets);
    this.exclusiveTestsCount = exclusiveTestsCount;
    this.eventBus = eventBus;
  }

  @Override
  public void invalidated(SkyKey skyKey, InvalidationState state) {}

  @Override
  public void enqueueing(SkyKey skyKey) {
    if (ActionExecutionValue.isReportWorthyAction(skyKey)) {
      // Remember all enqueued actions for the benefit of progress reporting.
      // We discover most actions early in the build, well before we start executing them.
      // Some of these will be cache hits and won't be executed, so we'll need to account for them
      // in the evaluated method too.
      enqueuedActions.add(skyKey);
    }
  }

  @Override
  public void computed(SkyKey skyKey, long elapsedTimeNanos) {}

  @Override
  public void evaluated(SkyKey skyKey, Supplier<SkyValue> skyValueSupplier, EvaluationState state) {
    SkyFunctionName type = skyKey.functionName();
    if (type.equals(SkyFunctions.TARGET_COMPLETION)) {
      TargetCompletionValue value = (TargetCompletionValue) skyValueSupplier.get();
      if (value != null) {
        ConfiguredTarget target = value.getConfiguredTarget();
        builtTargets.add(target);
        eventBus.post(TargetCompleteEvent.createSuccessful(target));
      }
    } else if (type.equals(SkyFunctions.ASPECT_COMPLETION)) {
      AspectCompletionValue value = (AspectCompletionValue) skyValueSupplier.get();
      if (value != null) {
        eventBus.post(AspectCompleteEvent.createSuccessful(value.getAspectValue()));
      }
    } else if (type.equals(SkyFunctions.ACTION_EXECUTION)) {
      // Remember all completed actions, even those in error, regardless of having been cached or
      // really executed.
      actionCompleted((Action) skyKey.argument());
    }
  }

  /**
   * {@inheritDoc}
   *
   * <p>This method adds the action to {@link #completedActions} and notifies the
   * {@link #activityIndicator}.
   *
   * <p>We could do this only in the {@link #evaluated} method too, but as it happens the action
   * executor tells the reporter about the completed action before the node is inserted into the
   * graph, so the reporter would find out about the completed action sooner than we could
   * have updated {@link #completedActions}, which would result in incorrect numbers on the
   * progress messages. However we have to store completed actions in {@link #evaluated} too,
   * because that's the only place we get notified about completed cached actions.
   */
  @Override
  public void actionCompleted(Action a) {
    if (ActionExecutionValue.isReportWorthyAction(a)) {
      completedActions.add(a);
      synchronized (activityIndicator) {
        activityIndicator.notifyAll();
      }
    }
  }

  @Override
  public String getProgressString() {
    return String.format(
        "[%s / %s]",
        PROGRESS_MESSAGE_NUMBER_FORMATTER.format(completedActions.size()),
        PROGRESS_MESSAGE_NUMBER_FORMATTER.format(exclusiveTestsCount + enqueuedActions.size()));
  }

  ActionExecutionInactivityWatchdog.InactivityMonitor createInactivityMonitor(
      final ActionExecutionStatusReporter statusReporter) {
    return new ActionExecutionInactivityWatchdog.InactivityMonitor() {

      @Override
      public boolean hasStarted() {
        return !enqueuedActions.isEmpty();
      }

      @Override
      public int getPending() {
        return statusReporter.getCount();
      }

      @Override
      public int waitForNextCompletion(int timeoutMilliseconds) throws InterruptedException {
        synchronized (activityIndicator) {
          int before = completedActions.size();
          long startTime = BlazeClock.instance().currentTimeMillis();
          while (true) {
            activityIndicator.wait(timeoutMilliseconds);

            int completed = completedActions.size() - before;
            long now = 0;
            if (completed > 0
                || (startTime + timeoutMilliseconds)
                    <= (now = BlazeClock.instance().currentTimeMillis())) {
              // Some actions completed, or timeout fully elapsed.
              return completed;
            } else {
              // Spurious Wakeup -- no actions completed and there's still time to wait.
              timeoutMilliseconds -= now - startTime; // account for elapsed wait time
              startTime = now;
            }
          }
        }
      }
    };
  }

  ActionExecutionInactivityWatchdog.InactivityReporter createInactivityReporter(
      final ActionExecutionStatusReporter statusReporter,
      final AtomicBoolean isBuildingExclusiveArtifacts) {
    return new ActionExecutionInactivityWatchdog.InactivityReporter() {
      @Override
      public void maybeReportInactivity() {
        // Do not report inactivity if we are currently running an exclusive test or a streaming
        // action (in practice only tests can stream and it implicitly makes them exclusive).
        if (!isBuildingExclusiveArtifacts.get()) {
          statusReporter.showCurrentlyExecutingActions(
              ExecutionProgressReceiver.this.getProgressString() + " ");
        }
      }
    };
  }
}
