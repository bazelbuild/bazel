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
package com.google.devtools.build.lib.actions;

import static java.util.Comparator.comparing;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Pair;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import javax.annotation.Nullable;

/**
 * Implements "Still waiting..." message functionality, displaying current status for "in-flight"
 * actions. Used by the ParallelBuilder.
 *
 * TODO(bazel-team): (2010) It would be nice if "duplicated" actions (e.g. test shards and multiple
 * test runs) were merged into the single line.
 */
@ThreadSafe
public final class ActionExecutionStatusReporter {
  // Maximum number of lines to output per each status category before truncation.
  private static final int MAX_LINES = 10;

  private static final String PREPARING_MESSAGE = "Preparing";

  private final EventHandler eventHandler;
  private final EventBus eventBus;
  private final Clock clock;

  /**
   * The status of each action "in flight", i.e. whose ExecuteBuildAction.call() method is active.
   * Used for implementing the "still waiting" message.
   */
  private final Map<ActionExecutionMetadata, Pair<String, Long>> actionStatus =
      new ConcurrentHashMap<>(100);

  public static ActionExecutionStatusReporter create(EventHandler eventHandler) {
    return create(eventHandler, null, null);
  }

  @VisibleForTesting
  static ActionExecutionStatusReporter create(EventHandler eventHandler, @Nullable Clock clock) {
    return create(eventHandler, null, clock);
  }

  public static ActionExecutionStatusReporter create(
      EventHandler eventHandler, @Nullable EventBus eventBus) {
    return create(eventHandler, eventBus, null);
  }

  private static ActionExecutionStatusReporter create(
      EventHandler eventHandler, @Nullable EventBus eventBus, @Nullable Clock clock) {
    ActionExecutionStatusReporter result =
        new ActionExecutionStatusReporter(
            eventHandler, eventBus, clock == null ? BlazeClock.instance() : clock);
    if (eventBus != null) {
      eventBus.register(result);
    }
    return result;
  }

  private ActionExecutionStatusReporter(
      EventHandler eventHandler, @Nullable EventBus eventBus, Clock clock) {
    this.eventHandler = Preconditions.checkNotNull(eventHandler);
    this.eventBus = eventBus;
    this.clock = Preconditions.checkNotNull(clock);
  }

  public void unregisterFromEventBus() {
    if (eventBus != null) {
      eventBus.unregister(this);
    }
  }

  private void setStatus(ActionExecutionMetadata action, String message) {
    actionStatus.put(action, Pair.of(message, clock.nanoTime()));
  }

  /**
   * Remove action from the list of active actions.
   */
  public void remove(Action action) {
    Preconditions.checkNotNull(actionStatus.remove(action), action);
  }

  @Subscribe
  @AllowConcurrentEvents
  public void updateStatus(ActionStartedEvent event) {
    ActionExecutionMetadata action = event.getAction();
    setStatus(action, PREPARING_MESSAGE);
  }

  @Subscribe
  @AllowConcurrentEvents
  public void updateStatus(AnalyzingActionEvent event) {
    ActionExecutionMetadata action = event.getActionMetadata();
    setStatus(action, "Analyzing");
  }

  @Subscribe
  @AllowConcurrentEvents
  public void updateStatus(SchedulingActionEvent event) {
    ActionExecutionMetadata action = event.getActionMetadata();
    setStatus(action, "Scheduling");
  }

  @Subscribe
  @AllowConcurrentEvents
  public void updateStatus(RunningActionEvent event) {
    ActionExecutionMetadata action = event.getActionMetadata();
    setStatus(action, String.format("Running (%s)", event.getStrategy()));
  }

  public int getCount() {
    return actionStatus.size();
  }

  private static void appendGroupStatus(StringBuilder buffer,
      Map<ActionExecutionMetadata, Pair<String, Long>> statusMap,  String status,
      long currentTime) {
    List<Pair<Long, ActionExecutionMetadata>> actions = new ArrayList<>();
    for (Map.Entry<ActionExecutionMetadata, Pair<String, Long>> entry : statusMap.entrySet()) {
      if (entry.getValue().first.equals(status)) {
        actions.add(Pair.of(entry.getValue().second, entry.getKey()));
      }
    }
    if (actions.isEmpty()) {
      return;
    }
    Collections.sort(actions, comparing(arg -> arg.first));

    buffer.append("\n      " + status + ":");

    boolean truncateList = actions.size() > MAX_LINES;
    for (Pair<Long, ActionExecutionMetadata> entry : actions.subList(0,
        truncateList ? MAX_LINES - 1 : actions.size())) {
      String message = entry.second.getProgressMessage();
      if (message == null) {
        // Actions will a null progress message should run so
        // fast we never see them here.  In any case...
        message = entry.second.prettyPrint();
      }
      buffer.append("\n        ").append(message);
      long runTime = (currentTime - entry.first) / 1000000000L; // Convert to seconds.
      buffer.append(", ").append(runTime).append(" s");
    }
    if (truncateList) {
      buffer.append("\n        ... ").append(actions.size() - MAX_LINES + 1).append(" more jobs");
    }
  }

  /**
   * Get message showing currently executing actions.
   */
  private String getExecutionStatusMessage(
      Map<ActionExecutionMetadata, Pair<String, Long>> statusMap) {
    int count = statusMap.size();
    StringBuilder s = count != 1
        ? new StringBuilder("Still waiting for ").append(count).append(" jobs to complete:")
        : new StringBuilder("Still waiting for 1 job to complete:");

    long currentTime = clock.nanoTime();

    // A tree is just as fast as HashSet for small data sets.
    Set<String> statuses = new TreeSet<>();
    for (Map.Entry<ActionExecutionMetadata, Pair<String, Long>> entry : statusMap.entrySet()) {
      statuses.add(entry.getValue().first);
    }

    for (String status : statuses) {
      appendGroupStatus(s, statusMap, status, currentTime);
    }
    return s.toString();
  }

  /**
   * Show currently executing actions.
   */
  public void showCurrentlyExecutingActions(String progressPercentageMessage) {
    // Defensive copy to ensure thread safety.
    Map<ActionExecutionMetadata, Pair<String, Long>> statusMap = new HashMap<>(actionStatus);
    if (!statusMap.isEmpty()) {
      eventHandler.handle(
          Event.progress(progressPercentageMessage + getExecutionStatusMessage(statusMap)));
    }
  }

  /**
   * Warn about actions that are still being executed.
   * Method is used to produce informative message when build is interrupted.
   */
  void warnAboutCurrentlyExecutingActions() {
    // Defensive copy to ensure thread safety.
    Map<ActionExecutionMetadata, Pair<String, Long>> statusMap = new HashMap<>(actionStatus);
    if (statusMap.isEmpty()) {
      // There are no tasks in the queue so there is nothing to report.
      eventHandler.handle(Event.warn("There are no active jobs - stopping the build"));
      return;
    }
    Iterator<ActionExecutionMetadata> iterator = statusMap.keySet().iterator();
    while (iterator.hasNext()) {
      // Filter out actions that are not executed yet.
      if (PREPARING_MESSAGE.equals(statusMap.get(iterator.next()).first)) {
        iterator.remove();
      }
    }
    if (!statusMap.isEmpty()) {
      eventHandler.handle(Event.warn(getExecutionStatusMessage(statusMap)
          + "\nBuild will be stopped after these tasks terminate"));
    } else {
      // It is possible that one or more tasks in "Preparing" state just started being executed.
      // So warn user just in case.
      eventHandler.handle(Event.warn("Still waiting for unfinished jobs"));
    }
  }

  /**
   * Returns the number of seconds to wait before reporting slow progress again.
   *
   * @param userSpecifiedProgressInterval value of the --progress_report_interval flag; 0 means
   *     use default 10, then 30, then 60 seconds wait times
   * @param previousWaitTime previous value returned by this method
   */
  public static int getWaitTime(int userSpecifiedProgressInterval, int previousWaitTime) {
    if (userSpecifiedProgressInterval > 0) {
      return userSpecifiedProgressInterval;
    }

    // Increase waitTime to 10, then to 30 and then to 60 seconds to reduce
    // spamming during long wait periods.  If the user specified a
    // waitTime directly through progressReportInterval, then use
    // that value.
    if (previousWaitTime == 0) {
      return 10;
    } else if (previousWaitTime == 10) {
      return 30;
    } else {
      return 60;
    }
  }
}
