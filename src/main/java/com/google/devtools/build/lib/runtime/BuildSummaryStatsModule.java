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

import com.google.common.base.Joiner;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.BlazeClock;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Blaze module for the build summary message that reports various stats to the user.
 */
public class BuildSummaryStatsModule extends BlazeModule {

  private static final Logger LOG = Logger.getLogger(BuildSummaryStatsModule.class.getName());

  private SimpleCriticalPathComputer criticalPathComputer;
  private EventBus eventBus;
  private Reporter reporter;
  private boolean enabled;
  private boolean discardActions;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    this.reporter = env.getReporter();
    this.eventBus = env.getEventBus();
    eventBus.register(this);
  }

  @Override
  public void afterCommand() {
    this.criticalPathComputer = null;
    this.eventBus = null;
    this.reporter = null;
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    enabled = env.getOptions().getOptions(ExecutionOptions.class).enableCriticalPathProfiling;
    discardActions = !env.getSkyframeExecutor().hasIncrementalState();
  }

  @Subscribe
  public void executionPhaseStarting(ExecutionStartingEvent event) {
    if (enabled) {
      criticalPathComputer = new SimpleCriticalPathComputer(BlazeClock.instance(), discardActions);
      eventBus.register(criticalPathComputer);
    }
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    try {
      // We might want to make this conditional on a flag; it can sometimes be a bit of a nuisance.
      List<String> items = new ArrayList<>();
      items.add(String.format("Elapsed time: %.3fs", event.getResult().getElapsedSeconds()));

      if (criticalPathComputer != null) {
        Profiler.instance().startTask(ProfilerTask.CRITICAL_PATH, "Critical path");
        AggregatedCriticalPath<SimpleCriticalPathComponent> criticalPath =
            criticalPathComputer.aggregate();
        items.add(criticalPath.toStringSummary());
        LOG.info(criticalPath.toString());
        LOG.info("Slowest actions:\n  " + Joiner.on("\n  ")
            .join(criticalPathComputer.getSlowestComponents()));
        // We reverse the critical path because the profiler expect events ordered by the time
        // when the actions were executed while critical path computation is stored in the reverse
        // way.
        for (SimpleCriticalPathComponent stat : criticalPath.components().reverse()) {
          Profiler.instance()
              .logSimpleTaskDuration(
                  stat.getStartNanos(),
                  stat.getElapsedTimeNanos(),
                  ProfilerTask.CRITICAL_PATH_COMPONENT,
                  stat.prettyPrintAction());
        }
        Profiler.instance().completeTask(ProfilerTask.CRITICAL_PATH);
      }

      reporter.handle(Event.info(Joiner.on(", ").join(items)));
    } finally {
      criticalPathComputer = null;
    }
  }
}
