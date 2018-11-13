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
import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.EventBus;
import com.google.common.eventbus.Subscribe;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionResultReceivedEvent;
import com.google.devtools.build.lib.buildeventstream.BuildToolLogs;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionStartingEvent;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.ExecutorBuilder;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.util.Pair;
import com.google.protobuf.ByteString;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

/**
 * Blaze module for the build summary message that reports various stats to the user.
 */
public class BuildSummaryStatsModule extends BlazeModule {

  private static final Logger logger = Logger.getLogger(BuildSummaryStatsModule.class.getName());

  private ActionKeyContext actionKeyContext;
  private CriticalPathComputer criticalPathComputer;
  private EventBus eventBus;
  private Reporter reporter;
  private boolean enabled;
  private boolean discardActions;

  private SpawnStats spawnStats;

  @Override
  public void beforeCommand(CommandEnvironment env) {
    this.reporter = env.getReporter();
    this.eventBus = env.getEventBus();
    this.actionKeyContext = env.getSkyframeExecutor().getActionKeyContext();
    this.spawnStats = new SpawnStats();
    eventBus.register(this);
  }

  @Override
  public void afterCommand() {
    this.criticalPathComputer = null;
    this.eventBus = null;
    this.reporter = null;
    this.spawnStats = null;
  }

  @Override
  public void executorInit(CommandEnvironment env, BuildRequest request, ExecutorBuilder builder) {
    enabled = env.getOptions().getOptions(ExecutionOptions.class).enableCriticalPathProfiling;
    discardActions = !env.getSkyframeExecutor().tracksStateForIncrementality();
  }

  @Subscribe
  public void executionPhaseStarting(ExecutionStartingEvent event) {
    if (enabled) {
      criticalPathComputer =
          new CriticalPathComputer(actionKeyContext, BlazeClock.instance(), discardActions);
      eventBus.register(criticalPathComputer);
    }
  }

  @Subscribe
  @AllowConcurrentEvents
  public void actionResultReceived(ActionResultReceivedEvent event) {
    spawnStats.countActionResult(event.getActionResult());
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    try {
      // We might want to make this conditional on a flag; it can sometimes be a bit of a nuisance.
      List<Pair<String, ByteString>> statistics = new ArrayList<>();
      List<String> items = new ArrayList<>();
      items.add(String.format("Elapsed time: %.3fs", event.getResult().getElapsedSeconds()));
      statistics.add(Pair.of("elapsed time", ByteString.copyFromUtf8(
          String.format("%f", event.getResult().getElapsedSeconds()))));

      if (criticalPathComputer != null) {
        try (SilentCloseable c =
            Profiler.instance().profile(ProfilerTask.CRITICAL_PATH, "Critical path")) {
          AggregatedCriticalPath criticalPath =
              criticalPathComputer.aggregate();
          items.add(criticalPath.toStringSummaryNoRemote());
          statistics.add(
              Pair.of("critical path", ByteString.copyFromUtf8(criticalPath.toString())));
          logger.info(criticalPath.toString());
          logger.info(
              "Slowest actions:\n  "
                  + Joiner.on("\n  ").join(criticalPathComputer.getSlowestComponents()));
          // We reverse the critical path because the profiler expect events ordered by the time
          // when the actions were executed while critical path computation is stored in the reverse
          // way.
          for (CriticalPathComponent stat : criticalPath.components().reverse()) {
            Profiler.instance()
                .logSimpleTaskDuration(
                    stat.getStartTimeNanos(),
                    stat.getElapsedTime(),
                    ProfilerTask.CRITICAL_PATH_COMPONENT,
                    stat.prettyPrintAction());
          }
        }
      }

      reporter.handle(Event.info(Joiner.on(", ").join(items)));

      String spawnSummary = spawnStats.getSummary();
      reporter.handle(Event.info(spawnSummary));
      statistics.add(Pair.of("process stats", ByteString.copyFromUtf8(spawnSummary)));

      reporter.post(new BuildToolLogs(statistics, ImmutableList.of(), ImmutableList.of()));
    } finally {
      criticalPathComputer = null;
    }
  }
}
