// Copyright 2017 The Bazel Authors. All rights reserved.
// //
// // Licensed under the Apache License, Version 2.0 (the "License");
// // you may not use this file except in compliance with the License.
// // You may obtain a copy of the License at
// //
// //    http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing, software
// // distributed under the License is distributed on an "AS IS" BASIS,
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// // See the License for the specific language governing permissions and
// // limitations under the License.
package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableList;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionResult;
import com.google.devtools.build.lib.actions.ActionResultReceivedEvent;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.common.options.OptionsParsingResult;
import java.util.Properties;
import java.util.Optional;
import java.time.Duration;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;
import static org.mockito.Mockito.verify;

@RunWith(JUnit4.class)
public class BuildSummaryStatsModuleTest {
  private BuildSummaryStatsModule buildSummaryStatsModule;
  private Reporter reporterMock;

  @Before
  public void setUp() throws Exception {
    CommandEnvironment env = mock(CommandEnvironment.class);
    ActionKeyContext actionKeyContextMock = mock(ActionKeyContext.class);
    reporterMock = mock(Reporter.class);
    SkyframeExecutor skyframeExecutorMock = mock(SkyframeExecutor.class);
    EventBus eventBusMock = mock(EventBus.class);
    OptionsParsingResult optionsParsingResultMock = mock(OptionsParsingResult.class);
    ExecutionOptions executionOptions = new ExecutionOptions();
    executionOptions.statsSummary = true;
    when(env.getReporter()).thenReturn(reporterMock);
    when(env.getSkyframeExecutor()).thenReturn(skyframeExecutorMock);
    when(skyframeExecutorMock.getActionKeyContext()).thenReturn(actionKeyContextMock);
    when(env.getEventBus()).thenReturn(eventBusMock);
    when(env.getOptions()).thenReturn(optionsParsingResultMock);
    when(optionsParsingResultMock.getOptions(ExecutionOptions.class)).thenReturn(executionOptions);
    buildSummaryStatsModule = new BuildSummaryStatsModule();
    buildSummaryStatsModule.beforeCommand(env);
    buildSummaryStatsModule.executorInit(env, null, null);
  }

  private ActionResultReceivedEvent createActionEvent(Duration userTime, Duration systemTime) {
    ActionResult result = mock(ActionResult.class);
    when(result.cumulativeCommandExecutionUserTime()).thenReturn(Optional.ofNullable(userTime));
    when(result.cumulativeCommandExecutionSystemTime()).thenReturn(Optional.ofNullable(systemTime));
    when(result.spawnResults()).thenReturn(ImmutableList.of());
    return new ActionResultReceivedEvent(null, result);
  }

  private BuildCompleteEvent createBuildEvent() {
    BuildResult buildResult = new BuildResult(1000);
    buildResult.setStopTime(2000);
    return new BuildCompleteEvent(buildResult);
  }

  @Test
  public void allCpuTimesAreSummarized() throws Exception {
    ActionResultReceivedEvent action1 = createActionEvent(Duration.ofSeconds(50), Duration.ofSeconds(20));
    ActionResultReceivedEvent action2 = createActionEvent(Duration.ofSeconds(5), Duration.ofSeconds(2));
    buildSummaryStatsModule.actionResultReceived(action1);
    buildSummaryStatsModule.actionResultReceived(action2);
    buildSummaryStatsModule.setCpuTimeForBazelJvm(Duration.ofMillis(11000));
    buildSummaryStatsModule.buildComplete(createBuildEvent());
    verify(reporterMock).handle(Event.info(String.format("CPU time %.2fs (user %.2fs, system %.2fs, bazel jvm %.2fs)",88.00, 55.00, 22.00, 11.00)));
  }

  @Test
  public void mixOfActionsWithKnownAndUnknownCpuTimesResultInUnknownTimes() throws Exception {
    // First action with unknown values
    ActionResultReceivedEvent action1 = createActionEvent(null, null);
    // Followed by action with known values
    ActionResultReceivedEvent action2 = createActionEvent(Duration.ofSeconds(50), Duration.ofSeconds(20));
    buildSummaryStatsModule.actionResultReceived(action1);
    buildSummaryStatsModule.actionResultReceived(action2);
    buildSummaryStatsModule.buildComplete(createBuildEvent());
    verify(reporterMock).handle(Event.info("CPU time ???s (user ???s, system ???s, bazel jvm ???s)"));
  }

  @Test
  public void knownAndUnknownCpuTimesForAnActionIsReportedAndSumBecomeUnknown() throws Exception {
    ActionResultReceivedEvent action1 = createActionEvent(Duration.ofSeconds(50), null);
    buildSummaryStatsModule.actionResultReceived(action1);
    buildSummaryStatsModule.buildComplete(createBuildEvent());
    verify(reporterMock).handle(Event.info(String.format("CPU time ???s (user %.2fs, system ???s, bazel jvm ???s)", 50.00)));
  }

  @Test
  public void reusedBuildSummaryStatsModuleIsClearedBetweenBuilds() throws Exception {
    ActionResultReceivedEvent action1 = createActionEvent(Duration.ofSeconds(50),
                                                          Duration.ofSeconds(20));
    buildSummaryStatsModule.actionResultReceived(action1);
    buildSummaryStatsModule.setCpuTimeForBazelJvm(Duration.ofMillis(10000));
    buildSummaryStatsModule.buildComplete(createBuildEvent());
    verify(reporterMock).handle(Event.info(String.format("CPU time %.2fs (user %.2fs, system %.2fs, bazel jvm %.2fs)",80.00, 50.00, 20.00, 10.00)));
    // One more build, and verify that previous values are not preserved.
    buildSummaryStatsModule.buildComplete(createBuildEvent());
    verify(reporterMock).handle(Event.info(String.format("CPU time ???s (user %.2fs, system %.2fs, bazel jvm ???s)",0.00, 0.00)));
  }
}
