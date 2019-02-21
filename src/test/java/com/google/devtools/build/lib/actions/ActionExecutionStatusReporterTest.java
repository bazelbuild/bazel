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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;
import static org.mockito.Mockito.when;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.util.ActionsTestUtil;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.testutil.ManualClock;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Test for the {@link ActionExecutionStatusReporter} class. */
@RunWith(JUnit4.class)
public class ActionExecutionStatusReporterTest {

  private EventCollectionApparatus events;
  private ActionExecutionStatusReporter statusReporter;
  private EventBus eventBus;
  private ManualClock clock = new ManualClock();

  private Action mockAction(String progressMessage) {
    Action action = Mockito.mock(Action.class);
    when(action.getOwner()).thenReturn(ActionsTestUtil.NULL_ACTION_OWNER);
    when(action.getProgressMessage()).thenReturn(progressMessage);
    if (progressMessage == null) {
      when(action.prettyPrint()).thenReturn("default message");
    }
    return action;
  }

  @Before
  public final void initializeEventBus() throws Exception  {
    events = new EventCollectionApparatus(EventKind.ALL_EVENTS);
    statusReporter = ActionExecutionStatusReporter.create(events.reporter(), clock);
    eventBus = new EventBus();
    eventBus.register(statusReporter);
  }

  private void verifyNoOutput() {
    events.clear();
    statusReporter.showCurrentlyExecutingActions("");
    assertThat(events.collector()).isEmpty();
  }

  private void verifyOutput(String... lines) throws Exception {
    events.clear();
    statusReporter.showCurrentlyExecutingActions("");
    assertThat(
            Splitter.on('\n')
                .omitEmptyStrings()
                .trimResults()
                .split(
                    Iterables.getOnlyElement(events.collector())
                        .getMessage()
                        .replaceAll(" +", " ")))
        .containsExactlyElementsIn(Arrays.asList(lines))
        .inOrder();
  }

  private void verifyWarningOutput(String... lines) throws Exception {
    events.setFailFast(false);
    events.clear();
    statusReporter.warnAboutCurrentlyExecutingActions();
    assertThat(
            Splitter.on('\n')
                .omitEmptyStrings()
                .trimResults()
                .split(
                    Iterables.getOnlyElement(events.collector())
                        .getMessage()
                        .replaceAll(" +", " ")))
        .containsExactlyElementsIn(Arrays.asList(lines))
        .inOrder();
  }

  @Test
  public void testCategories() throws Exception {
    verifyNoOutput();
    verifyWarningOutput("There are no active jobs - stopping the build");
    setPreparing(mockAction("action1"));
    clock.advanceMillis(1000);
    verifyWarningOutput("Still waiting for unfinished jobs");
    setScheduling(mockAction("action2"), "remote");
    clock.advanceMillis(1000);
    setRunning(mockAction("action3"), "remote");
    clock.advanceMillis(1000);
    setRunning(mockAction("action4"), "something else");
    verifyOutput("Still waiting for 4 jobs to complete:",
        "Preparing:", "action1, 3 s",
        "Running (remote):", "action3, 1 s",
        "Running (something else):", "action4, 0 s",
        "Scheduling:", "action2, 2 s");
    verifyWarningOutput("Still waiting for 3 jobs to complete:",
        "Running (remote):", "action3, 1 s",
        "Running (something else):", "action4, 0 s",
        "Scheduling:", "action2, 2 s",
        "Build will be stopped after these tasks terminate");
  }

  @Test
  public void testSingleAction() throws Exception {
    Action action = mockAction("action1");
    verifyNoOutput();
    setPreparing(action);
    clock.advanceMillis(1200);
    verifyOutput("Still waiting for 1 job to complete:", "Preparing:", "action1, 1 s");
    clock.advanceMillis(5000);

    setScheduling(action, "remote");
    clock.advanceMillis(1200);
    // Only started *scheduling* 1200 ms ago, not 6200 ms ago.
    verifyOutput("Still waiting for 1 job to complete:", "Scheduling:", "action1, 1 s");
    setRunning(action, "remote");
    clock.advanceMillis(3000);
    // Only started *running* 3000 ms ago, not 4200 ms ago.
    verifyOutput("Still waiting for 1 job to complete:", "Running (remote):", "action1, 3 s");
    statusReporter.remove(action);
    verifyNoOutput();
  }

  @Test
  public void testDynamicUpdate() throws Exception {
    Action action = mockAction("action1");
    verifyNoOutput();
    setPreparing(action);
    clock.advanceMillis(1000);
    verifyOutput("Still waiting for 1 job to complete:", "Preparing:", "action1, 1 s");
    setScheduling(action, "remote");
    clock.advanceMillis(1000);
    verifyOutput("Still waiting for 1 job to complete:", "Scheduling:", "action1, 1 s");
    setRunning(action, "remote");
    clock.advanceMillis(1000);
    verifyOutput("Still waiting for 1 job to complete:", "Running (remote):", "action1, 1 s");
    clock.advanceMillis(1000);

    eventBus.post(new AnalyzingActionEvent(action));
    // Locality strategy was changed, so timer was reset to 0 s.
    verifyOutput("Still waiting for 1 job to complete:", "Analyzing:", "action1, 0 s");
    statusReporter.remove(action);
    verifyNoOutput();
  }

  @Test
  public void testGroups() throws Exception {
    verifyNoOutput();
    List<Action> actions = ImmutableList.of(
        mockAction("remote1"), mockAction("remote2"), mockAction("remote3"),
        mockAction("local1"), mockAction("local2"), mockAction("local3"));

    for (Action a : actions) {
      setScheduling(a, "remote");
      clock.advanceMillis(1000);
    }

    verifyOutput("Still waiting for 6 jobs to complete:",
        "Scheduling:",
        "remote1, 6 s", "remote2, 5 s", "remote3, 4 s",
        "local1, 3 s", "local2, 2 s", "local3, 1 s");

    for (Action a : actions) {
      setRunning(a, a.getProgressMessage().startsWith("remote") ? "remote" : "something else");
      clock.advanceMillis(2000);
    }

    // Timers got reset because now they are no longer scheduling but running.
    verifyOutput("Still waiting for 6 jobs to complete:",
        "Running (remote):", "remote1, 12 s", "remote2, 10 s", "remote3, 8 s",
        "Running (something else):", "local1, 6 s", "local2, 4 s", "local3, 2 s");

    statusReporter.remove(actions.get(0));
    verifyOutput("Still waiting for 5 jobs to complete:",
        "Running (remote):", "remote2, 10 s", "remote3, 8 s",
        "Running (something else):", "local1, 6 s", "local2, 4 s", "local3, 2 s");
  }

  @Test
  public void testTruncation() throws Exception {
    verifyNoOutput();
    List<Action> actions = new ArrayList<>();
    for (int i = 1; i <= 100; i++) {
      Action a = mockAction("a" + i);
      actions.add(a);
      setScheduling(a, "remote");
      clock.advanceMillis(1000);
    }
    verifyOutput("Still waiting for 100 jobs to complete:", "Scheduling:",
        "a1, 100 s", "a2, 99 s", "a3, 98 s", "a4, 97 s", "a5, 96 s",
        "a6, 95 s", "a7, 94 s", "a8, 93 s", "a9, 92 s", "... 91 more jobs");

    for (int i = 0; i < 5; i++) {
      setRunning(actions.get(i), "something else");
      clock.advanceMillis(1000);
    }
    verifyOutput("Still waiting for 100 jobs to complete:",
        "Running (something else):", "a1, 5 s", "a2, 4 s", "a3, 3 s", "a4, 2 s", "a5, 1 s",
        "Scheduling:", "a6, 100 s", "a7, 99 s", "a8, 98 s", "a9, 97 s", "a10, 96 s",
        "a11, 95 s", "a12, 94 s", "a13, 93 s", "a14, 92 s", "... 86 more jobs");
  }

  @Test
  public void testOrdering() throws Exception {
    verifyNoOutput();
    setScheduling(mockAction("a1"), "remote");
    clock.advanceMillis(1000);
    setPreparing(mockAction("b1"));
    clock.advanceMillis(1000);
    setPreparing(mockAction("b2"));
    clock.advanceMillis(1000);
    setScheduling(mockAction("a2"), "remote");
    clock.advanceMillis(1000);
    verifyOutput("Still waiting for 4 jobs to complete:",
        "Preparing:", "b1, 3 s", "b2, 2 s",
        "Scheduling:", "a1, 4 s", "a2, 1 s");
  }

  @Test
  public void testNoProgressMessage() throws Exception {
    verifyNoOutput();
    setScheduling(mockAction(null), "remote");
    verifyOutput("Still waiting for 1 job to complete:", "Scheduling:", "default message, 0 s");
  }

  @Test
  public void testWaitTimeCalculation() throws Exception {
    // --progress_report_interval=0
    assertThat(ActionExecutionStatusReporter.getWaitTime(0, 0)).isEqualTo(10);
    assertThat(ActionExecutionStatusReporter.getWaitTime(0, 10)).isEqualTo(30);
    assertThat(ActionExecutionStatusReporter.getWaitTime(0, 30)).isEqualTo(60);
    assertThat(ActionExecutionStatusReporter.getWaitTime(0, 60)).isEqualTo(60);

    // --progress_report_interval=42
    assertThat(ActionExecutionStatusReporter.getWaitTime(42, 0)).isEqualTo(42);
    assertThat(ActionExecutionStatusReporter.getWaitTime(42, 42)).isEqualTo(42);

    // --progress_report_interval=30 (looks like one of the default timeout stages)
    assertThat(ActionExecutionStatusReporter.getWaitTime(30, 0)).isEqualTo(30);
    assertThat(ActionExecutionStatusReporter.getWaitTime(30, 30)).isEqualTo(30);
  }

  private void setScheduling(ActionExecutionMetadata action, String strategy) {
    eventBus.post(new SchedulingActionEvent(action, strategy));
  }

  private void setPreparing(Action action) {
    eventBus.post(new ActionStartedEvent(action, 0));
  }

  private void setRunning(ActionExecutionMetadata action, String strategy) {
    eventBus.post(new RunningActionEvent(action, strategy));
  }
}
