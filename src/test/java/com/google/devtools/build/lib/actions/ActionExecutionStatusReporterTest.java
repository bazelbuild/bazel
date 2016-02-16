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
import static org.junit.Assert.assertEquals;
import static org.mockito.Mockito.when;

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.util.EventCollectionApparatus;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Preconditions;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Test for the {@link ActionExecutionStatusReporter} class.
 */
@RunWith(JUnit4.class)
public class ActionExecutionStatusReporterTest {
  private static final class MockClock implements Clock {
    private long millis = 0;

    public void advance() {
      advanceBy(1000);
    }

    public void advanceBy(long millis) {
      Preconditions.checkArgument(millis > 0);
      this.millis += millis;
    }

    @Override
    public long currentTimeMillis() {
      return millis;
    }

    @Override
    public long nanoTime() {
      // There's no reason to use a nanosecond-precision for a mock clock.
      return millis * 1000000L;
    }
  }

  private EventCollectionApparatus events;
  private ActionExecutionStatusReporter statusReporter;
  private EventBus eventBus;
  private MockClock clock = new MockClock();

  private Action mockAction(String progressMessage) {
    Action action = Mockito.mock(Action.class);
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
    assertThat(Splitter.on('\n').omitEmptyStrings().trimResults().split(
        Iterables.getOnlyElement(events.collector()).getMessage().replaceAll(" +", " ")))
        .containsExactlyElementsIn(Arrays.asList(lines)).inOrder();
  }

  private void verifyWarningOutput(String... lines) throws Exception {
    events.setFailFast(false);
    events.clear();
    statusReporter.warnAboutCurrentlyExecutingActions();
    assertThat(Splitter.on('\n').omitEmptyStrings().trimResults().split(
        Iterables.getOnlyElement(events.collector()).getMessage().replaceAll(" +", " ")))
        .containsExactlyElementsIn(Arrays.asList(lines)).inOrder();
  }

  @Test
  public void testCategories() throws Exception {
    verifyNoOutput();
    verifyWarningOutput("There are no active jobs - stopping the build");
    setPreparing(mockAction("action1"));
    clock.advance();
    verifyWarningOutput("Still waiting for unfinished jobs");
    setScheduling(mockAction("action2"));
    clock.advance();
    setRunning(mockAction("action3"), "remote");
    clock.advance();
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
    clock.advanceBy(1200);
    verifyOutput("Still waiting for 1 job to complete:", "Preparing:", "action1, 1 s");
    clock.advanceBy(5000);

    setScheduling(action);
    clock.advanceBy(1200);
    // Only started *scheduling* 1200 ms ago, not 6200 ms ago.
    verifyOutput("Still waiting for 1 job to complete:", "Scheduling:", "action1, 1 s");
    setRunning(action, "remote");
    clock.advanceBy(3000);
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
    clock.advance();
    verifyOutput("Still waiting for 1 job to complete:", "Preparing:", "action1, 1 s");
    setScheduling(action);
    clock.advance();
    verifyOutput("Still waiting for 1 job to complete:", "Scheduling:", "action1, 1 s");
    setRunning(action, "remote");
    clock.advance();
    verifyOutput("Still waiting for 1 job to complete:", "Running (remote):", "action1, 1 s");
    clock.advance();

    eventBus.post(ActionStatusMessage.analysisStrategy(action));
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
      setScheduling(a);
      clock.advance();
    }

    verifyOutput("Still waiting for 6 jobs to complete:",
        "Scheduling:",
        "remote1, 6 s", "remote2, 5 s", "remote3, 4 s",
        "local1, 3 s", "local2, 2 s", "local3, 1 s");

    for (Action a : actions) {
      setRunning(a, a.getProgressMessage().startsWith("remote") ? "remote" : "something else");
      clock.advanceBy(2000);
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
      setScheduling(a);
      clock.advance();
    }
    verifyOutput("Still waiting for 100 jobs to complete:", "Scheduling:",
        "a1, 100 s", "a2, 99 s", "a3, 98 s", "a4, 97 s", "a5, 96 s",
        "a6, 95 s", "a7, 94 s", "a8, 93 s", "a9, 92 s", "... 91 more jobs");

    for (int i = 0; i < 5; i++) {
      setRunning(actions.get(i), "something else");
      clock.advance();
    }
    verifyOutput("Still waiting for 100 jobs to complete:",
        "Running (something else):", "a1, 5 s", "a2, 4 s", "a3, 3 s", "a4, 2 s", "a5, 1 s",
        "Scheduling:", "a6, 100 s", "a7, 99 s", "a8, 98 s", "a9, 97 s", "a10, 96 s",
        "a11, 95 s", "a12, 94 s", "a13, 93 s", "a14, 92 s", "... 86 more jobs");
  }

  @Test
  public void testOrdering() throws Exception {
    verifyNoOutput();
    setScheduling(mockAction("a1"));
    clock.advance();
    setPreparing(mockAction("b1"));
    clock.advance();
    setPreparing(mockAction("b2"));
    clock.advance();
    setScheduling(mockAction("a2"));
    clock.advance();
    verifyOutput("Still waiting for 4 jobs to complete:",
        "Preparing:", "b1, 3 s", "b2, 2 s",
        "Scheduling:", "a1, 4 s", "a2, 1 s");
  }

  @Test
  public void testNoProgressMessage() throws Exception {
    verifyNoOutput();
    setScheduling(mockAction(null));
    verifyOutput("Still waiting for 1 job to complete:", "Scheduling:", "default message, 0 s");
  }

  @Test
  public void testWaitTimeCalculation() throws Exception {
    // --progress_report_interval=0
    assertEquals(10, ActionExecutionStatusReporter.getWaitTime(0, 0));
    assertEquals(30, ActionExecutionStatusReporter.getWaitTime(0, 10));
    assertEquals(60, ActionExecutionStatusReporter.getWaitTime(0, 30));
    assertEquals(60, ActionExecutionStatusReporter.getWaitTime(0, 60));

    // --progress_report_interval=42
    assertEquals(42, ActionExecutionStatusReporter.getWaitTime(42, 0));
    assertEquals(42, ActionExecutionStatusReporter.getWaitTime(42, 42));

    // --progress_report_interval=30 (looks like one of the default timeout stages)
    assertEquals(30, ActionExecutionStatusReporter.getWaitTime(30, 0));
    assertEquals(30, ActionExecutionStatusReporter.getWaitTime(30, 30));
  }

  private void setScheduling(ActionMetadata action) {
    eventBus.post(ActionStatusMessage.schedulingStrategy(action));
  }

  private void setPreparing(ActionMetadata action) {
    eventBus.post(ActionStatusMessage.preparingStrategy(action));
  }

  private void setRunning(ActionMetadata action, String strategy) {
    eventBus.post(ActionStatusMessage.runningStrategy(action, strategy));
  }
}
