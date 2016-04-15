// Copyright 2016 The Bazel Authors. All rights reserved.
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

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.LoggingTerminalWriter;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Tests {@link ExperimentalStateTracker}.
 */
@RunWith(JUnit4.class)
public class ExperimentalStateTrackerTest extends FoundationTestCase {

  private Action mockAction(String progressMessage, String primaryOutput) {
    Path path = outputBase.getRelative(new PathFragment(primaryOutput));
    Artifact artifact = new Artifact(path, Root.asSourceRoot(path));

    Action action = Mockito.mock(Action.class);
    when(action.getProgressMessage()).thenReturn(progressMessage);
    when(action.getPrimaryOutput()).thenReturn(artifact);
    return action;
  }

  @Test
  public void testActionVisible() throws IOException {
    // If there is only one action running, it should be visible
    // somewhere in the progress bar, and also the short version thereof.

    String message = "Building foo";
    ManualClock clock = new ManualClock();
    clock.advanceMillis(120000);

    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    stateTracker.actionStarted(new ActionStartedEvent(mockAction(message, "bar/foo"), 123456789));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();
    assertTrue(
        "Action message '" + message + "' should be present in output: " + output,
        output.contains(message));

    terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter, /* shortVersion=*/ true);
    output = terminalWriter.getTranscript();
    assertTrue(
        "Action message '" + message + "' should be present in short output: " + output,
        output.contains(message));

  }

  @Test
  public void testCompletedActionNotShown() throws IOException {
    // Completed actions should not be reported in the progress bar, nor in the
    // short progress bar.

    String messageFast = "Running quick action";
    String messageSlow = "Running slow action";

    ManualClock clock = new ManualClock();
    clock.advanceMillis(120000);
    Action fastAction = mockAction(messageFast, "foo/fast");
    Action slowAction = mockAction(messageSlow, "bar/slow");
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    stateTracker.actionStarted(new ActionStartedEvent(fastAction, 123456789));
    stateTracker.actionStarted(new ActionStartedEvent(slowAction, 123456999));
    stateTracker.actionCompletion(new ActionCompletionEvent(20, fastAction));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();
    assertFalse(
        "Completed action '" + messageFast + "' should not be present in output: " + output,
        output.contains(messageFast));
    assertTrue(
        "Only running action '" + messageSlow + "' should be present in output: " + output,
        output.contains(messageSlow));

    terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter, /* shortVersion=*/ true);
    output = terminalWriter.getTranscript();
    assertFalse(
        "Completed action '" + messageFast + "' should not be present in short output: " + output,
        output.contains(messageFast));
    assertTrue(
        "Only running action '" + messageSlow + "' should be present in short output: " + output,
        output.contains(messageSlow));
  }

  @Test
  public void testOldestActionVisible() throws IOException {
    // The earliest-started action is always visible somehow in the progress bar
    // and its short version.

    String messageOld = "Running the first-started action";

    ManualClock clock = new ManualClock();
    clock.advanceMillis(120000);
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    stateTracker.actionStarted(
        new ActionStartedEvent(mockAction(messageOld, "bar/foo"), 123456789));
    for (int i = 0; i < 30; i++) {
      stateTracker.actionStarted(
          new ActionStartedEvent(
              mockAction("Other action " + i, "some/other/actions/number" + i), 123456790 + i));
    }

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();
    assertTrue(
        "Longest running action '" + messageOld + "' should be visible in output: " + output,
        output.contains(messageOld));

    terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter, /* shortVersion=*/ true);
    output = terminalWriter.getTranscript();
    assertTrue(
        "Longest running action '" + messageOld + "' should be visible in short output: " + output,
        output.contains(messageOld));
  }

  @Test
  public void testTimesShown() throws IOException {
    // For sufficiently long running actions, the time that has passed since their start is shown.
    // In the short version of the progress bar, this should be true at least for the oldest action.

    ManualClock clock = new ManualClock();
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(123));
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(2));

    stateTracker.actionStarted(
        new ActionStartedEvent(mockAction("First action", "foo"), clock.nanoTime()));
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(7));
    stateTracker.actionStarted(
        new ActionStartedEvent(mockAction("Second action", "bar"), clock.nanoTime()));
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(20));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();
    assertTrue(
        "Runtime of first action should be visible in output: " + output, output.contains("27s"));
    assertTrue(
        "Runtime of second action should be visible in output: " + output, output.contains("20s"));

    terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter, /* shortVersion=*/ true);
    output = terminalWriter.getTranscript();
    assertTrue(
        "Runtime of first action should be visible in short output: " + output,
        output.contains("27s"));
  }

  @Test
  public void initialProgressBarTimeIndependent() {
    ManualClock clock = new ManualClock();
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(123));
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);

    assertFalse(
        "Initial progress status should be time independent",
        stateTracker.progressBarTimeDependent());
  }

  @Test
  public void runningActionTimeIndependent() {
    ManualClock clock = new ManualClock();
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(123));
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.actionStarted(
        new ActionStartedEvent(mockAction("Some action", "foo"), clock.nanoTime()));

    assertTrue(
        "Progress bar showing a running action should be time dependent",
        stateTracker.progressBarTimeDependent());
  }

  @Test
  public void testCountVisible() throws Exception {
    // The test count should be visible in the status bar, as well as the short status bar
    ManualClock clock = new ManualClock();
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    TestFilteringCompleteEvent filteringComplete = Mockito.mock(TestFilteringCompleteEvent.class);
    Label labelA = Label.parseAbsolute("//foo/bar:baz");
    ConfiguredTarget targetA = Mockito.mock(ConfiguredTarget.class);
    when(targetA.getLabel()).thenReturn(labelA);
    ConfiguredTarget targetB = Mockito.mock(ConfiguredTarget.class);
    when(filteringComplete.getTestTargets()).thenReturn(ImmutableSet.of(targetA, targetB));
    TestSummary testSummary = Mockito.mock(TestSummary.class);
    when(testSummary.getTarget()).thenReturn(targetA);

    stateTracker.testFilteringComplete(filteringComplete);
    stateTracker.testSummary(testSummary);

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();
    assertTrue(
        "Test count should be visible in output: " + output, output.contains(" 1 / 2 tests"));

    terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter, /* shortVersion=*/ true);
    output = terminalWriter.getTranscript();
    assertTrue(
        "Test count should be visible in short output: " + output, output.contains(" 1 / 2 tests"));
  }

  @Test
  public void testPassedVisible() throws Exception {
    // The last test that passed should still be visible in the long status bar.
    ManualClock clock = new ManualClock();
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    TestFilteringCompleteEvent filteringComplete = Mockito.mock(TestFilteringCompleteEvent.class);
    Label labelA = Label.parseAbsolute("//foo/bar:baz");
    ConfiguredTarget targetA = Mockito.mock(ConfiguredTarget.class);
    when(targetA.getLabel()).thenReturn(labelA);
    ConfiguredTarget targetB = Mockito.mock(ConfiguredTarget.class);
    when(filteringComplete.getTestTargets()).thenReturn(ImmutableSet.of(targetA, targetB));
    TestSummary testSummary = Mockito.mock(TestSummary.class);
    when(testSummary.getStatus()).thenReturn(BlazeTestStatus.PASSED);
    when(testSummary.getTarget()).thenReturn(targetA);

    stateTracker.testFilteringComplete(filteringComplete);
    stateTracker.testSummary(testSummary);

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Label " + labelA.toString() + " should be present in progress bar: " + output,
        output.contains(labelA.toString()));
  }
}
