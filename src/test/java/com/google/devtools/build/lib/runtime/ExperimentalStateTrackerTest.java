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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

import java.io.IOException;
import java.util.concurrent.TimeUnit;

/**
 * Tests {@link ExperimentalStateTrackerTest}
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

  private class LoggingTerminalWriter implements AnsiTerminalWriter {
    private String written;

    LoggingTerminalWriter() {
      this.written = "";
    }

    @Override
    public LoggingTerminalWriter append(String text) throws IOException {
      written += text;
      return this;
    }

    @Override
    public LoggingTerminalWriter newline() throws IOException {
      return this;
    }

    @Override
    public LoggingTerminalWriter okStatus() throws IOException {
      return this;
    }

    @Override
    public LoggingTerminalWriter failStatus() throws IOException {
      return this;
    }

    @Override
    public LoggingTerminalWriter normal() throws IOException {
      return this;
    }

    String getWritten() {
      return written;
    }
  }

  @Test
  public void testActionVisible() throws IOException {
    // If there is only one action running, it should be visible
    // somewhere in the progress bar.

    String message = "Building foo";
    ManualClock clock = new ManualClock();
    clock.advanceMillis(120000);

    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    stateTracker.actionStarted(new ActionStartedEvent(mockAction(message, "bar/foo"), 123456789));
    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter);

    String output = terminalWriter.getWritten();
    assertTrue(
        "Action message '" + message + "' should be present in output: " + output,
        output.contains(message));
  }

  @Test
  public void testCompletedActionNotShown() throws IOException {
    // Completed actions should not be reported in the progress bar.

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

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter);

    String output = terminalWriter.getWritten();
    assertFalse(
        "Completed action '" + messageFast + "' should not be present in output: " + output,
        output.contains(messageFast));
    assertTrue(
        "Only running action '" + messageSlow + "' should be present in output: " + output,
        output.contains(messageSlow));
  }

  @Test
  public void testOldestActionVisible() throws IOException {
    // The earliest-started action is always visible somehow in the progress bar.

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

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter);

    String output = terminalWriter.getWritten();
    assertTrue(
        "Longest running action '" + messageOld + "' should be visible in output: " + output,
        output.contains(messageOld));
  }

  @Test
  public void testTimesShown() throws IOException {
    // For sufficiently long running actions, the time that has passed since their start is shown.

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

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getWritten();

    assertTrue(
        "Runtime of first action should be visible in output: " + output, output.contains("27s"));
    assertTrue(
        "Runtime of second action should be visible in output: " + output, output.contains("20s"));
  }

}
