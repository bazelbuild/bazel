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

import static org.hamcrest.CoreMatchers.containsString;
import static org.hamcrest.CoreMatchers.not;
import static org.hamcrest.MatcherAssert.assertThat;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.when;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Root;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.bazel.repository.downloader.DownloadProgressEvent;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import com.google.devtools.build.lib.skyframe.LoadingPhaseStartedEvent;
import com.google.devtools.build.lib.skyframe.PackageProgressReceiver;
import com.google.devtools.build.lib.testutil.FoundationTestCase;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.LoggingTerminalWriter;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import java.io.IOException;
import java.net.URL;
import java.util.concurrent.TimeUnit;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/**
 * Tests {@link ExperimentalStateTracker}.
 */
@RunWith(JUnit4.class)
public class ExperimentalStateTrackerTest extends FoundationTestCase {

  private Action mockAction(String progressMessage, String primaryOutput) {
    Path path = outputBase.getRelative(PathFragment.create(primaryOutput));
    Artifact artifact = new Artifact(path, Root.asSourceRoot(path));

    Action action = Mockito.mock(Action.class);
    when(action.getProgressMessage()).thenReturn(progressMessage);
    when(action.getPrimaryOutput()).thenReturn(artifact);
    return action;
  }

  private int longestLine(String output) {
    int maxLength = 0;
    for (String line : output.split("\n")) {
      maxLength = Math.max(maxLength, line.length());
    }
    return maxLength;
  }

  @Test
  public void testLoadingActivity() throws IOException {
    // During loading phase, state and activity, as reported by the PackageProgressReceiver,
    // should be visible in the progress bar.
    String state = "42 packages loaded";
    String activity = "currently loading //src/foo/bar and 17 more";
    PackageProgressReceiver progress = Mockito.mock(PackageProgressReceiver.class);
    when(progress.progressState()).thenReturn(new Pair<String, String>(state, activity));

    ManualClock clock = new ManualClock();
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);

    stateTracker.loadingStarted(new LoadingPhaseStartedEvent(progress));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Output should indicate that we are in the loading phase, but was:\n" + output,
        output.contains("Loading"));
    assertTrue(
        "Output should contain loading state '" + state + "', but was:\n" + output,
        output.contains(state));
    assertTrue(
        "Output should contain loading state '" + activity + "', but was:\n" + output,
        output.contains(activity));
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
    stateTracker.actionCompletion(
        new ActionCompletionEvent(20, fastAction, Mockito.mock(ActionLookupData.class)));

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
  public void testSampleSize() throws IOException {
    // Verify that the number of actions shown in the progress bar can be set as sample size.
    ManualClock clock = new ManualClock();
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(123));
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(2));

    // Start 10 actions (numbered 0 to 9).
    for (int i = 0; i < 10; i++) {
      clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
      Action action = mockAction("Performing action A" + i + ".", "action_A" + i + ".out");
      stateTracker.actionStarted(new ActionStartedEvent(action, clock.nanoTime()));
    }

    // For various sample sizes verify the progress bar
    for (int i = 1; i < 11; i++) {
      stateTracker.setSampleSize(i);
      LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
      stateTracker.writeProgressBar(terminalWriter);
      String output = terminalWriter.getTranscript();
      assertTrue(
          "Action " + (i - 1) + " should still be shown in the output: '" + output,
          output.contains("A" + (i - 1) + "."));
      assertFalse(
          "Action " + i + " should not be shown in the output: " + output,
          output.contains("A" + i + "."));
      if (i < 10) {
        assertTrue("Ellipsis symbol should be shown in output: " + output, output.contains("..."));
      } else {
        assertFalse(
            "Ellipsis symbol should not be shown in output: " + output, output.contains("..."));
      }
    }
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
    // The last test should still be visible in the long status bar, and colored as ok if it passed.
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

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    String expected = LoggingTerminalWriter.OK + labelA;
    assertTrue(
        "Sequence '" + expected + "' should be present in colored progress bar: " + output,
        output.contains(expected));
  }

  @Test
  public void testFailedVisible() throws Exception {
    // The last test should still be visible in the long status bar, and colored as fail if it
    // did not pass.
    ManualClock clock = new ManualClock();
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    TestFilteringCompleteEvent filteringComplete = Mockito.mock(TestFilteringCompleteEvent.class);
    Label labelA = Label.parseAbsolute("//foo/bar:baz");
    ConfiguredTarget targetA = Mockito.mock(ConfiguredTarget.class);
    when(targetA.getLabel()).thenReturn(labelA);
    ConfiguredTarget targetB = Mockito.mock(ConfiguredTarget.class);
    when(filteringComplete.getTestTargets()).thenReturn(ImmutableSet.of(targetA, targetB));
    TestSummary testSummary = Mockito.mock(TestSummary.class);
    when(testSummary.getStatus()).thenReturn(BlazeTestStatus.FAILED);
    when(testSummary.getTarget()).thenReturn(targetA);

    stateTracker.testFilteringComplete(filteringComplete);
    stateTracker.testSummary(testSummary);

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter();
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    String expected = LoggingTerminalWriter.FAIL + labelA;
    assertTrue(
        "Sequence '" + expected + "' should be present in colored progress bar: " + output,
        output.contains(expected));
  }

  @Test
  public void testSensibleShortening() throws Exception {
    // Verify that in the typical case, we shorten the progress message by shortening
    // the path implicit in it, that can also be extracted from the label. In particular,
    // the parts
    ManualClock clock = new ManualClock();
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock, 70);
    Action action = mockAction(
        "Building some/very/very/long/path/for/some/library/directory/foo.jar (42 source files)",
        "/home/user/bazel/out/abcdef/some/very/very/long/path/for/some/library/directory/foo.jar");
    Label label =
        Label.parseAbsolute("//some/very/very/long/path/for/some/library/directory:libfoo");
    ActionOwner owner = ActionOwner.create(
        label, ImmutableList.<AspectDescriptor>of(), null, null, null, "fedcba", null);
    when(action.getOwner()).thenReturn(owner);

    clock.advanceMillis(TimeUnit.SECONDS.toMillis(3));
    stateTracker.actionStarted(new ActionStartedEvent(action, clock.nanoTime()));
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(5));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Progress bar should contain 'Building ', but was:\n" + output,
        output.contains("Building "));
    assertTrue(
        "Progress bar should contain 'foo.jar (42 source files)', but was:\n" + output,
        output.contains("foo.jar (42 source files)"));
  }

  @Test
  public void testActionStrategyVisible() throws Exception {
    // verify that, if a strategy was reported for a shown action, it is visible
    // in the progress bar.
    String strategy = "verySpecialStrategy";
    String primaryOutput = "some/path/to/a/file";

    ManualClock clock = new ManualClock();
    Path path = outputBase.getRelative(PathFragment.create(primaryOutput));
    Artifact artifact = new Artifact(path, Root.asSourceRoot(path));
    ActionExecutionMetadata actionMetadata = Mockito.mock(ActionExecutionMetadata.class);
    when(actionMetadata.getOwner()).thenReturn(Mockito.mock(ActionOwner.class));
    when(actionMetadata.getPrimaryOutput()).thenReturn(artifact);

    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock);
    stateTracker.actionStarted(
        new ActionStartedEvent(mockAction("Some random action", primaryOutput), clock.nanoTime()));
    stateTracker.actionStatusMessage(ActionStatusMessage.runningStrategy(actionMetadata, strategy));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Output should mention strategy '" + strategy + "', but was: " + output,
        output.contains(strategy));
  }

  private void doTestOutputLength(boolean withTest, int actions) throws Exception {
    // If we target 70 characters, then there should be enough space to both,
    // keep the line limit, and show the local part of the running actions and
    // the passed test.
    ManualClock clock = new ManualClock();
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock, 70);

    Action foobuildAction = mockAction(
        "Building //src/some/very/long/path/long/long/long/long/long/long/long/foo/foobuild.jar",
        "//src/some/very/long/path/long/long/long/long/long/long/long/foo:foobuild");
    Action bazbuildAction = mockAction(
        "Building //src/some/very/long/path/long/long/long/long/long/long/long/baz/bazbuild.jar",
        "//src/some/very/long/path/long/long/long/long/long/long/long/baz:bazbuild");

    Label bartestLabel =
        Label.parseAbsolute(
            "//src/another/very/long/long/path/long/long/long/long/long/long/long/long/bars:bartest");
    ConfiguredTarget bartestTarget = Mockito.mock(ConfiguredTarget.class);
    when(bartestTarget.getLabel()).thenReturn(bartestLabel);

    TestFilteringCompleteEvent filteringComplete = Mockito.mock(TestFilteringCompleteEvent.class);
    when(filteringComplete.getTestTargets()).thenReturn(ImmutableSet.of(bartestTarget));

    TestSummary testSummary = Mockito.mock(TestSummary.class);
    when(testSummary.getStatus()).thenReturn(BlazeTestStatus.PASSED);
    when(testSummary.getTarget()).thenReturn(bartestTarget);

    if (actions >= 1) {
      stateTracker.actionStarted(new ActionStartedEvent(foobuildAction, 123456789));
    }
    if (actions >= 2) {
      stateTracker.actionStarted(new ActionStartedEvent(bazbuildAction, 123456900));
    }
    if (withTest) {
      stateTracker.testFilteringComplete(filteringComplete);
      stateTracker.testSummary(testSummary);
    }

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Only lines with at most 70 chars should be present in the output:\n" + output,
        longestLine(output) <= 70);
    if (actions >= 1) {
      assertTrue(
          "Running action 'foobuild' should be mentioned in output:\n" + output,
          output.contains("foobuild"));
    }
    if (actions >= 2) {
      assertTrue(
          "Running action 'bazbuild' should be mentioned in output:\n" + output,
          output.contains("bazbuild"));
    }
    if (withTest) {
      assertTrue(
          "Passed test ':bartest' should be mentioned in output:\n" + output,
          output.contains(":bartest"));
    }
  }

  @Test
  public void testOutputLength() throws Exception {
    for (int i = 0; i < 3; i++) {
      doTestOutputLength(true, i);
      doTestOutputLength(false, i);
    }
  }

  @Test
  public void testAggregation() throws Exception {
    // Assert that actions for the same test are aggregated so that an action afterwards
    // is still shown.
    ManualClock clock = new ManualClock();
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1234));
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock, 80);

    Label labelFooTest = Label.parseAbsolute("//foo/bar:footest");
    ConfiguredTarget targetFooTest = Mockito.mock(ConfiguredTarget.class);
    when(targetFooTest.getLabel()).thenReturn(labelFooTest);
    ActionOwner fooOwner =
        ActionOwner.create(labelFooTest,
            ImmutableList.<AspectDescriptor>of(), null, null, null, "abcdef", null);

    Label labelBarTest = Label.parseAbsolute("//baz:bartest");
    ConfiguredTarget targetBarTest = Mockito.mock(ConfiguredTarget.class);
    when(targetBarTest.getLabel()).thenReturn(labelBarTest);
    TestFilteringCompleteEvent filteringComplete = Mockito.mock(TestFilteringCompleteEvent.class);
    when(filteringComplete.getTestTargets())
        .thenReturn(ImmutableSet.of(targetFooTest, targetBarTest));
    ActionOwner barOwner =
        ActionOwner.create(labelBarTest,
            ImmutableList.<AspectDescriptor>of(), null, null, null, "fedcba", null);

    stateTracker.testFilteringComplete(filteringComplete);

    // First produce 10 actions for footest...
    for (int i = 0; i < 10; i++) {
      clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
      Action action = mockAction("Testing foo, shard " + i, "testlog_foo_" + i);
      when(action.getOwner()).thenReturn(fooOwner);
      stateTracker.actionStarted(new ActionStartedEvent(action, clock.nanoTime()));
    }
    // ...then produce 10 actions for bartest...
    for (int i = 0; i < 10; i++) {
      clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
      Action action = mockAction("Testing bar, shard " + i, "testlog_bar_" + i);
      when(action.getOwner()).thenReturn(barOwner);
      stateTracker.actionStarted(new ActionStartedEvent(action, clock.nanoTime()));
    }
    // ...and finally a completely unrelated action
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.actionStarted(
        new ActionStartedEvent(mockAction("Other action", "other/action"), clock.nanoTime()));
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Progress bar should contain ':footest', but was:\n" + output, output.contains(":footest"));
    assertTrue(
        "Progress bar should contain ':bartest', but was:\n" + output, output.contains(":bartest"));
    assertTrue(
        "Progress bar should contain 'Other action', but was:\n" + output,
        output.contains("Other action"));
  }


  @Test
  public void testSuffix() throws Exception {
    assertEquals("bar", ExperimentalStateTracker.suffix("foobar", 3));
    assertEquals("", ExperimentalStateTracker.suffix("foo", -2));
    assertEquals("foobar", ExperimentalStateTracker.suffix("foobar", 200));
  }

  @Test
  public void testDownloadShown() throws Exception {
    // Verify that, whenever a single download is running in loading face, it is shown in the status
    // bar.
    ManualClock clock = new ManualClock();
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1234));
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock, 80);

    URL url = new URL("http://example.org/first/dep");

    stateTracker.buildStarted(null);
    stateTracker.downloadProgress(new DownloadProgressEvent(url));
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(6));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Progress bar should contain '" + url.toString() + "', but was:\n" + output,
        output.contains(url.toString()));
    assertTrue("Progress bar should contain '6s', but was:\n" + output, output.contains("6s"));

    // Progress on the pending download should be reported appropriately
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.downloadProgress(new DownloadProgressEvent(url, 256));

    terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    output = terminalWriter.getTranscript();

    assertTrue(
        "Progress bar should contain '" + url.toString() + "', but was:\n" + output,
        output.contains(url.toString()));
    assertTrue("Progress bar should contain '7s', but was:\n" + output, output.contains("7s"));
    assertTrue("Progress bar should contain '256', but was:\n" + output, output.contains("256"));

    // After finishing the download, it should no longer be reported.
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.downloadProgress(new DownloadProgressEvent(url, 256, true));

    terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    output = terminalWriter.getTranscript();

    assertFalse(
        "Progress bar should not contain url, but was:\n" + output, output.contains("example.org"));
  }

  @Test
  public void testDownloadOutputLength() throws Exception {
    // Verify that URLs are shortened in a reasonable way, if the terminal is not wide enough
    // Also verify that the length is respected, even if only a download sample is shown.
    ManualClock clock = new ManualClock();
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1234));
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock, 60);
    URL url = new URL("http://example.org/some/really/very/very/long/path/filename.tar.gz");

    stateTracker.buildStarted(null);
    stateTracker.downloadProgress(new DownloadProgressEvent(url));
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(6));
    for (int i = 0; i < 10; i++) {
      stateTracker.downloadProgress(
          new DownloadProgressEvent(
              new URL(
                  "http://otherhost.example/another/also/length/path/to/another/download"
                      + i
                      + ".zip")));
      clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    }

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(/*discardHighlight=*/ true);
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();

    assertTrue(
        "Only lines with at most 60 chars should be present in the output:\n" + output,
        longestLine(output) <= 60);
    assertTrue(
        "Output still should contain the filename, but was:\n" + output,
        output.contains("filename.tar.gz"));
    assertTrue(
        "Output still should contain the host name, but was:\n" + output,
        output.contains("example.org"));
  }

  @Test
  public void testMultipleBuildEventProtocolTransports() throws Exception {
    // Verify that all announced transports are present in the progress bar
    // and that as transports are closed they disappear from the progress bar.
    // Verify that the wait duration is displayed.
    // Verify that after all transports have been closed, the build status is displayed.
    ManualClock clock = new ManualClock();
    BuildEventTransport transport1 = newBepTransport("BuildEventTransport1");
    BuildEventTransport transport2 = newBepTransport("BuildEventTransport2");
    BuildEventTransport transport3 = newBepTransport("BuildEventTransport3");
    BuildResult buildResult = new BuildResult(clock.currentTimeMillis());
    buildResult.setExitCondition(ExitCode.SUCCESS);
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    buildResult.setStopTime(clock.currentTimeMillis());

    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock, 80);
    stateTracker.buildStarted(null);
    stateTracker.buildEventTransportsAnnounced(
        new AnnounceBuildEventTransportsEvent(
            ImmutableList.of(transport1, transport2)));
    stateTracker.buildEventTransportsAnnounced(
        new AnnounceBuildEventTransportsEvent(ImmutableList.of(transport3)));
    stateTracker.buildComplete(new BuildCompleteEvent(buildResult));

    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(true);

    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();
    assertThat(output, containsString("1s"));
    assertThat(output, containsString("BuildEventTransport1"));
    assertThat(output, containsString("BuildEventTransport2"));
    assertThat(output, containsString("BuildEventTransport3"));
    assertThat(output, containsString("success"));
    assertThat(output, containsString("complete"));

    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.buildEventTransportClosed(new BuildEventTransportClosedEvent(transport1));
    terminalWriter = new LoggingTerminalWriter(true);
    stateTracker.writeProgressBar(terminalWriter);
    output = terminalWriter.getTranscript();
    assertThat(output, containsString("2s"));
    assertThat(output, not(containsString("BuildEventTransport1")));
    assertThat(output, containsString("BuildEventTransport2"));
    assertThat(output, containsString("BuildEventTransport3"));
    assertThat(output, containsString("success"));
    assertThat(output, containsString("complete"));

    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.buildEventTransportClosed(new BuildEventTransportClosedEvent(transport3));
    terminalWriter = new LoggingTerminalWriter(true);
    stateTracker.writeProgressBar(terminalWriter);
    output = terminalWriter.getTranscript();
    assertThat(output, containsString("3s"));
    assertThat(output, not(containsString("BuildEventTransport1")));
    assertThat(output, containsString("BuildEventTransport2"));
    assertThat(output, not(containsString("BuildEventTransport3")));
    assertThat(output, containsString("success"));
    assertThat(output, containsString("complete"));

    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.buildEventTransportClosed(new BuildEventTransportClosedEvent(transport2));
    terminalWriter = new LoggingTerminalWriter(true);
    stateTracker.writeProgressBar(terminalWriter);
    output = terminalWriter.getTranscript();
    assertThat(output, not(containsString("3s")));
    assertThat(output, not(containsString("BuildEventTransport1")));
    assertThat(output, not(containsString("BuildEventTransport2")));
    assertThat(output, not(containsString("BuildEventTransport3")));
    assertThat(output, containsString("success"));
    assertThat(output, containsString("complete"));
    assertEquals(1, output.split("\\n").length);
  }

  @Test
  public void testBuildEventTransportsOnNarrowTerminal() throws IOException{
    // Verify that the progress bar contains useful information on a 60-character terminal.
    //   - Too long names should be shortened to reasonably long prefixes of the name.
    ManualClock clock = new ManualClock();
    BuildEventTransport transport1 =
        newBepTransport(Strings.repeat("A", 61));
    BuildEventTransport transport2 = newBepTransport("BuildEventTransport");
    BuildResult buildResult = new BuildResult(clock.currentTimeMillis());
    buildResult.setExitCondition(ExitCode.SUCCESS);
    LoggingTerminalWriter terminalWriter = new LoggingTerminalWriter(true);
    ExperimentalStateTracker stateTracker = new ExperimentalStateTracker(clock, 60);
    stateTracker.buildStarted(null);
    stateTracker.buildEventTransportsAnnounced(
        new AnnounceBuildEventTransportsEvent(ImmutableList.of(transport1, transport2)));
    stateTracker.buildComplete(new BuildCompleteEvent(buildResult));
    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.writeProgressBar(terminalWriter);
    String output = terminalWriter.getTranscript();
    assertTrue(longestLine(output) <= 60);
    assertThat(output, containsString("1s"));
    assertThat(output, containsString(Strings.repeat("A", 30) + "..."));
    assertThat(output, containsString("BuildEventTransport"));
    assertThat(output, containsString("success"));
    assertThat(output, containsString("complete"));

    clock.advanceMillis(TimeUnit.SECONDS.toMillis(1));
    stateTracker.buildEventTransportClosed(new BuildEventTransportClosedEvent(transport2));
    terminalWriter = new LoggingTerminalWriter(true);
    stateTracker.writeProgressBar(terminalWriter);
    output = terminalWriter.getTranscript();
    assertTrue(longestLine(output) <= 60);
    assertThat(output, containsString("2s"));
    assertThat(output, containsString(Strings.repeat("A", 30) + "..."));
    assertThat(output, not(containsString("BuildEventTransport")));
    assertThat(output, containsString("success"));
    assertThat(output, containsString("complete"));
    assertEquals(2, output.split("\\n").length);
  }

  private BuildEventTransport newBepTransport(String name) {
    BuildEventTransport transport = Mockito.mock(BuildEventTransport.class);
    when(transport.name()).thenReturn(name);
    return transport;
  }
}
