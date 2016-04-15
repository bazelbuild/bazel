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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.buildtool.ExecutionProgressReceiver;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionProgressReceiverAvailableEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.skyframe.LoadingPhaseStartedEvent;
import com.google.devtools.build.lib.skyframe.LoadingProgressReceiver;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.PositionAwareAnsiTerminalWriter;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.Map;
import java.util.TreeMap;

/**
 * An experimental state tracker for the new experimental UI.
 */
class ExperimentalStateTracker {

  static final int SAMPLE_SIZE = 3;
  static final long SHOW_TIME_THRESHOLD_SECONDS = 3;
  static final String ELLIPSIS = "...";

  private String status;
  private String additionalMessage;

  private final Clock clock;

  // Desired maximal width of the progress bar, if positive.
  // Non-positive values indicate not to aim for a particular width.
  private final int targetWidth;

  // currently running actions, using the path of the primary
  // output as unique identifier.
  private final Deque<String> runningActions;
  private final Map<String, Action> actions;
  private final Map<String, Long> actionNanoStartTimes;

  private int actionsCompleted;
  private int totalTests;
  private int completedTests;
  private TestSummary mostRecentTest;
  private int failedTests;
  private boolean ok;

  private ExecutionProgressReceiver executionProgressReceiver;
  private LoadingProgressReceiver loadingProgressReceiver;

  ExperimentalStateTracker(Clock clock, int targetWidth) {
    this.runningActions = new ArrayDeque<>();
    this.actions = new TreeMap<>();
    this.actionNanoStartTimes = new TreeMap<>();
    this.ok = true;
    this.clock = clock;
    this.targetWidth = targetWidth;
  }

  ExperimentalStateTracker(Clock clock) {
    this(clock, 0);
  }

  void buildStarted(BuildStartingEvent event) {
    status = "Loading";
    additionalMessage = "";
  }

  void loadingStarted(LoadingPhaseStartedEvent event) {
    status = null;
    loadingProgressReceiver = event.getLoadingProgressReceiver();
  }

  void loadingComplete(LoadingPhaseCompleteEvent event) {
    loadingProgressReceiver = null;
    int count = event.getTargets().size();
    status = "Analysing";
    additionalMessage = "" + count + " targets";
  }

  void analysisComplete(AnalysisPhaseCompleteEvent event) {
    status = null;
  }

  void progressReceiverAvailable(ExecutionProgressReceiverAvailableEvent event) {
    executionProgressReceiver = event.getExecutionProgressReceiver();
  }

  void buildComplete(BuildCompleteEvent event) {
    if (event.getResult().getSuccess()) {
      status = "INFO";
      additionalMessage = "Build completed successfully, " + actionsCompleted + " total actions";
    } else {
      ok = false;
      status = "FAILED";
      additionalMessage = "Build did NOT complete successfully";
    }
  }

  synchronized void actionStarted(ActionStartedEvent event) {
    Action action = event.getAction();
    String name = action.getPrimaryOutput().getPath().getPathString();
    Long nanoStartTime = event.getNanoTimeStart();
    runningActions.addLast(name);
    actions.put(name, action);
    actionNanoStartTimes.put(name, nanoStartTime);
  }

  synchronized void actionCompletion(ActionCompletionEvent event) {
    actionsCompleted++;
    Action action = event.getAction();
    String name = action.getPrimaryOutput().getPath().getPathString();
    runningActions.remove(name);
    actions.remove(name);
    actionNanoStartTimes.remove(name);

    // As callers to the experimental state tracker assume we will fully report the new state once
    // informed of an action completion, we need to make sure the progress receiver is aware of the
    // completion, even though it might be called later on the event bus.
    if (executionProgressReceiver != null) {
      executionProgressReceiver.actionCompleted(action);
    }
  }

  /**
   * From a string, take a suffix of at most the given length.
   */
  private String suffix(String s, int len) {
    int startPos = s.length() - len;
    if (startPos <= 0) {
      return s;
    }
    return s.substring(startPos);
  }

  /**
   * If possible come up with a human-readable description of the label
   * that fits within the given width; a non-positive width indicates not
   * no restriction at all.
   */
  private String shortenedLabelString(Label label, int width) {
    if (width <= 0) {
      return label.toString();
    }
    String name = label.toString();
    if (name.length() <= width) {
      return name;
    }
    name = suffix(name, width - ELLIPSIS.length());
    int slashPos = name.indexOf('/');
    if (slashPos >= 0) {
      return ELLIPSIS + name.substring(slashPos);
    }
    int colonPos = name.indexOf(':');
    if (slashPos >= 0) {
      return ELLIPSIS + name.substring(colonPos);
    }
    // no reasonable place found to shorten; as last resort, just truncate
    if (3 * ELLIPSIS.length() <= width) {
      return ELLIPSIS + suffix(label.toString(), width - ELLIPSIS.length());
    }
    return label.toString();
  }

  private String describeAction(String name, long nanoTime, int desiredWidth) {
    Action action = actions.get(name);

    String postfix = "";
    long nanoRuntime = nanoTime - actionNanoStartTimes.get(name);
    long runtimeSeconds = nanoRuntime / 1000000000;
    if (runtimeSeconds > SHOW_TIME_THRESHOLD_SECONDS) {
      postfix = " " + runtimeSeconds + "s";
    }

    String message = action.getProgressMessage();
    if (message == null) {
      message = action.prettyPrint();
    }

    if (desiredWidth <= 0) {
      return message + postfix;
    }
    if (message.length() + postfix.length() <= desiredWidth) {
      return message + postfix;
    }
    if (action.getOwner() != null) {
      if (action.getOwner().getLabel() != null) {
        String shortLabel =
            shortenedLabelString(action.getOwner().getLabel(), desiredWidth - postfix.length());
        if (shortLabel.length() + postfix.length() <= desiredWidth) {
          return shortLabel + postfix;
        }
      }
    }
    if (3 * ELLIPSIS.length() <= desiredWidth) {
      message = ELLIPSIS + suffix(message, desiredWidth - ELLIPSIS.length() - postfix.length());
    }
    return message + postfix;
  }

  private void sampleOldestActions(AnsiTerminalWriter terminalWriter) throws IOException {
    int count = 0;
    long nanoTime = clock.nanoTime();
    int actionCount = runningActions.size();
    for (String action : runningActions) {
      count++;
      int width = (count >= SAMPLE_SIZE && count < actionCount) ? targetWidth - 8 : targetWidth - 4;
      terminalWriter.newline().append("    " + describeAction(action, nanoTime, width));
      if (count >= SAMPLE_SIZE) {
        break;
      }
    }
    if (count < actionCount) {
      terminalWriter.append(" ...");
    }
  }

  public void testFilteringComplete(TestFilteringCompleteEvent event) {
    if (event.getTestTargets() != null) {
      totalTests = event.getTestTargets().size();
    }
  }

  public synchronized void testSummary(TestSummary summary) {
    completedTests++;
    mostRecentTest = summary;
    if (summary.getStatus() != BlazeTestStatus.PASSED) {
      failedTests++;
    }
  }


  /***
   * Predicate indicating whether the contents of the progress bar can change, if the
   * only thing that happens is that time passes; this is the case, e.g., if the progress
   * bar shows time information relative to the current time.
   */
  boolean progressBarTimeDependent() {
    if (status != null) {
      return false;
    }
    if (runningActions.size() >= 1) {
      return true;
    }
    if (loadingProgressReceiver != null) {
      // This is kind-of a hack: since the event handler does not get informed about updates
      // in the loading phase, indicate that the progress bar might change even though no
      // explicit update event is known to the event handler.
      return true;
    }
    return false;
  }

  /**
   * Maybe add a note about the last test that passed. Return true, if the note was added (and
   * hence a line break is appropriate if more data is to come. If a null value is provided for
   * the terminal writer, only return wether a note would be added.
   *
   * The width parameter gives advice on to which length the the description of the test should
   * the shortened to, if possible.
   */
  private boolean maybeShowRecentTest(
      AnsiTerminalWriter terminalWriter, boolean shortVersion, int width) throws IOException {
    final String prefix = "; last test: ";
    if (!shortVersion && mostRecentTest != null) {
      if (terminalWriter != null) {
        terminalWriter
            .normal()
            .append(prefix + shortenedLabelString(
                mostRecentTest.getTarget().getLabel(), width - prefix.length()));
      }
      return true;
    } else {
      return false;
    }
  }

  synchronized void writeProgressBar(AnsiTerminalWriter rawTerminalWriter, boolean shortVersion)
      throws IOException {
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(rawTerminalWriter);
    if (status != null) {
      if (ok) {
        terminalWriter.okStatus();
      } else {
        terminalWriter.failStatus();
      }
      terminalWriter.append(status + ":").normal().append(" " + additionalMessage);
      return;
    }
    if (loadingProgressReceiver != null) {
      terminalWriter
          .okStatus()
          .append("Loading:")
          .normal()
          .append(" " + loadingProgressReceiver.progressState());
      return;
    }
    if (executionProgressReceiver != null) {
      terminalWriter.okStatus().append(executionProgressReceiver.getProgressString());
    } else {
      terminalWriter.okStatus().append("Building:");
    }
    if (completedTests > 0) {
      terminalWriter.normal().append(" " + completedTests + " / " + totalTests + " tests");
      if (failedTests > 0) {
        terminalWriter.append(", ").failStatus().append("" + failedTests + " failed").normal();
      }
      terminalWriter.append(";");
    }
    if (runningActions.size() == 0) {
      terminalWriter.normal().append(" no action");
      maybeShowRecentTest(terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
    } else if (runningActions.size() == 1) {
      if (maybeShowRecentTest(null, shortVersion, targetWidth - terminalWriter.getPosition())) {
        // As we will break lines anyway, also show the number of running actions, to keep
        // things stay roughly in the same place (also compensating for the missing plural-s
        // in the word action).
        terminalWriter.normal().append("  1 action");
        maybeShowRecentTest(
            terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
        String statusMessage =
            describeAction(runningActions.peekFirst(), clock.nanoTime(), targetWidth - 4);
        terminalWriter.normal().newline().append("    " + statusMessage);
      } else {
        String statusMessage =
            describeAction(
                runningActions.peekFirst(),
                clock.nanoTime(),
                targetWidth - terminalWriter.getPosition() - 1);
        terminalWriter.normal().append(" " + statusMessage);
      }
    } else {
      if (shortVersion) {
        String statusMessage =
            describeAction(
                runningActions.peekFirst(),
                clock.nanoTime(),
                targetWidth - terminalWriter.getPosition());
        statusMessage += " ... (" + runningActions.size() + " actions)";
        terminalWriter.normal().append(" " + statusMessage);
      } else {
        String statusMessage = "" + runningActions.size() + " actions";
        terminalWriter.normal().append(" " + statusMessage);
        maybeShowRecentTest(
            terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
        sampleOldestActions(terminalWriter);
      }
    }
  }

  void writeProgressBar(AnsiTerminalWriter terminalWriter) throws IOException {
    writeProgressBar(terminalWriter, false);
  }
}
