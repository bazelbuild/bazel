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
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildtool.ExecutionProgressReceiver;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionProgressReceiverAvailableEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.skyframe.LoadingPhaseStartedEvent;
import com.google.devtools.build.lib.skyframe.PackageProgressReceiver;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.PositionAwareAnsiTerminalWriter;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;

import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;

/**
 * An experimental state tracker for the new experimental UI.
 */
class ExperimentalStateTracker {

  static final long SHOW_TIME_THRESHOLD_SECONDS = 3;
  static final String ELLIPSIS = "...";

  static final int NANOS_PER_SECOND = 1000000000;

  private int sampleSize = 3;

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
  private final Map<String, String> actionStrategy;

  // For each test, the list of actions (again identified by the path of the
  // primary output) currently running for that test (identified by its label),
  // in order they got started. A key is present in the map if and only if that
  // was discovered as a test.
  private final Map<Label, Set<String>> testActions;

  private int actionsCompleted;
  private int totalTests;
  private int completedTests;
  private TestSummary mostRecentTest;
  private int failedTests;
  private boolean ok;

  private ExecutionProgressReceiver executionProgressReceiver;
  private PackageProgressReceiver packageProgressReceiver;

  ExperimentalStateTracker(Clock clock, int targetWidth) {
    this.runningActions = new ArrayDeque<>();
    this.actions = new TreeMap<>();
    this.actionNanoStartTimes = new TreeMap<>();
    this.actionStrategy = new TreeMap<>();
    this.testActions = new TreeMap<>();
    this.ok = true;
    this.clock = clock;
    this.targetWidth = targetWidth;
  }

  ExperimentalStateTracker(Clock clock) {
    this(clock, 0);
  }

  /**
   * Set the maximal number of actions shown in the progress bar.
   */
  void setSampleSize(int sampleSize) {
    if (sampleSize >= 1) {
      this.sampleSize = sampleSize;
    } else {
      this.sampleSize = 1;
    }
  }

  void buildStarted(BuildStartingEvent event) {
    status = "Loading";
    additionalMessage = "";
  }

  void loadingStarted(LoadingPhaseStartedEvent event) {
    status = null;
    packageProgressReceiver = event.getPackageProgressReceiver();
  }

  void loadingComplete(LoadingPhaseCompleteEvent event) {
    int count = event.getTargets().size();
    status = "Analyzing";
    if (count == 1) {
      additionalMessage = "target " + event.getTargets().asList().get(0).getLabel();
    } else {
      additionalMessage = "" + count + " targets";
    }
  }

  /**
   * Make the state tracker aware of the fact that the analyis has finished. Return a summary of the
   * work done in the analysis phase.
   */
  synchronized String analysisComplete(AnalysisPhaseCompleteEvent event) {
    String workDone = "Analysed " + additionalMessage;
    if (packageProgressReceiver != null) {
      Pair<String, String> progress = packageProgressReceiver.progressState();
      workDone += " (" + progress.getFirst() + ")";
    }
    workDone += ".";
    status = null;
    packageProgressReceiver = null;
    return workDone;
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
    if (action.getOwner() != null) {
      Label owner = action.getOwner().getLabel();
      if (owner != null) {
        Set<String> testActionsForOwner = testActions.get(owner);
        if (testActionsForOwner != null) {
          testActionsForOwner.add(name);
        }
      }
    }
  }

  void actionStatusMessage(ActionStatusMessage event) {
    String strategy = event.getStrategy();
    if (strategy != null) {
      String name = event.getActionMetadata().getPrimaryOutput().getPath().getPathString();
      synchronized (this) {
        actionStrategy.put(name, strategy);
      }
    }
  }

  synchronized void actionCompletion(ActionCompletionEvent event) {
    actionsCompleted++;
    Action action = event.getAction();
    String name = action.getPrimaryOutput().getPath().getPathString();
    runningActions.remove(name);
    actions.remove(name);
    actionNanoStartTimes.remove(name);
    actionStrategy.remove(name);

    if (action.getOwner() != null) {
      Label owner = action.getOwner().getLabel();
      if (owner != null) {
        Set<String> testActionsForOwner = testActions.get(owner);
        if (testActionsForOwner != null) {
          testActionsForOwner.remove(name);
        }
      }
    }

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

  // Describe a group of actions running for the same test.
  private String describeTestGroup(
      Label owner, long nanoTime, int desiredWidth, Set<String> allActions) {
    String prefix = "Testing ";
    String labelSep = " [";
    String postfix = " (" + allActions.size() + " actions)]";
    // Leave enough room for at least 3 samples of run times, each 4 characters
    // (a digit, 's', comma, and space).
    int labelWidth = desiredWidth - prefix.length() - labelSep.length() - postfix.length() - 12;
    StringBuffer message =
        new StringBuffer(prefix).append(shortenedLabelString(owner, labelWidth)).append(labelSep);

    // Compute the remaining width for the sample times, but if the desired width is too small
    // anyway, then show at least one sample.
    int remainingWidth = desiredWidth - message.length() - postfix.length();
    if (remainingWidth < 0) {
      remainingWidth = 5;
    }

    String sep = "";
    int count = 0;
    for (String action : allActions) {
      long nanoRuntime = nanoTime - actionNanoStartTimes.get(action);
      long runtimeSeconds = nanoRuntime / NANOS_PER_SECOND;
      String text = sep + runtimeSeconds + "s";
      if (remainingWidth < text.length()) {
        break;
      }
      message.append(text);
      remainingWidth -= text.length();
      count++;
      sep = ", ";
    }
    if (count == allActions.size()) {
      postfix = "]";
    }
    return message.append(postfix).toString();
  }

  // Describe an action by a string of the desired length; if describing that action includes
  // describing other actions, add those to the to set of actions to skip in further samples of
  // actions.
  private String describeAction(String name, long nanoTime, int desiredWidth, Set<String> toSkip) {
    Action action = actions.get(name);
    if (action.getOwner() != null) {
      Label owner = action.getOwner().getLabel();
      if (owner != null) {
        Set<String> allRelatedActions = testActions.get(owner);
        if (allRelatedActions != null && allRelatedActions.size() > 1) {
          if (toSkip != null) {
            toSkip.addAll(allRelatedActions);
          }
          return describeTestGroup(owner, nanoTime, desiredWidth, allRelatedActions);
        }
      }
    }

    String postfix = "";
    long nanoRuntime = nanoTime - actionNanoStartTimes.get(name);
    long runtimeSeconds = nanoRuntime / NANOS_PER_SECOND;
    String strategy = actionStrategy.get(name);
    // To keep the UI appearance more stable, always show the elapsed
    // time if we also show a strategy (otherwise the strategy will jump in
    // the progress bar).
    if (strategy != null || runtimeSeconds > SHOW_TIME_THRESHOLD_SECONDS) {
      postfix = "; " + runtimeSeconds + "s";
    }
    if (strategy != null) {
      postfix += " " + strategy;
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

    // We have to shorten the message to fit into the line.

    if (action.getOwner() != null) {
      if (action.getOwner().getLabel() != null) {
        // First attempt is to shorten the package path string in the messge, if it occurs there
        String pathString = action.getOwner().getLabel().getPackageFragment().toString();
        int pathIndex = message.indexOf(pathString);
        if (pathIndex >= 0) {
          String start = message.substring(0, pathIndex);
          String end = message.substring(pathIndex + pathString.length());
          int pathTargetLength = desiredWidth - start.length() - end.length() - postfix.length();
          // This attempt of shortening is reasonable if what is left from the label
          // is significantly longer (twice as long) as the ellipsis symbols introduced.
          if (pathTargetLength >= 3 * ELLIPSIS.length()) {
            String shortPath = suffix(pathString, pathTargetLength - ELLIPSIS.length());
            int slashPos = shortPath.indexOf('/');
            if (slashPos >= 0) {
              return start + ELLIPSIS + shortPath.substring(slashPos) + end + postfix;
            }
          }
        }

        // Second attempt: just take a shortened version of the label.
        String shortLabel =
            shortenedLabelString(action.getOwner().getLabel(), desiredWidth - postfix.length());
        if (shortLabel.length() + postfix.length() <= desiredWidth) {
          return shortLabel + postfix;
        }
      }
    }
    if (3 * ELLIPSIS.length() + postfix.length() <= desiredWidth) {
      message = ELLIPSIS + suffix(message, desiredWidth - ELLIPSIS.length() - postfix.length());
    }

    return message + postfix;
  }

  private void sampleOldestActions(AnsiTerminalWriter terminalWriter) throws IOException {
    int count = 0;
    int totalCount = 0;
    long nanoTime = clock.nanoTime();
    int actionCount = runningActions.size();
    Set<String> toSkip = new TreeSet<>();
    for (String action : runningActions) {
      totalCount++;
      if (toSkip.contains(action)) {
        continue;
      }
      count++;
      if (count > sampleSize) {
        totalCount--;
        break;
      }
      int width = (count >= sampleSize && count < actionCount) ? targetWidth - 8 : targetWidth - 4;
      terminalWriter.newline().append("    " + describeAction(action, nanoTime, width, toSkip));
    }
    if (totalCount < actionCount) {
      terminalWriter.append(" ...");
    }
  }

  public synchronized void testFilteringComplete(TestFilteringCompleteEvent event) {
    if (event.getTestTargets() != null) {
      totalTests = event.getTestTargets().size();
      for (ConfiguredTarget target : event.getTestTargets()) {
        if (target.getLabel() != null) {
          testActions.put(target.getLabel(), new LinkedHashSet<String>());
        }
      }
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
    if (packageProgressReceiver != null) {
      return true;
    }
    if (status != null) {
      return false;
    }
    if (runningActions.size() >= 1) {
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
        terminalWriter.normal().append(prefix);
        if (mostRecentTest.getStatus() == BlazeTestStatus.PASSED) {
          terminalWriter.okStatus();
        } else {
          terminalWriter.failStatus();
        }
        terminalWriter.append(
            shortenedLabelString(mostRecentTest.getTarget().getLabel(), width - prefix.length()));
        terminalWriter.normal();
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
      if (packageProgressReceiver != null) {
        Pair<String, String> progress = packageProgressReceiver.progressState();
        terminalWriter.append(" (" + progress.getFirst() + ")");
        if (progress.getSecond().length() > 0) {
          terminalWriter.newline().append("    " + progress.getSecond());
        }
      }
      return;
    }
    if (packageProgressReceiver != null) {
      Pair<String, String> progress = packageProgressReceiver.progressState();
      terminalWriter.okStatus().append("Loading:").normal().append(" " + progress.getFirst());
      if (progress.getSecond().length() > 0) {
        terminalWriter.newline().append("    " + progress.getSecond());
      }
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
      terminalWriter.normal().append(" no action running");
      maybeShowRecentTest(terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
    } else if (runningActions.size() == 1) {
      if (maybeShowRecentTest(null, shortVersion, targetWidth - terminalWriter.getPosition())) {
        // As we will break lines anyway, also show the number of running actions, to keep
        // things stay roughly in the same place (also compensating for the missing plural-s
        // in the word action).
        terminalWriter.normal().append("  1 action running");
        maybeShowRecentTest(
            terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
        String statusMessage =
            describeAction(runningActions.peekFirst(), clock.nanoTime(), targetWidth - 4, null);
        terminalWriter.normal().newline().append("    " + statusMessage);
      } else {
        String statusMessage =
            describeAction(
                runningActions.peekFirst(),
                clock.nanoTime(),
                targetWidth - terminalWriter.getPosition() - 1,
                null);
        terminalWriter.normal().append(" " + statusMessage);
      }
    } else {
      if (shortVersion) {
        String statusMessage =
            describeAction(
                runningActions.peekFirst(),
                clock.nanoTime(),
                targetWidth - terminalWriter.getPosition(),
                null);
        statusMessage += " ... (" + runningActions.size() + " actions running)";
        terminalWriter.normal().append(" " + statusMessage);
      } else {
        String statusMessage = "" + runningActions.size() + " actions running";
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
