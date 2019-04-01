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

import static java.util.concurrent.TimeUnit.MILLISECONDS;

import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.AnalyzingActionEvent;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.actions.SchedulingActionEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildtool.ExecutionProgressReceiver;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionProgressReceiverAvailableEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.skyframe.ConfigurationPhaseStartedEvent;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetProgressReceiver;
import com.google.devtools.build.lib.skyframe.LoadingPhaseStartedEvent;
import com.google.devtools.build.lib.skyframe.PackageProgressReceiver;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.PositionAwareAnsiTerminalWriter;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * An experimental state tracker for the new experimental UI.
 */
class ExperimentalStateTracker {

  static final long SHOW_TIME_THRESHOLD_SECONDS = 3;
  static final String ELLIPSIS = "...";
  static final String FETCH_PREFIX = "    Fetching ";
  static final String AND_MORE = " ...";
  static final String NO_STATUS = "-----";
  static final int STATUS_LENGTH = 5;

  static final int NANOS_PER_SECOND = 1000000000;
  static final String URL_PROTOCOL_SEP = "://";

  private int sampleSize = 3;

  private String status;
  private String additionalMessage;

  private final Clock clock;

  // Desired maximal width of the progress bar, if positive.
  // Non-positive values indicate not to aim for a particular width.
  private final int targetWidth;

  private static class ActionState {
    public final Action action;
    public final long nanoStartTime;
    public final boolean executing;
    public final String status;

    private ActionState(Action action, long nanoStartTime, boolean executing, String status) {
      this.action = action;
      this.nanoStartTime = nanoStartTime;
      this.executing = executing;
      this.status = status;
    }
  }

  private final Map<String, ActionState> activeActions;
  private final Map<String, String> notStartedActionStatus;

  // running downloads are identified by the original URL they were trying to access.
  private final Deque<String> runningDownloads;
  private final Map<String, Long> downloadNanoStartTimes;
  private final Map<String, FetchProgress> downloads;

  // For each test, the list of actions (again identified by the path of the
  // primary output) currently running for that test (identified by its label),
  // in order they got started. A key is present in the map if and only if that
  // was discovered as a test.
  private final Map<Label, Set<String>> testActions;

  private final AtomicInteger actionsCompleted;
  private int totalTests;
  private int completedTests;
  private TestSummary mostRecentTest;
  private int failedTests;
  private boolean ok;
  private boolean buildComplete;
  private String defaultStatus = "Loading";
  private String defaultActivity = "loading...";

  private ExecutionProgressReceiver executionProgressReceiver;
  private PackageProgressReceiver packageProgressReceiver;
  private ConfiguredTargetProgressReceiver configuredTargetProgressReceiver;

  // Set of build event protocol transports that need yet to be closed.
  private Set<BuildEventTransport> bepOpenTransports = new HashSet<>();
  // The point in time when closing of BEP transports was started.
  private long bepTransportClosingStartTimeMillis;

  ExperimentalStateTracker(Clock clock, int targetWidth) {
    this.activeActions = new ConcurrentHashMap<>();
    this.notStartedActionStatus = new ConcurrentHashMap<>();

    this.actionsCompleted = new AtomicInteger();
    this.testActions = new ConcurrentHashMap<>();
    this.runningDownloads = new ArrayDeque<>();
    this.downloads = new TreeMap<>();
    this.downloadNanoStartTimes = new TreeMap<>();
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

  void configurationStarted(ConfigurationPhaseStartedEvent event) {
    configuredTargetProgressReceiver = event.getConfiguredTargetProgressReceiver();
  }

  void loadingComplete(LoadingPhaseCompleteEvent event) {
    int count = event.getLabels().size();
    status = "Analyzing";
    if (count == 1) {
      additionalMessage = "target " + Iterables.getOnlyElement(event.getLabels());
    } else {
      additionalMessage = "" + count + " targets";
    }
  }

  /**
   * Make the state tracker aware of the fact that the analysis has finished. Return a summary of
   * the work done in the analysis phase.
   */
  synchronized String analysisComplete(AnalysisPhaseCompleteEvent event) {
    String workDone = "Analyzed " + additionalMessage;
    if (packageProgressReceiver != null) {
      Pair<String, String> progress = packageProgressReceiver.progressState();
      workDone += " (" + progress.getFirst();
      if (configuredTargetProgressReceiver != null) {
        workDone += ", " + configuredTargetProgressReceiver.getProgressString();
      }
      workDone += ")";
    }
    workDone += ".";
    status = null;
    packageProgressReceiver = null;
    configuredTargetProgressReceiver = null;
    defaultStatus = "Building";
    defaultActivity = "checking cached actions";
    return workDone;
  }

  void progressReceiverAvailable(ExecutionProgressReceiverAvailableEvent event) {
    executionProgressReceiver = event.getExecutionProgressReceiver();
  }

  void buildComplete(BuildCompleteEvent event, String additionalInfo) {
    buildComplete = true;
    // Build event protocol transports are closed right after the build complete event.
    bepTransportClosingStartTimeMillis = clock.currentTimeMillis();

    if (event.getResult().getSuccess()) {
      status = "INFO";
      int actionsCompleted = this.actionsCompleted.get();
      if (failedTests == 0) {
        additionalMessage =
            additionalInfo + "Build completed successfully, "
            + actionsCompleted + " total action" + (actionsCompleted == 1 ? "" : "s");
      } else {
        additionalMessage =
            additionalInfo + "Build completed, "
            + failedTests + " test" + (failedTests == 1 ? "" : "s") + " FAILED, "
            + actionsCompleted + " total action" + (actionsCompleted == 1 ? "" : "s");
      }
    } else {
      ok = false;
      status = "FAILED";
      additionalMessage = additionalInfo + "Build did NOT complete successfully";
    }
  }

  void buildComplete(BuildCompleteEvent event) {
    buildComplete(event, "");
  }

  synchronized void downloadProgress(FetchProgress event) {
    String url = event.getResourceIdentifier();
    if (event.isFinished()) {
      // a download is finished, clean it up
      runningDownloads.remove(url);
      downloadNanoStartTimes.remove(url);
      downloads.remove(url);
    } else if (runningDownloads.contains(url)) {
      // a new progress update on an already known, still running download
      downloads.put(url, event);
    } else {
      // Start of a new download
      long nanoTime = clock.nanoTime();
      runningDownloads.add(url);
      downloads.put(url, event);
      downloadNanoStartTimes.put(url, nanoTime);
    }
  }

  void actionStarted(ActionStartedEvent event) {
    Action action = event.getAction();
    String name = action.getPrimaryOutput().getPath().getPathString();
    long nanoStartTime = event.getNanoTimeStart();

    String status = notStartedActionStatus.remove(name);
    boolean nowExecuting = status != null;
    activeActions.put(
        name,
        new ActionState(action, nanoStartTime, nowExecuting, nowExecuting ? status : null));

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

  void analyzingAction(AnalyzingActionEvent event) {
    String name = event.getActionMetadata().getPrimaryOutput().getPath().getPathString();
    ActionState state = activeActions.remove(name);
    if (state != null) {
      activeActions.put(
          name, new ActionState(state.action, clock.nanoTime(), /*executing=*/ false, "Analyzing"));
    }
    notStartedActionStatus.remove(name);
  }

  void schedulingAction(SchedulingActionEvent event) {
    String name = event.getActionMetadata().getPrimaryOutput().getPath().getPathString();
    ActionState state = activeActions.remove(name);
    if (state != null) {
      activeActions.put(
          name,
          new ActionState(state.action, clock.nanoTime(), /*executing=*/ false, "Scheduling"));
    }
    notStartedActionStatus.remove(name);
  }

  void runningAction(RunningActionEvent event) {
    String strategy = event.getStrategy();
    String name = event.getActionMetadata().getPrimaryOutput().getPath().getPathString();

    ActionState state = activeActions.remove(name);
    if (state != null) {
      activeActions.put(
          name, new ActionState(state.action, clock.nanoTime(), /*executing=*/ true, strategy));
    } else {
      notStartedActionStatus.put(name, strategy);
    }
  }

  void actionCompletion(ActionCompletionEvent event) {
    actionsCompleted.incrementAndGet();
    Action action = event.getAction();
    String name = action.getPrimaryOutput().getPath().getPathString();
    activeActions.remove(name);
    boolean removed = notStartedActionStatus.containsKey(name);
    if (removed && !action.discoversInputs()) {
      BugReport.sendBugReport(
          new IllegalStateException(
              "Should not complete an action before starting it, and action did not discover "
                  + "inputs, so should not have published a status before execution: "
                  + action));
    }

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
      executionProgressReceiver.actionCompleted(event.getActionLookupData());
    }
  }

  /**
   * From a string, take a suffix of at most the given length.
   */
  static String suffix(String s, int len) {
    if (len <= 0) {
      return "";
    }
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

  private String shortenedString(String s, int maxWidth) {
    if (maxWidth <= 3 * ELLIPSIS.length() || s.length() <= maxWidth) {
      return s;
    }
    return s.substring(0, maxWidth - ELLIPSIS.length()) + ELLIPSIS;
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
      remainingWidth = "[1s], ".length() + 1;
    }

    String sep = "";
    boolean allReported = true;
    for (String action : allActions) {
      ActionState actionState = activeActions.get(action);
      if (actionState == null) {
        // This action must have completed while we were constructing this output. Skip it.
        continue;
      }
      long nanoRuntime = nanoTime - actionState.nanoStartTime;
      long runtimeSeconds = nanoRuntime / NANOS_PER_SECOND;
      String text =
          actionState.executing ? sep + runtimeSeconds + "s" : sep + "[" + runtimeSeconds + "s]";
      if (remainingWidth < text.length()) {
        allReported = false;
        break;
      }
      message.append(text);
      remainingWidth -= text.length();
      sep = ", ";
    }
    return message.append(allReported ? "]" : postfix).toString();
  }

  // Describe an action by a string of the desired length; if describing that action includes
  // describing other actions, add those to the to set of actions to skip in further samples of
  // actions.
  private String describeAction(
      ActionState actionState, long nanoTime, int desiredWidth, Set<String> toSkip) {
    Action action = actionState.action;
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
    String prefix = "";
    long nanoRuntime = nanoTime - actionState.nanoStartTime;
    long runtimeSeconds = nanoRuntime / NANOS_PER_SECOND;
    String strategy = null;
    if (actionState.executing) {
      strategy = actionState.status;
    } else {
      String status = actionState.status;
      if (status == null) {
        status = NO_STATUS;
      }
      if (status.length() > STATUS_LENGTH) {
        status = status.substring(0, STATUS_LENGTH);
      }
      prefix = prefix + "[" + status + "] ";
    }
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
      return prefix + message + postfix;
    }
    if (prefix.length() + message.length() + postfix.length() <= desiredWidth) {
      return prefix + message + postfix;
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
          int pathTargetLength =
              desiredWidth - start.length() - end.length() - postfix.length() - prefix.length();
          // This attempt of shortening is reasonable if what is left from the label
          // is significantly longer (twice as long) as the ellipsis symbols introduced.
          if (pathTargetLength >= 3 * ELLIPSIS.length()) {
            String shortPath = suffix(pathString, pathTargetLength - ELLIPSIS.length());
            int slashPos = shortPath.indexOf('/');
            if (slashPos >= 0) {
              return prefix + start + ELLIPSIS + shortPath.substring(slashPos) + end + postfix;
            }
          }
        }

        // Second attempt: just take a shortened version of the label.
        String shortLabel =
            shortenedLabelString(
                action.getOwner().getLabel(), desiredWidth - postfix.length() - prefix.length());
        if (prefix.length() + shortLabel.length() + postfix.length() <= desiredWidth) {
          return prefix + shortLabel + postfix;
        }
      }
    }
    if (3 * ELLIPSIS.length() + postfix.length() + prefix.length() <= desiredWidth) {
      message =
          ELLIPSIS
              + suffix(
                  message, desiredWidth - ELLIPSIS.length() - postfix.length() - prefix.length());
    }

    return prefix + message + postfix;
  }

  private ActionState getOldestAction() {
    long minStart = Long.MAX_VALUE;
    ActionState result = null;
    for (ActionState action : activeActions.values()) {
      if (action.nanoStartTime < minStart) {
        minStart = action.nanoStartTime;
        result = action;
      }
    }
    return result;
  }

  private String countActions() {
    // TODO(djasper): Iterating over the actions here is slow, but it's only done once per refresh
    // and thus might be faster than trying to update these values in the critical path.
    // Re-investigate if this ever turns up in a profile.
    int actionsCount = 0;
    int executingActionsCount = 0;
    for (ActionState actionState : activeActions.values()) {
      actionsCount++;
      if (actionState.executing) {
        executingActionsCount++;
      }
    }

    if (actionsCount == 1) {
      return " 1 action running";
    } else if (actionsCount == executingActionsCount) {
      return "" + actionsCount + " actions running";
    } else {
      return "" + actionsCount + " actions, " + executingActionsCount + " running";
    }
  }

  private void sampleOldestActions(AnsiTerminalWriter terminalWriter) throws IOException {
    int count = 0;
    int totalCount = 0;
    long nanoTime = clock.nanoTime();
    int actionCount = activeActions.size();
    Set<String> toSkip = new TreeSet<>();
    ArrayList<Map.Entry<String, ActionState>> copy = new ArrayList<>(activeActions.entrySet());
    copy.sort(
        Comparator.comparing((Map.Entry<String, ActionState> entry) -> !entry.getValue().executing)
            .thenComparing(entry -> entry.getValue().nanoStartTime));
    for (Map.Entry<String, ActionState> entry : copy) {
      totalCount++;
      if (toSkip.contains(entry.getKey())) {
        continue;
      }
      count++;
      if (count > sampleSize) {
        totalCount--;
        break;
      }
      int width =
          targetWidth - 4 - ((count >= sampleSize && count < actionCount) ? AND_MORE.length() : 0);
      terminalWriter
          .newline()
          .append("    " + describeAction(entry.getValue(), nanoTime, width, toSkip));
    }
    if (totalCount < actionCount) {
      terminalWriter.append(AND_MORE);
    }
  }

  public synchronized void testFilteringComplete(TestFilteringCompleteEvent event) {
    if (event.getTestTargets() != null) {
      totalTests = event.getTestTargets().size();
      for (ConfiguredTarget target : event.getTestTargets()) {
        if (target.getLabel() != null) {
          testActions.put(target.getLabel(), Sets.newConcurrentHashSet());
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

  synchronized void buildEventTransportsAnnounced(AnnounceBuildEventTransportsEvent event) {
    this.bepOpenTransports.addAll(event.transports());
  }

  synchronized void buildEventTransportClosed(BuildEventTransportClosedEvent event) {
    bepOpenTransports.remove(event.transport());
  }

  synchronized int pendingTransports() {
    return bepOpenTransports.size();
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
    if (runningDownloads.size() >= 1) {
      return true;
    }
    if (buildComplete && !bepOpenTransports.isEmpty()) {
      return true;
    }
    if (status != null) {
      return false;
    }
    if (activeActions.size() >= 1) {
      return true;
    }
    return false;
  }

  /**
   * Maybe add a note about the last test that passed. Return true, if the note was added (and hence
   * a line break is appropriate if more data is to come. If a null value is provided for the
   * terminal writer, only return whether a note would be added.
   *
   * <p>The width parameter gives advice on to which length the description of the test should the
   * shortened to, if possible.
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
        terminalWriter.append(shortenedLabelString(
            mostRecentTest.getLabel(), width - prefix.length()));
        terminalWriter.normal();
      }
      return true;
    } else {
      return false;
    }
  }

  private String shortenUrl(String url, int width) {

    if (url.length() < width) {
      return url;
    }

    // Try to shorten to the form prot://host/.../rest/path/filename
    String prefix = "";
    int protocolIndex = url.indexOf(URL_PROTOCOL_SEP);
    if (protocolIndex > 0) {
      prefix = url.substring(0, protocolIndex + URL_PROTOCOL_SEP.length() + 1);
      url = url.substring(protocolIndex + URL_PROTOCOL_SEP.length() + 1);
      int hostIndex = url.indexOf("/");
      if (hostIndex > 0) {
        prefix = prefix + url.substring(0, hostIndex + 1);
        url = url.substring(hostIndex + 1);
        int targetLength = width - prefix.length();
        // accept this form of shortening, if what is left from the filename is
        // significantly longer (twice as long) as the ellipsis symbol introduced
        if (targetLength > 3 * ELLIPSIS.length()) {
          String shortPath = suffix(url, targetLength - ELLIPSIS.length());
          int slashPos = shortPath.indexOf("/");
          if (slashPos >= 0) {
            return prefix + ELLIPSIS + shortPath.substring(slashPos);
          } else {
            return prefix + ELLIPSIS + shortPath;
          }
        }
      }
    }

    // Last resort: just take a suffix
    if (width <= ELLIPSIS.length()) {
      // No chance to shorten anyway
      return "";
    }
    return ELLIPSIS + suffix(url, width - ELLIPSIS.length());
  }

  private void reportOnOneDownload(
      String url, long nanoTime, int width, AnsiTerminalWriter terminalWriter) throws IOException {

    String postfix = "";

    FetchProgress download = downloads.get(url);
    long nanoDownloadTime = nanoTime - downloadNanoStartTimes.get(url);
    long downloadSeconds = nanoDownloadTime / NANOS_PER_SECOND;

    String progress = download.getProgress();
    if (progress.length() > 0) {
      postfix = postfix + " " + progress;
    }
    if (downloadSeconds > SHOW_TIME_THRESHOLD_SECONDS) {
      postfix = postfix + " " + downloadSeconds + "s";
    }
    if (postfix.length() > 0) {
      postfix = ";" + postfix;
    }
    url = shortenUrl(url, Math.max(width - postfix.length(), 3 * ELLIPSIS.length()));
    terminalWriter.append(url + postfix);
  }

  private void reportOnDownloads(AnsiTerminalWriter terminalWriter) throws IOException {
    int count = 0;
    long nanoTime = clock.nanoTime();
    int downloadCount = runningDownloads.size();
    String suffix = AND_MORE + " (" + downloadCount + " fetches)";
    for (String url : runningDownloads) {
      if (count >= sampleSize) {
        break;
      }
      count++;
      terminalWriter.newline().append(FETCH_PREFIX);
      reportOnOneDownload(
          url,
          nanoTime,
          targetWidth
              - FETCH_PREFIX.length()
              - ((count >= sampleSize && count < downloadCount) ? suffix.length() : 0),
          terminalWriter);
    }
    if (count < downloadCount) {
      terminalWriter.append(suffix);
    }
  }

  /**
   * Display any BEP transports that are still open after the build. Most likely, because
   * uploading build events takes longer than the build itself.
   */
  private void maybeReportBepTransports(PositionAwareAnsiTerminalWriter terminalWriter)
      throws IOException {
    if (!buildComplete || bepOpenTransports.isEmpty()) {
      return;
    }
    long sinceSeconds =
        MILLISECONDS.toSeconds(clock.currentTimeMillis() - bepTransportClosingStartTimeMillis);
    if (sinceSeconds == 0) {
      // Special case for when bazel was interrupted, in which case we don't want to have
      // a BEP upload message.
      return;
    }
    int count = bepOpenTransports.size();
    // Can just use targetWidth, because we always write to a new line
    int maxWidth = targetWidth;

    String waitMessage = "Waiting for build events upload: ";
    String name = bepOpenTransports.iterator().next().name();
    String line = waitMessage + name + " " + sinceSeconds + "s";

    if (count == 1 && line.length() <= maxWidth) {
      terminalWriter.newline().append(line);
    } else if (count == 1) {
      waitMessage = "Waiting for: ";
      String waitSecs = " " + sinceSeconds + "s";
      int maxNameWidth = maxWidth - waitMessage.length() - waitSecs.length();
      terminalWriter.newline().append(waitMessage + shortenedString(name, maxNameWidth) + waitSecs);
    } else {
      terminalWriter.newline().append(waitMessage + sinceSeconds + "s");
      for (BuildEventTransport transport : bepOpenTransports) {
        name = "  " + transport.name();
        terminalWriter.newline().append(shortenedString(name, maxWidth));
      }
    }
  }

  synchronized void writeProgressBar(
      AnsiTerminalWriter rawTerminalWriter, boolean shortVersion, String timestamp)
      throws IOException {
    PositionAwareAnsiTerminalWriter terminalWriter =
        new PositionAwareAnsiTerminalWriter(rawTerminalWriter);
    int actionsCount = activeActions.size();
    if (timestamp != null) {
      terminalWriter.append(timestamp);
    }
    if (status != null) {
      if (ok) {
        terminalWriter.okStatus();
      } else {
        terminalWriter.failStatus();
      }
      terminalWriter.append(status + ":").normal().append(" " + additionalMessage);
      if (packageProgressReceiver != null) {
        Pair<String, String> progress = packageProgressReceiver.progressState();
        terminalWriter.append(" (" + progress.getFirst());
        if (configuredTargetProgressReceiver != null) {
          terminalWriter.append(", " + configuredTargetProgressReceiver.getProgressString());
        }
        terminalWriter.append(")");
        if (progress.getSecond().length() > 0 && !shortVersion) {
          terminalWriter.newline().append("    " + progress.getSecond());
        }
      }
      if (!shortVersion) {
        reportOnDownloads(terminalWriter);
        maybeReportBepTransports(terminalWriter);
      }
      return;
    }
    if (packageProgressReceiver != null) {
      Pair<String, String> progress = packageProgressReceiver.progressState();
      terminalWriter.okStatus().append("Loading:").normal().append(" " + progress.getFirst());
      if (progress.getSecond().length() > 0) {
        terminalWriter.newline().append("    " + progress.getSecond());
      }
      if (!shortVersion) {
        reportOnDownloads(terminalWriter);
      }
      return;
    }
    if (executionProgressReceiver != null) {
      terminalWriter.okStatus().append(executionProgressReceiver.getProgressString());
    } else {
      terminalWriter.okStatus().append(defaultStatus).append(":");
    }
    if (completedTests > 0) {
      terminalWriter.normal().append(" " + completedTests + " / " + totalTests + " tests");
      if (failedTests > 0) {
        terminalWriter.append(", ").failStatus().append("" + failedTests + " failed").normal();
      }
      terminalWriter.append(";");
    }
    // Get the oldest action. Note that actions might have finished in the meantime and thus there
    // might not be one.
    ActionState oldestAction = getOldestAction();
    if (actionsCount == 0 || oldestAction == null) {
      terminalWriter.normal().append(" ").append(defaultActivity);
      maybeShowRecentTest(terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
    } else if (actionsCount == 1) {
      if (maybeShowRecentTest(null, shortVersion, targetWidth - terminalWriter.getPosition())) {
        // As we will break lines anyway, also show the number of running actions, to keep
        // things stay roughly in the same place (also compensating for the missing plural-s
        // in the word action).
        terminalWriter.normal().append("  1 action");
        maybeShowRecentTest(
            terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
        String statusMessage =
            describeAction(oldestAction, clock.nanoTime(), targetWidth - 4, null);
        terminalWriter.normal().newline().append("    " + statusMessage);
      } else {
        String statusMessage =
            describeAction(
                oldestAction,
                clock.nanoTime(),
                targetWidth - terminalWriter.getPosition() - 1,
                null);
        terminalWriter.normal().append(" " + statusMessage);
      }
    } else {
      if (shortVersion) {
        String statusMessage =
            describeAction(
                oldestAction, clock.nanoTime(), targetWidth - terminalWriter.getPosition(), null);
        statusMessage += " ... (" + countActions() + ")";
        terminalWriter.normal().append(" " + statusMessage);
      } else {
        String statusMessage = countActions();
        terminalWriter.normal().append(" " + statusMessage);
        maybeShowRecentTest(
            terminalWriter, shortVersion, targetWidth - terminalWriter.getPosition());
        sampleOldestActions(terminalWriter);
      }
    }
    if (!shortVersion) {
      reportOnDownloads(terminalWriter);
      maybeReportBepTransports(terminalWriter);
    }
  }

  void writeProgressBar(AnsiTerminalWriter terminalWriter, boolean shortVersion)
      throws IOException {
    writeProgressBar(terminalWriter, shortVersion, null);
  }

  void writeProgressBar(AnsiTerminalWriter terminalWriter) throws IOException {
    writeProgressBar(terminalWriter, false);
  }
}
