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
package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.Subscribe;
import com.google.common.primitives.Bytes;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.ActionStatusMessage;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionProgressReceiverAvailableEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.skyframe.LoadingPhaseStartedEvent;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LineCountingAnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LineWrappingAnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LoggingTerminalWriter;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;
import java.util.logging.Logger;
import org.joda.time.format.DateTimeFormat;
import org.joda.time.format.DateTimeFormatter;

/** An experimental new output stream. */
public class ExperimentalEventHandler implements EventHandler {
  private static Logger LOG = Logger.getLogger(ExperimentalEventHandler.class.getName());
  /** Latest refresh of the progress bar, if contents other than time changed */
  static final long MAXIMAL_UPDATE_DELAY_MILLIS = 200L;
  /** Minimal rate limiting (in ms), if the progress bar cannot be updated in place */
  static final long NO_CURSES_MINIMAL_PROGRESS_RATE_LIMIT = 1000L;
  /**
   * Minimal rate limiting, as fraction of the request time so far, if the progress bar cannot be
   * updated in place
   */
  static final double NO_CURSES_MINIMAL_RELATIVE_PROGRESS_RATE_LMIT = 0.15;
  /** Periodic update interval of a time-dependent progress bar if it can be updated in place */
  static final long SHORT_REFRESH_MILLIS = 1000L;
  /** Periodic update interval of a time-dependent progress bar if it cannot be updated in place */
  static final long LONG_REFRESH_MILLIS = 20000L;

  private static final DateTimeFormatter TIMESTAMP_FORMAT =
      DateTimeFormat.forPattern("(HH:mm:ss) ");

  private final boolean cursorControl;
  private final Clock clock;
  private final long uiStartTimeMillis;
  private final AnsiTerminal terminal;
  private final boolean debugAllEvents;
  private final ExperimentalStateTracker stateTracker;
  private final boolean showProgress;
  private final boolean progressInTermTitle;
  private final boolean showTimestamp;
  private final OutErr outErr;
  private long minimalDelayMillis;
  private long minimalUpdateInterval;
  private long lastRefreshMillis;
  private long mustRefreshAfterMillis;
  private boolean dateShown;
  private int numLinesProgressBar;
  private boolean buildComplete;
  // Number of open build even protocol transports.
  private boolean progressBarNeedsRefresh;
  private Thread updateThread;
  private byte[] stdoutBuffer;
  private byte[] stderrBuffer;

  private final long outputLimit;
  private final AtomicLong counter;
  /**
   * The following constants determine how the output limiting is done gracefully. They are all
   * values for the remaining relative capacity left at which we start taking given measure.
   *
   * <p>The degrading of progress updates to stay within output limit is done in the following
   * steps. - We limit progress updates to at most one per second; this is the granularity at which
   * times in he progress bar are shown. So the appearance won't look too bad. Hence we start that
   * measure realatively early. - We only show the short version of the progress bar, even if curses
   * are enabled. - We reduce the update frequency of the progress bar to at most one update per 5s.
   * This still looks as moving and is is line with escalation strategy that so far, every step
   * reduces output by about a factor of 5. - We start decreasing the update frequency to what we
   * would do, if curses were not allowed. Note that now the time between updates is at least a
   * fixed fraction of the time that passed so far; so the time between progress updates will
   * continue to increase.
   */
  private static final double CAPACITY_INCREASE_UPDATE_DELAY = 0.7;

  private static final double CAPACITY_SHORT_PROGRESS_BAR = 0.5;
  private static final double CAPACITY_UPDATE_DELAY_5_SECONDS = 0.4;
  private static final double CAPACITY_UPDATE_DELAY_AS_NO_CURSES = 0.3;
  /**
   * The degrading of printing stdout/stderr is achieved by limiting the output for an individual
   * event if printing it fully would get us above the threshold. If limited, at most a given
   * fraction of the remaining capacity my be used by any such event; larger events are truncated to
   * their end (this is what the user would anyway only see on the terminal if the output is very
   * large). In any case, we always allow at least twice the terminal width, to make the output at
   * least somewhat useful.
   */
  private static final double CAPACITY_LIMIT_OUT_ERR_EVENTS = 0.6;

  private static final double RELATIVE_OUT_ERR_LIMIT = 0.1;

  public final int terminalWidth;

  static class CountingOutputStream extends OutputStream {
    private final OutputStream stream;
    private final AtomicLong counter;

    CountingOutputStream(OutputStream stream, AtomicLong counter) {
      this.stream = stream;
      this.counter = counter;
    }

    @Override
    public void write(int b) throws IOException {
      if (counter.decrementAndGet() >= 0) {
        stream.write(b);
      }
    }
  }

  public ExperimentalEventHandler(
      OutErr outErr, BlazeCommandEventHandler.Options options, Clock clock) {
    this.outputLimit = options.experimentalUiLimitConsoleOutput;
    this.counter = new AtomicLong(outputLimit);
    if (outputLimit > 0) {
      this.outErr =
          OutErr.create(
              new CountingOutputStream(outErr.getOutputStream(), this.counter),
              new CountingOutputStream(outErr.getErrorStream(), this.counter));
    } else {
      // unlimited output; no need to count and limit
      this.outErr = outErr;
    }
    this.cursorControl = options.useCursorControl();
    this.terminal = new AnsiTerminal(this.outErr.getErrorStream());
    this.terminalWidth = (options.terminalColumns > 0 ? options.terminalColumns : 80);
    this.showProgress = options.showProgress;
    this.progressInTermTitle = options.progressInTermTitle && options.useCursorControl();
    this.showTimestamp = options.showTimestamp;
    this.clock = clock;
    this.uiStartTimeMillis = clock.currentTimeMillis();
    this.debugAllEvents = options.experimentalUiDebugAllEvents;
    // If we have cursor control, we try to fit in the terminal width to avoid having
    // to wrap the progress bar. We will wrap the progress bar to terminalWidth - 1
    // characters to avoid depending on knowing whether the underlying terminal does the
    // line feed already when reaching the last character of the line, or only once an
    // additional character is written. Another column is lost for the continuation character
    // in the wrapping process.
    this.stateTracker =
        this.cursorControl
            ? new ExperimentalStateTracker(clock, this.terminalWidth - 2)
            : new ExperimentalStateTracker(clock);
    this.stateTracker.setSampleSize(options.experimentalUiActionsShown);
    this.numLinesProgressBar = 0;
    if (this.cursorControl) {
      this.minimalDelayMillis = Math.round(options.showProgressRateLimit * 1000);
    } else {
      this.minimalDelayMillis =
          Math.max(
              Math.round(options.showProgressRateLimit * 1000),
              NO_CURSES_MINIMAL_PROGRESS_RATE_LIMIT);
    }
    this.minimalUpdateInterval = Math.max(this.minimalDelayMillis, MAXIMAL_UPDATE_DELAY_MILLIS);
    this.stdoutBuffer = new byte[] {};
    this.stderrBuffer = new byte[] {};
    this.dateShown = false;
    // The progress bar has not been updated yet.
    ignoreRefreshLimitOnce();
  }

  /**
   * Return the remaining output capacity, relative to the total capacity, afer a write of the given
   * number of bytes.
   */
  private double remainingCapacity(long wantWrite) {
    if (outputLimit <= 0) {
      // we have unlimited capacity, so we're still at full capacity, regardless of
      // how much we write.
      return 1.0;
    }
    return (counter.get() - wantWrite) / (double) outputLimit;
  }

  private double remainingCapacity() {
    return remainingCapacity(0);
  }

  /**
   * Flush buffers for stdout and stderr. Return if either of them flushed a non-zero number of
   * symbols.
   */
  private synchronized boolean flushStdOutStdErrBuffers() {
    boolean didFlush = false;
    try {
      if (stdoutBuffer.length > 0) {
        outErr.getOutputStream().write(stdoutBuffer);
        outErr.getOutputStream().flush();
        stdoutBuffer = new byte[] {};
        didFlush = true;
      }
      if (stderrBuffer.length > 0) {
        outErr.getErrorStream().write(stderrBuffer);
        outErr.getErrorStream().flush();
        stderrBuffer = new byte[] {};
        didFlush = true;
      }
    } catch (IOException e) {
      LOG.warning("IO Error writing to output stream: " + e);
    }
    return didFlush;
  }

  private synchronized void maybeAddDate() {
    if (!showTimestamp || dateShown || buildComplete) {
      return;
    }
    dateShown = true;
    handle(
        Event.info(
            null,
            "Current date is "
                + DateTimeFormat.forPattern("YYYY-MM-dd").print(clock.currentTimeMillis())));
  }

  @Override
  public synchronized void handle(Event event) {
    try {
      if (debugAllEvents) {
        // Debugging only: show all events visible to the new UI.
        clearProgressBar();
        terminal.flush();
        outErr.getOutputStream().write((event + "\n").getBytes(StandardCharsets.UTF_8));
        outErr.getOutputStream().flush();
        addProgressBar();
        terminal.flush();
      } else {
        maybeAddDate();
        switch (event.getKind()) {
          case STDOUT:
          case STDERR:
            OutputStream stream =
                event.getKind() == EventKind.STDOUT
                    ? outErr.getOutputStream()
                    : outErr.getErrorStream();
            if (buildComplete) {
              stream.write(event.getMessageBytes());
              stream.flush();
            } else {
              byte[] message = event.getMessageBytes();
              if (remainingCapacity(message.length) < CAPACITY_LIMIT_OUT_ERR_EVENTS) {
                // Have to ensure the message is not too large.
                long allowedLength =
                    Math.max(2 * terminalWidth, Math.round(RELATIVE_OUT_ERR_LIMIT * counter.get()));
                if (message.length > allowedLength) {
                  // Have to truncate the message
                  message =
                      Arrays.copyOfRange(
                          message, message.length - (int) allowedLength, message.length);
                  // Mark message as truncated
                  message[0] = '.';
                  message[1] = '.';
                  message[2] = '.';
                }
              }
              int eolIndex = Bytes.lastIndexOf(message, (byte) '\n');
              if (eolIndex >= 0) {
                clearProgressBar();
                terminal.flush();
                stream.write(event.getKind() == EventKind.STDOUT ? stdoutBuffer : stderrBuffer);
                stream.write(Arrays.copyOf(message, eolIndex + 1));
                byte[] restMessage = Arrays.copyOfRange(message, eolIndex + 1, message.length);
                if (event.getKind() == EventKind.STDOUT) {
                  stdoutBuffer = restMessage;
                } else {
                  stderrBuffer = restMessage;
                }
                stream.flush();
                if (showProgress && cursorControl) {
                  addProgressBar();
                }
                terminal.flush();
              } else {
                if (event.getKind() == EventKind.STDOUT) {
                  stdoutBuffer = Bytes.concat(stdoutBuffer, message);
                } else {
                  stderrBuffer = Bytes.concat(stderrBuffer, message);
                }
              }
            }
            break;
          case ERROR:
          case FAIL:
          case WARNING:
          case INFO:
          case SUBCOMMAND:
            boolean incompleteLine;
            if (showProgress && !buildComplete) {
              clearProgressBar();
            }
            incompleteLine = flushStdOutStdErrBuffers();
            if (incompleteLine) {
              crlf();
            }
            if (showTimestamp) {
              terminal.writeString(TIMESTAMP_FORMAT.print(clock.currentTimeMillis()));
            }
            setEventKindColor(event.getKind());
            terminal.writeString(event.getKind() + ": ");
            terminal.resetTerminal();
            incompleteLine = true;
            if (event.getLocation() != null) {
              terminal.writeString(event.getLocation() + ": ");
            }
            if (event.getMessage() != null) {
              terminal.writeString(event.getMessage());
              incompleteLine = !event.getMessage().endsWith("\n");
            }
            if (incompleteLine) {
              crlf();
            }
            if (showProgress && !buildComplete && cursorControl) {
              addProgressBar();
            }
            terminal.flush();
            break;
          case PROGRESS:
            if (stateTracker.progressBarTimeDependent()) {
              refresh();
            }
            break;
          case START:
          case FINISH:
          case PASS:
          case TIMEOUT:
          case DEPCHECKER:
            break;
        }
      }
    } catch (IOException e) {
      LOG.warning("IO Error writing to output stream: " + e);
    }
  }

  private void setEventKindColor(EventKind kind) throws IOException {
    switch (kind) {
      case ERROR:
      case FAIL:
        terminal.textRed();
        terminal.textBold();
        break;
      case WARNING:
        terminal.textMagenta();
        break;
      case INFO:
        terminal.textGreen();
        break;
      case SUBCOMMAND:
        terminal.textBlue();
        break;
      default:
        terminal.resetTerminal();
    }
  }

  @Subscribe
  public void buildStarted(BuildStartingEvent event) {
    maybeAddDate();
    stateTracker.buildStarted(event);
    // As a new phase started, inform immediately.
    ignoreRefreshLimitOnce();
    refresh();
  }

  @Subscribe
  public void loadingStarted(LoadingPhaseStartedEvent event) {
    maybeAddDate();
    stateTracker.loadingStarted(event);
    // As a new phase started, inform immediately.
    ignoreRefreshLimitOnce();
    refresh();
    startUpdateThread();
  }

  @Subscribe
  public void loadingComplete(LoadingPhaseCompleteEvent event) {
    stateTracker.loadingComplete(event);
    refresh();
  }

  @Subscribe
  public synchronized void analysisComplete(AnalysisPhaseCompleteEvent event) {
    String analysisSummary = stateTracker.analysisComplete(event);
    handle(Event.info(null, analysisSummary));
  }

  @Subscribe
  public void progressReceiverAvailable(ExecutionProgressReceiverAvailableEvent event) {
    stateTracker.progressReceiverAvailable(event);
    // As this is the first time we have a progress message, update immediately.
    ignoreRefreshLimitOnce();
    startUpdateThread();
  }

  @Subscribe
  public void buildComplete(BuildCompleteEvent event) {
    // The final progress bar will flow into the scroll-back buffer, to if treat
    // it as an event and add a timestamp, if events are supposed to have a timestmap.
    synchronized (this) {
      stateTracker.buildComplete(event);
      ignoreRefreshLimitOnce();
      refresh();

      // After a build has completed, only stop updating the UI if there is no more BEP
      // upload happening.
      if (stateTracker.pendingTransports() == 0) {
        buildComplete = true;
        stopUpdateThread();
        flushStdOutStdErrBuffers();
      }
    }
  }

  @Subscribe
  public void noBuild(NoBuildEvent event) {
    synchronized (this) {
      buildComplete = true;
    }
    stopUpdateThread();
    flushStdOutStdErrBuffers();
  }

  @Subscribe
  public void afterCommand(AfterCommandEvent event) {
    synchronized (this) {
      buildComplete = true;
    }
    stopUpdateThread();
  }

  @Subscribe
  public void downloadProgress(FetchProgress event) {
    maybeAddDate();
    stateTracker.downloadProgress(event);
    refresh();
  }

  @Subscribe
  public void actionStarted(ActionStartedEvent event) {
    stateTracker.actionStarted(event);
    refresh();
  }

  @Subscribe
  public void actionStatusMessage(ActionStatusMessage event) {
    stateTracker.actionStatusMessage(event);
    refresh();
  }

  @Subscribe
  public void actionCompletion(ActionCompletionEvent event) {
    stateTracker.actionCompletion(event);
    refreshSoon();
  }

  @Subscribe
  public void testFilteringComplete(TestFilteringCompleteEvent event) {
    stateTracker.testFilteringComplete(event);
    refresh();
  }

  /**
   * Return true, if the test summary provides information that is both
   * worth being shown in the scroll-back buffer and new with respect to
   * the alreay shown failure messages.
   */
  private boolean testSummaryProvidesNewInformation(TestSummary summary) {
    ImmutableSet<BlazeTestStatus> statusToIgnore =
        ImmutableSet.of(
            BlazeTestStatus.PASSED,
            BlazeTestStatus.FAILED_TO_BUILD,
            BlazeTestStatus.BLAZE_HALTED_BEFORE_TESTING,
            BlazeTestStatus.NO_STATUS);

    if (statusToIgnore.contains(summary.getStatus())) {
      return false;
    }
    if (summary.getStatus() == BlazeTestStatus.FAILED && summary.getFailedLogs().size() == 1) {
      return false;
    }
    return true;
  }

  @Subscribe
  public synchronized void testSummary(TestSummary summary) {
    stateTracker.testSummary(summary);
    if (testSummaryProvidesNewInformation(summary)) {
      // For failed test, write the failure to the scroll-back buffer immediately
      try {
        clearProgressBar();
        crlf();
        setEventKindColor(EventKind.ERROR);
        terminal.writeString("" + summary.getStatus() + ": ");
        terminal.resetTerminal();
        terminal.writeString(summary.getTarget().getLabel().toString());
        terminal.writeString(" (Summary)");
        crlf();
        for (Path logPath : summary.getFailedLogs()) {
          terminal.writeString("      " + logPath.getPathString());
          crlf();
        }
        if (showProgress && cursorControl) {
          addProgressBar();
        }
        terminal.flush();
      } catch (IOException e) {
        LOG.warning("IO Error writing to output stream: " + e);
      }
    } else {
      refresh();
    }
  }

  @Subscribe
  public synchronized void buildEventTransportsAnnounced(AnnounceBuildEventTransportsEvent event) {
    stateTracker.buildEventTransportsAnnounced(event);
    if (debugAllEvents) {
      String message = "Transports announced:";
      for (BuildEventTransport transport : event.transports()) {
        message += " " + transport.name();
      }
      this.handle(Event.info(null, message));
    }
  }

  @Subscribe
  public synchronized void buildEventTransportClosed(BuildEventTransportClosedEvent event) {
    stateTracker.buildEventTransportClosed(event);
    if (debugAllEvents) {
      this.handle(Event.info(null, "Transport " + event.transport().name() + " closed"));
    }

    if (stateTracker.pendingTransports() == 0) {
      stopUpdateThread();
      flushStdOutStdErrBuffers();
      ignoreRefreshLimitOnce();
      refresh();
    } else {
      refresh();
    }
  }

  private void refresh() {
    if (showProgress) {
      progressBarNeedsRefresh = true;
      doRefresh();
    }
  }

  private void doRefresh(boolean fromUpdateThread) {
    if (buildComplete) {
      return;
    }
    long nowMillis = clock.currentTimeMillis();
    if (lastRefreshMillis + minimalDelayMillis < nowMillis) {
      synchronized (this) {
        try {
          if (showProgress && (progressBarNeedsRefresh || timeBasedRefresh())) {
            progressBarNeedsRefresh = false;
            clearProgressBar();
            addProgressBar();
            terminal.flush();
            double remaining = remainingCapacity();
            if (remaining < CAPACITY_INCREASE_UPDATE_DELAY) {
              // Increase the update interval if the start producing too much output
              minimalDelayMillis = Math.max(minimalDelayMillis, 1000);
              if (remaining < CAPACITY_UPDATE_DELAY_5_SECONDS) {
                minimalDelayMillis = Math.max(minimalDelayMillis, 5000);
              }
            }
            if (!cursorControl || remaining < CAPACITY_UPDATE_DELAY_AS_NO_CURSES) {
              // If we can't update the progress bar in place, make sure we increase the update
              // interval as time progresses, to avoid too many progress messages in place.
              minimalDelayMillis =
                  Math.max(
                      minimalDelayMillis,
                      Math.round(
                          NO_CURSES_MINIMAL_RELATIVE_PROGRESS_RATE_LMIT
                              * (clock.currentTimeMillis() - uiStartTimeMillis)));
              minimalUpdateInterval = Math.max(minimalDelayMillis, MAXIMAL_UPDATE_DELAY_MILLIS);
            }
          }
        } catch (IOException e) {
          LOG.warning("IO Error writing to output stream: " + e);
        }
      }
    } else {
      // We skipped an update due to rate limiting. If this however, turned
      // out to be the last update for a long while, we need to show it in a
      // timely manner, as it best describes the current state.
      if (!fromUpdateThread) {
        startUpdateThread();
      }
    }
  }

  private void doRefresh() {
    doRefresh(false);
  }

  private void refreshSoon() {
    // Schedule an update of the progress bar in the near future, unless there is already
    // a future update scheduled.
    long nowMillis = clock.currentTimeMillis();
    synchronized (this) {
      if (mustRefreshAfterMillis <= lastRefreshMillis) {
        mustRefreshAfterMillis = Math.max(nowMillis + minimalUpdateInterval, lastRefreshMillis + 1);
      }
    }
    startUpdateThread();
  }

  /**
   * Decide wheter the progress bar should be redrawn only for the reason
   * that time has passed.
   */
  private synchronized boolean timeBasedRefresh() {
    if (!stateTracker.progressBarTimeDependent()) {
      return false;
    }
    long nowMillis = clock.currentTimeMillis();
    long intervalMillis = cursorControl ? SHORT_REFRESH_MILLIS : LONG_REFRESH_MILLIS;
    if (lastRefreshMillis < mustRefreshAfterMillis
        && mustRefreshAfterMillis < nowMillis + minimalDelayMillis) {
      // Within the a smal interval from now, an update is scheduled anyway,
      // so don't do a time-based update of the progress bar now, to avoid
      // updates too close to each other.
      return false;
    }
    return lastRefreshMillis + intervalMillis < nowMillis;
  }

  private void ignoreRefreshLimitOnce() {
    // Set refresh time variables in a state such that the next progress bar
    // update will definitely be written out.
    lastRefreshMillis = clock.currentTimeMillis() - minimalDelayMillis - 1;
  }

  private void startUpdateThread() {
    Thread threadToStart = null;
    synchronized (this) {
      // Refuse to start an update thread once the build is complete; such a situation might
      // arise if the completion of the build is reported (shortly) before the completion of
      // the last action is reported.
      if (!buildComplete && updateThread == null) {
        final ExperimentalEventHandler eventHandler = this;
        updateThread =
            new Thread(
                new Runnable() {
                  @Override
                  public void run() {
                    try {
                      while (true) {
                        Thread.sleep(minimalUpdateInterval);
                        if (lastRefreshMillis < mustRefreshAfterMillis
                            && mustRefreshAfterMillis < clock.currentTimeMillis()) {
                          progressBarNeedsRefresh = true;
                        }
                        eventHandler.doRefresh(/* fromUpdateThread= */ true);
                      }
                    } catch (InterruptedException e) {
                      // Ignore
                    }
                  }
                });
        threadToStart = updateThread;
      }
    }
    if (threadToStart != null) {
      threadToStart.start();
    }
  }

  private void stopUpdateThread() {
    Thread threadToWaitFor = null;
    synchronized (this) {
      if (updateThread != null) {
        threadToWaitFor = updateThread;
        updateThread = null;
      }
    }
    if (threadToWaitFor != null) {
      threadToWaitFor.interrupt();
      Uninterruptibles.joinUninterruptibly(threadToWaitFor);
    }
  }

  public void resetTerminal() {
    try {
      terminal.resetTerminal();
    } catch (IOException e) {
      LOG.warning("IO Error writing to user terminal: " + e);
    }
  }

  private void clearProgressBar() throws IOException {
    if (!cursorControl) {
      return;
    }
    for (int i = 0; i < numLinesProgressBar; i++) {
      terminal.cr();
      terminal.cursorUp(1);
      terminal.clearLine();
    }
    numLinesProgressBar = 0;
  }

  private void crlf() throws IOException {
    terminal.cr();
    terminal.writeString("\n");
  }

  private synchronized void addProgressBar() throws IOException {
    LineCountingAnsiTerminalWriter countingTerminalWriter =
        new LineCountingAnsiTerminalWriter(terminal);
    AnsiTerminalWriter terminalWriter = countingTerminalWriter;
    lastRefreshMillis = clock.currentTimeMillis();
    if (cursorControl) {
      terminalWriter = new LineWrappingAnsiTerminalWriter(terminalWriter, terminalWidth - 1);
    }
    String timestamp = null;
    if (showTimestamp) {
      timestamp = TIMESTAMP_FORMAT.print(clock.currentTimeMillis());
    }
    stateTracker.writeProgressBar(
        terminalWriter,
        /* shortVersion=*/ !cursorControl || remainingCapacity() < CAPACITY_SHORT_PROGRESS_BAR,
        timestamp);
    terminalWriter.newline();
    numLinesProgressBar = countingTerminalWriter.getWrittenLines();
    if (progressInTermTitle) {
      LoggingTerminalWriter stringWriter = new LoggingTerminalWriter(true);
      stateTracker.writeProgressBar(stringWriter, true);
      terminal.setTitle(stringWriter.getTranscript());
    }
  }
}
