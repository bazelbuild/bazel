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

import com.google.common.base.Splitter;
import com.google.common.collect.ImmutableSet;
import com.google.common.eventbus.AllowConcurrentEvents;
import com.google.common.eventbus.Subscribe;
import com.google.common.flogger.GoogleLogger;
import com.google.common.primitives.Bytes;
import com.google.common.util.concurrent.Uninterruptibles;
import com.google.devtools.build.lib.actions.ActionCompletionEvent;
import com.google.devtools.build.lib.actions.ActionScanningCompletedEvent;
import com.google.devtools.build.lib.actions.ActionStartedEvent;
import com.google.devtools.build.lib.actions.RunningActionEvent;
import com.google.devtools.build.lib.actions.ScanningActionEvent;
import com.google.devtools.build.lib.actions.SchedulingActionEvent;
import com.google.devtools.build.lib.actions.StoppedScanningActionEvent;
import com.google.devtools.build.lib.analysis.AnalysisPhaseCompleteEvent;
import com.google.devtools.build.lib.analysis.NoBuildEvent;
import com.google.devtools.build.lib.analysis.NoBuildRequestFinishedEvent;
import com.google.devtools.build.lib.buildeventstream.AnnounceBuildEventTransportsEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransport;
import com.google.devtools.build.lib.buildeventstream.BuildEventTransportClosedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionProgressReceiverAvailableEvent;
import com.google.devtools.build.lib.buildtool.buildevent.TestFilteringCompleteEvent;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.events.ExtendedEventHandler.FetchProgress;
import com.google.devtools.build.lib.pkgcache.LoadingPhaseCompleteEvent;
import com.google.devtools.build.lib.pkgcache.PathPackageLocator;
import com.google.devtools.build.lib.skyframe.ConfigurationPhaseStartedEvent;
import com.google.devtools.build.lib.skyframe.LoadingPhaseStartedEvent;
import com.google.devtools.build.lib.syntax.Location;
import com.google.devtools.build.lib.util.io.AnsiTerminal;
import com.google.devtools.build.lib.util.io.AnsiTerminal.Color;
import com.google.devtools.build.lib.util.io.AnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.FileOutErr.OutputReference;
import com.google.devtools.build.lib.util.io.LineCountingAnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LineWrappingAnsiTerminalWriter;
import com.google.devtools.build.lib.util.io.LoggingTerminalWriter;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.test.TestStatus.BlazeTestStatus;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.time.Instant;
import java.time.ZoneId;
import java.time.format.DateTimeFormatter;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;
import javax.annotation.Nullable;

/** An experimental new output stream. */
public class UiEventHandler implements EventHandler {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
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

  /**
   * Even if the output is not limited, we restrict the message size to something we can still
   * handle internally. This is the maximal size specified here. Currently, it is the maximal length
   * of a byte[] acceptable by {@code new String(message, 0, message.length,
   * StandardCharsets.UTF_8}. (In JDK9+, if the message buffer contains a byte whose high bit is
   * set, a UTF-8 decoding path is taken that allocates a new byte[] buffer twice as large as the
   * message byte[] buffer)
   */
  static final int MAXIMAL_MESSAGE_LENGTH = (Integer.MAX_VALUE - 8) >> 1;

  private static final DateTimeFormatter TIMESTAMP_FORMAT =
      DateTimeFormatter.ofPattern("(HH:mm:ss) ");
  private static final DateTimeFormatter DATE_FORMAT = DateTimeFormatter.ofPattern("yyyy-MM-dd");

  private final boolean cursorControl;
  private final Clock clock;
  private final long uiStartTimeMillis;
  private final AnsiTerminal terminal;
  private final boolean debugAllEvents;
  private final UiStateTracker stateTracker;
  private final LocationPrinter locationPrinter;
  private final boolean showProgress;
  private final boolean progressInTermTitle;
  private final boolean showTimestamp;
  private final boolean deduplicate;
  private final OutErr outErr;
  private final ImmutableSet<EventKind> filteredEvents;
  private long minimalDelayMillis;
  private long minimalUpdateInterval;
  private long lastRefreshMillis;
  private long mustRefreshAfterMillis;
  private boolean dateShown;
  private int numLinesProgressBar;
  private boolean buildRunning;
  // Number of open build even protocol transports.
  private boolean progressBarNeedsRefresh;
  private volatile boolean shutdown;
  private final AtomicReference<Thread> updateThread;
  private final Lock updateLock;
  private byte[] stdoutBuffer;
  private byte[] stderrBuffer;
  private final Set<String> messagesSeen;
  private long deduplicateCount;

  private final long outputLimit;
  private long reservedOutputCapacity;
  private final AtomicLong counter;
  private long droppedEvents;
  /**
   * The following constants determine how the output limiting is done gracefully. They are all
   * values for the remaining relative capacity left at which we start taking given measure.
   *
   * <p>The degrading of progress updates to stay within output limit is done in the following
   * steps.
   *
   * <ul>
   *   <li>We limit progress updates to at most one per second; this is the granularity at which
   *       times in the progress bar are shown. So the appearance won't look too bad. Hence we start
   *       that measure relatively early.
   *   <li>We only show the short version of the progress bar, even if curses are enabled.
   *   <li>We reduce the update frequency of the progress bar to at most one update per 5s. This
   *       still looks moving and is in line with the escalation strategy that so far: every step
   *       reduces output by about a factor of 5.
   *   <li>We start decreasing the update frequency to what we would do, if curses were not allowed.
   *       Note that now the time between updates is at least a fixed fraction of the time that
   *       passed so far; so the time between progress updates will continue to increase.
   *   <li>We do not show any event, except for errors.
   *   <li>The last small fraction of the output, we reserve for a post-build status messages (in
   *       particular test summaries).
   * </ul>
   */
  private static final double CAPACITY_INCREASE_UPDATE_DELAY = 0.9;

  private static final double CAPACITY_SHORT_PROGRESS_BAR = 0.6;
  private static final double CAPACITY_UPDATE_DELAY_5_SECONDS = 0.4;
  private static final double CAPACITY_UPDATE_DELAY_AS_NO_CURSES = 0.3;
  private static final double CAPACITY_ERRORS_ONLY = 0.2;
  /**
   * The degrading of printing stdout/stderr is achieved by limiting the output for an individual
   * event if printing it fully would get us above the threshold. If limited, at most a given
   * fraction of the remaining capacity my be used by any such event; larger events are truncated to
   * their end (this is what the user would anyway only see on the terminal if the output is very
   * large). In any case, we always allow at least twice the terminal width, to make the output at
   * least somewhat useful. From a given threshold onwards, we always restrict to at most twice the
   * terminal width.
   */
  private static final double CAPACITY_LIMIT_OUT_ERR_EVENTS = 0.8;

  private static final double CAPACITY_STRONG_LIMIT_OUT_ERR_EVENTS = 0.5;
  private static final double RELATIVE_OUT_ERR_LIMIT = 0.1;

  /**
   * The reservation of output capacity for the final status is computed as follows: we always
   * reserve at least a certain numer of lines, and at least a certain fraction of the overall
   * capacity, to show more status in scenarios where we have a bigger limit.
   */
  private static final long MINIMAL_POST_BUILD_OUTPUT_LINES = 14;

  private static final double MINIMAL_POST_BUILD_OUTPUT_CAPACITY = 0.05;

  public final int terminalWidth;

  /**
   * An output stream that wraps another output stream and that fully buffers writes until flushed.
   * Additionally, it optionally takes into account a budget for the number of bytes it may still
   * write to the wrapped stream.
   */
  private static class FullyBufferedOutputStreamMaybeWithCounting extends ByteArrayOutputStream {
    /** The (possibly unbuffered) stream wrapped by this one. */
    private final OutputStream wrapped;
    /** The counter for the amount of bytes we're still allowed to write */
    @Nullable private final AtomicLong counter;

    /**
     * Constructs a new fully-buffered output stream that wraps an unbuffered one.
     *
     * @param wrapped the (possibly unbuffered) stream wrapped by this one
     * @param counter a counter specifying the number of bytes the stream may still write
     */
    FullyBufferedOutputStreamMaybeWithCounting(OutputStream wrapped, @Nullable AtomicLong counter) {
      this.wrapped = wrapped;
      this.counter = counter;
    }

    @Override
    public void flush() throws IOException {
      super.flush();
      try {
        if (counter == null || counter.addAndGet(-count) >= 0) {
          writeTo(wrapped);
          wrapped.flush();
        }
      } finally {
        // If we failed to write our current buffered contents to the output, there is not much
        // we can do because reporting an error would require another write, and that write would
        // probably fail. So, instead, we silently discard whatever was previously buffered in the
        // hopes that the data itself was what caused the problem.
        reset();
      }
    }
  }

  public UiEventHandler(
      OutErr outErr, UiOptions options, Clock clock, @Nullable PathFragment workspacePathFragment) {
    this.terminalWidth = (options.terminalColumns > 0 ? options.terminalColumns : 80);
    this.outputLimit = options.experimentalUiLimitConsoleOutput;
    this.counter = new AtomicLong(outputLimit);
    this.droppedEvents = 0;
    if (outputLimit > 0) {
      this.outErr =
          OutErr.create(
              new FullyBufferedOutputStreamMaybeWithCounting(
                  outErr.getOutputStream(), this.counter),
              new FullyBufferedOutputStreamMaybeWithCounting(
                  outErr.getErrorStream(), this.counter));
      reservedOutputCapacity =
          Math.max(
              MINIMAL_POST_BUILD_OUTPUT_LINES * this.terminalWidth,
              Math.round(MINIMAL_POST_BUILD_OUTPUT_CAPACITY * outputLimit));
    } else {
      // unlimited output; no need to count, but still fully buffer
      this.outErr =
          OutErr.create(
              new FullyBufferedOutputStreamMaybeWithCounting(outErr.getOutputStream(), null),
              new FullyBufferedOutputStreamMaybeWithCounting(outErr.getErrorStream(), null));
    }
    this.cursorControl = options.useCursorControl();
    this.terminal = new AnsiTerminal(this.outErr.getErrorStream());
    this.showProgress = options.showProgress;
    this.progressInTermTitle = options.progressInTermTitle && options.useCursorControl();
    this.showTimestamp = options.showTimestamp;
    this.clock = clock;
    this.uiStartTimeMillis = clock.currentTimeMillis();
    this.debugAllEvents = options.experimentalUiDebugAllEvents;
    this.deduplicate = options.experimentalUiDeduplicate;
    this.messagesSeen = new HashSet<>();
    this.locationPrinter =
        new LocationPrinter(options.attemptToPrintRelativePaths, workspacePathFragment);
    // If we have cursor control, we try to fit in the terminal width to avoid having
    // to wrap the progress bar. We will wrap the progress bar to terminalWidth - 1
    // characters to avoid depending on knowing whether the underlying terminal does the
    // line feed already when reaching the last character of the line, or only once an
    // additional character is written. Another column is lost for the continuation character
    // in the wrapping process.
    this.stateTracker =
        this.cursorControl
            ? new UiStateTracker(clock, this.terminalWidth - 2)
            : new UiStateTracker(clock);
    this.stateTracker.setProgressMode(options.uiProgressMode, options.uiSamplesShown);
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
    this.updateThread = new AtomicReference<>();
    this.updateLock = new ReentrantLock();
    this.filteredEvents = ImmutableSet.copyOf(options.eventFilters);
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
    return (counter.get() - wantWrite - reservedOutputCapacity) / (double) outputLimit;
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
      logger.atWarning().withCause(e).log("IO Error writing to output stream");
    }
    return didFlush;
  }

  private synchronized void maybeAddDate() {
    if (!showTimestamp || dateShown || !buildRunning) {
      return;
    }
    dateShown = true;
    handle(
        Event.info(
            null,
            "Current date is "
                + DATE_FORMAT.format(
                    Instant.ofEpochMilli(clock.currentTimeMillis())
                        .atZone(ZoneId.systemDefault()))));
  }

  @Override
  public void handle(Event event) {
    if (!debugAllEvents
        && !showTimestamp
        && (event.getKind() == EventKind.START
            || event.getKind() == EventKind.FINISH
            || event.getKind() == EventKind.PASS
            || event.getKind() == EventKind.TIMEOUT
            || event.getKind() == EventKind.DEPCHECKER)) {
      // Keep this in sync with the list of no-op event kinds in handleLocked below.
      return;
    }
    handleLocked(event, /* isFollowUp= */ false);
  }

  private synchronized void handleLocked(Event event, boolean isFollowUp) {
    if (this.filteredEvents.contains(event.getKind())) {
      return;
    }
    try {
      if (debugAllEvents) {
        // Debugging only: show all events visible to the new UI.
        clearProgressBar();
        terminal.flush();
        OutputStream stream = outErr.getOutputStream();
        stream.write((event + "\n").getBytes(StandardCharsets.ISO_8859_1));
        byte[] stdout = event.getStdOut();
        if (stdout != null) {
          stream.write("... with STDOUT: ".getBytes(StandardCharsets.ISO_8859_1));
          stream.write(stdout);
          stream.write("\n".getBytes(StandardCharsets.ISO_8859_1));
        }
        byte[] stderr = event.getStdErr();
        if (stderr != null) {
          stream.write("... with STDERR: ".getBytes(StandardCharsets.ISO_8859_1));
          stream.write(stderr);
          stream.write("\n".getBytes(StandardCharsets.ISO_8859_1));
        }
        stream.flush();
        addProgressBar();
        terminal.flush();
      } else {
        if (!isFollowUp
            && (remainingCapacity() < CAPACITY_ERRORS_ONLY)
            && (event.getKind() != EventKind.ERROR)) {
          droppedEvents++;
          return;
        }
        if (shouldDeduplicate(event)) {
          return;
        }
        maybeAddDate();
        switch (event.getKind()) {
          case STDOUT:
          case STDERR:
            OutputStream stream =
                event.getKind() == EventKind.STDOUT
                    ? outErr.getOutputStream()
                    : outErr.getErrorStream();
            if (!buildRunning) {
              stream.write(event.getMessageBytes());
              stream.flush();
            } else {
              if (remainingCapacity() < 0) {
                return;
              }
              writeToStream(
                  stream,
                  event.getKind(),
                  event.getMessageReference(),
                  /* readdProgressBar= */ showProgress && cursorControl);
            }
            break;
          case ERROR:
          case FAIL:
          case WARNING:
          case CANCELLED:
          case INFO:
          case DEBUG:
          case SUBCOMMAND:
            boolean incompleteLine;
            if (showProgress && buildRunning) {
              clearProgressBar();
            }
            incompleteLine = flushStdOutStdErrBuffers();
            if (incompleteLine) {
              crlf();
            }
            if (remainingCapacity() < 0) {
              terminal.flush();
              return;
            }
            if (showTimestamp) {
              terminal.writeString(
                  TIMESTAMP_FORMAT.format(
                      Instant.ofEpochMilli(clock.currentTimeMillis())
                          .atZone(ZoneId.systemDefault())));
            }
            setEventKindColor(event.getKind());
            terminal.writeString(event.getKind() + ": ");
            terminal.resetTerminal();
            incompleteLine = true;
            Location location = event.getLocation();
            if (location != null) {
              terminal.writeString(locationPrinter.getLocationString(location) + ": ");
            }
            if (event.getMessage() != null) {
              terminal.writeString(event.getMessage());
              incompleteLine = !event.getMessage().endsWith("\n");
            }
            if (incompleteLine) {
              crlf();
            }
            if (showProgress && buildRunning && cursorControl) {
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
        if (event.hasStdoutStderr()) {
          clearProgressBar();
          terminal.flush();
          writeToStream(
              outErr.getErrorStream(),
              EventKind.STDERR,
              event.getStdErrReference(),
              /* readdProgressBar= */ false);
          outErr.getErrorStream().flush();
          writeToStream(
              outErr.getOutputStream(),
              EventKind.STDOUT,
              event.getStdOutReference(),
              /* readdProgressBar= */ false);
          outErr.getOutputStream().flush();
          if (showProgress && cursorControl) {
            addProgressBar();
          }
          terminal.flush();
        }
      }
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("IO Error writing to output stream");
    }
  }

  private void writeToStream(
      OutputStream stream, EventKind eventKind, OutputReference reference, boolean readdProgressBar)
      throws IOException {
    byte[] message;
    double cap = remainingCapacity(reference.getLength());
    if (cap < CAPACITY_LIMIT_OUT_ERR_EVENTS) {
      // Have to ensure the message is not too large.
      long allowedLength =
          Math.max(2 * terminalWidth, Math.round(RELATIVE_OUT_ERR_LIMIT * counter.get()));
      if (cap < CAPACITY_STRONG_LIMIT_OUT_ERR_EVENTS) {
        allowedLength = Math.min(allowedLength, 2 * terminalWidth);
      }
      message = reference.getFinalBytes((int) allowedLength);
      if (reference.getLength() > allowedLength) {
        // Mark message as truncated
        message[0] = '.';
        message[1] = '.';
        message[2] = '.';
      }
    } else {
      message = reference.getFinalBytes(MAXIMAL_MESSAGE_LENGTH);
      if (message.length == MAXIMAL_MESSAGE_LENGTH) {
        logger.atWarning().log("truncated message longer than %d bytes", MAXIMAL_MESSAGE_LENGTH);
      }
    }
    int eolIndex = Bytes.lastIndexOf(message, (byte) '\n');
    if (eolIndex >= 0) {
      clearProgressBar();
      terminal.flush();
      stream.write(eventKind == EventKind.STDOUT ? stdoutBuffer : stderrBuffer);
      stream.write(Arrays.copyOf(message, eolIndex + 1));
      byte[] restMessage = Arrays.copyOfRange(message, eolIndex + 1, message.length);
      if (eventKind == EventKind.STDOUT) {
        stdoutBuffer = restMessage;
      } else {
        stderrBuffer = restMessage;
      }
      stream.flush();
      if (readdProgressBar) {
        addProgressBar();
        terminal.flush();
      }
    } else {
      if (eventKind == EventKind.STDOUT) {
        stdoutBuffer = Bytes.concat(stdoutBuffer, message);
      } else {
        stderrBuffer = Bytes.concat(stderrBuffer, message);
      }
    }
  }

  private void setEventKindColor(EventKind kind) throws IOException {
    switch (kind) {
      case ERROR:
      case FAIL:
        terminal.setTextColor(Color.RED);
        terminal.textBold();
        break;
      case WARNING:
      case CANCELLED:
        terminal.setTextColor(Color.MAGENTA);
        break;
      case INFO:
        terminal.setTextColor(Color.GREEN);
        break;
      case DEBUG:
        terminal.setTextColor(Color.YELLOW);
        break;
      case SUBCOMMAND:
        terminal.setTextColor(Color.BLUE);
        break;
      default:
        terminal.resetTerminal();
    }
  }

  // TODO(jmmv): This feature needs to be removed. It has only caused confusion in the past
  // (see cl/261885840 and https://github.com/bazelbuild/bazel/issues/7184) and would let us
  // simplify the code here significantly.
  private boolean shouldDeduplicate(Event event) {
    if (!deduplicate) {
      // deduplication disabled
      return false;
    }
    if (event.getKind() == EventKind.DEBUG) {
      String message = event.getMessage();
      synchronized (this) {
        if (messagesSeen.contains(message)) {
          deduplicateCount++;
          return true;
        } else {
          messagesSeen.add(message);
        }
      }
    }
    if (event.getKind() != EventKind.INFO) {
      // only deduplicate INFO messages
      return false;
    }
    byte[] stdout = event.getStdOut();
    byte[] stderr = event.getStdErr();
    if (stdout == null && stderr == null) {
      // We deduplicate on the attached output (assuming the event itself only describes
      // the source of the output). If no output is attached it is a different kind of event
      // and should not be deduplicated.
      return false;
    }
    boolean allMessagesSeen = true;
    if (stdout != null) {
      String stdoutString = new String(stdout, StandardCharsets.ISO_8859_1);
      for (String line : Splitter.on("\n").split(stdoutString)) {
        if (!messagesSeen.contains(line)) {
          allMessagesSeen = false;
          messagesSeen.add(line);
        }
      }
    }
    if (stderr != null) {
      String stderrString = new String(stderr, StandardCharsets.ISO_8859_1);
      for (String line : Splitter.on("\n").split(stderrString)) {
        if (!messagesSeen.contains(line)) {
          allMessagesSeen = false;
          messagesSeen.add(line);
        }
      }
    }
    if (allMessagesSeen) {
      synchronized (this) {
        deduplicateCount++;
      }
    }
    return allMessagesSeen;
  }

  @Subscribe
  public void buildStarted(BuildStartingEvent event) {
    synchronized (this) {
      buildRunning = true;
    }
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
  public void configurationStarted(ConfigurationPhaseStartedEvent event) {
    maybeAddDate();
    stateTracker.configurationStarted(event);
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
    String analysisSummary = stateTracker.analysisComplete();
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
    // it as an event and add a timestamp, if events are supposed to have a timestamp.
    boolean done = false;
    synchronized (this) {
      stateTracker.buildComplete(event);
      reservedOutputCapacity = 0;
      ignoreRefreshLimitOnce();
      refresh();

      // After a build has completed, only stop updating the UI if there is no more BEP
      // upload happening.
      if (stateTracker.pendingTransports() == 0) {
        buildRunning = false;
        done = true;
      }
    }
    if (done) {
      stopUpdateThread();
      flushStdOutStdErrBuffers();
    }
  }

  private void completeBuild() {
    synchronized (this) {
      if (!buildRunning) {
        return;
      }
      buildRunning = false;
    }
    stopUpdateThread();
    synchronized (this) {
      try {
        // If a progress bar is currently present, clean it and redraw it.
        boolean progressBarPresent = numLinesProgressBar > 0;
        if (progressBarPresent) {
          clearProgressBar();
        }
        terminal.flush();
        boolean incompleteLine = flushStdOutStdErrBuffers();
        if (incompleteLine) {
          crlf();
        }
        if (deduplicateCount > 0) {
          handle(Event.info(null, "deduplicated " + deduplicateCount + " events"));
        }
        if (droppedEvents > 0) {
          handleLocked(
              Event.info(
                  null,
                  "dropped "
                      + droppedEvents
                      + " events on the console,"
                      + " to stay within output limit."),
              /* isFollowUp= */ true);
        }
        if (progressBarPresent) {
          addProgressBar();
        }
        terminal.flush();
      } catch (IOException e) {
        logger.atWarning().withCause(e).log("IO Error writing to output stream");
      }
    }
  }

  @Subscribe
  public void packageLocatorCreated(PathPackageLocator packageLocator) {
    locationPrinter.packageLocatorCreated(packageLocator);
  }

  @Subscribe
  public void noBuild(NoBuildEvent event) {
    if (event.showProgress()) {
      synchronized (this) {
        buildRunning = true;
      }
      return;
    }
    completeBuild();
  }

  @Subscribe
  public void noBuildFinished(NoBuildRequestFinishedEvent event) {
    completeBuild();
  }

  @Subscribe
  public void afterCommand(AfterCommandEvent event) {
    synchronized (this) {
      buildRunning = true;
    }
    completeBuild();
    try {
      terminal.resetTerminal();
      terminal.flush();
    } catch (IOException e) {
      logger.atWarning().withCause(e).log("IO Error writing to user terminal");
    }
  }

  @Subscribe
  public void downloadProgress(FetchProgress event) {
    maybeAddDate();
    stateTracker.downloadProgress(event);
    refresh();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void actionStarted(ActionStartedEvent event) {
    stateTracker.actionStarted(event);
    refresh();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void scanningAction(ScanningActionEvent event) {
    stateTracker.scanningAction(event);
    refresh();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void stopScanningAction(StoppedScanningActionEvent event) {
    stateTracker.stopScanningAction(event);
    refresh();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void schedulingAction(SchedulingActionEvent event) {
    stateTracker.schedulingAction(event);
    refresh();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void runningAction(RunningActionEvent event) {
    stateTracker.runningAction(event);
    refresh();
  }

  @Subscribe
  @AllowConcurrentEvents
  public void actionCompletion(ActionScanningCompletedEvent event) {
    stateTracker.actionCompletion(event);
    refreshSoon();
  }

  @Subscribe
  @AllowConcurrentEvents
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
   * Return true, if the test summary provides information that is both worth being shown in the
   * scroll-back buffer and new with respect to the alreay shown failure messages.
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
        setEventKindColor(
            summary.getStatus() == BlazeTestStatus.FLAKY ? EventKind.WARNING : EventKind.ERROR);
        terminal.writeString("" + summary.getStatus() + ": ");
        terminal.resetTerminal();
        terminal.writeString(summary.getLabel().toString());
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
        logger.atWarning().withCause(e).log("IO Error writing to output stream");
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
  public void buildEventTransportClosed(BuildEventTransportClosedEvent event) {
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
    if (!buildRunning) {
      return;
    }
    long nowMillis = clock.currentTimeMillis();
    if (lastRefreshMillis + minimalDelayMillis < nowMillis) {
      if (updateLock.tryLock()) {
        try {
          synchronized (this) {
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
          }
        } catch (IOException e) {
          logger.atWarning().withCause(e).log("IO Error writing to output stream");
        } finally {
          updateLock.unlock();
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
    if (mustRefreshAfterMillis <= lastRefreshMillis) {
      mustRefreshAfterMillis = Math.max(nowMillis + minimalUpdateInterval, lastRefreshMillis + 1);
    }
    startUpdateThread();
  }

  /** Decide whether the progress bar should be redrawn only for the reason that time has passed. */
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
    // Refuse to start an update thread once the build is complete; such a situation might
    // arise if the completion of the build is reported (shortly) before the completion of
    // the last action is reported.
    if (buildRunning && updateThread.get() == null) {
      final UiEventHandler eventHandler = this;
      Thread threadToStart =
          new Thread(
              () -> {
                try {
                  while (!shutdown) {
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
              },
              "cli-update-thread");
      if (updateThread.compareAndSet(null, threadToStart)) {
        threadToStart.start();
      }
    }
  }

  /**
   * Stop the update thread and wait for it to terminate. As the update thread, which is a separate
   * thread, might have to call a synchronized method between being interrupted and terminating, DO
   * NOT CALL from a SYNCHRONIZED block, as this will give the opportunity for dead locks.
   */
  private void stopUpdateThread() {
    shutdown = true;
    Thread threadToWaitFor = updateThread.getAndSet(null);
    if (threadToWaitFor != null) {
      threadToWaitFor.interrupt();
      Uninterruptibles.joinUninterruptibly(threadToWaitFor);
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

  /** Terminate the line in the way appropriate for the operating system. */
  private void crlf() throws IOException {
    terminal.writeString(System.lineSeparator());
  }

  private synchronized void addProgressBar() throws IOException {
    if (remainingCapacity() < 0) {
      return;
    }
    LineCountingAnsiTerminalWriter countingTerminalWriter =
        new LineCountingAnsiTerminalWriter(terminal);
    AnsiTerminalWriter terminalWriter = countingTerminalWriter;
    lastRefreshMillis = clock.currentTimeMillis();
    if (cursorControl) {
      terminalWriter = new LineWrappingAnsiTerminalWriter(terminalWriter, terminalWidth - 1);
    }
    String timestamp = null;
    if (showTimestamp) {
      timestamp =
          TIMESTAMP_FORMAT.format(
              Instant.ofEpochMilli(clock.currentTimeMillis()).atZone(ZoneId.systemDefault()));
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
