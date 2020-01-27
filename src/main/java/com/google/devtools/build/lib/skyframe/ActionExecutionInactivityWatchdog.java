// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * An object that can monitor whether actions are getting completed in a timely manner.
 *
 * <p>If there's nothing happening for a while, a background thread will print (and update) the
 * "Still waiting for N actions to complete..." message.
 */
public final class ActionExecutionInactivityWatchdog {

  /** An object used in monitoring action execution inactivity. */
  public interface InactivityMonitor {

    /** Returns whether action execution has started. */
    boolean hasStarted();

    /** Returns the number of enqueued but not yet completed actions. */
    int getPending();

    /**
     * Waits for any action to complete, or the timeout to elapse.
     *
     * <p>The thread must wait at least for the specified timeout, unless some action completes in
     * the meantime. It's not allowed to return 0 too early.
     *
     * <p>Note that it's acceptable to return (any value) later than specified by the timeout.
     *
     * @return the number of actions completed during the wait
     */
    int waitForNextCompletion(int timeoutSeconds) throws InterruptedException;
  }

  /** An object that the watchdog can report inactivity to. */
  public interface InactivityReporter {

    /**
     * Report that actions are not getting completed in a timely manner.
     *
     * <p>Inactivity is typically not reported if tests with streaming output are being run.
     */
    void maybeReportInactivity();
  }

  @VisibleForTesting
  interface Sleep {
    void sleep(int durationMilliseconds) throws InterruptedException;
  }

  private static final class WaitTime {
    private final int progressIntervalFlagValue;
    private int prev;

    public WaitTime(int progressIntervalFlagValue) {
      this.progressIntervalFlagValue = progressIntervalFlagValue;
    }

    public void reset() {
      prev = 0;
    }

    public int next() {
      prev = ActionExecutionStatusReporter.getWaitTime(progressIntervalFlagValue, prev);
      return prev;
    }
  }

  private final AtomicBoolean isRunning = new AtomicBoolean(false);
  private final InactivityMonitor monitor;
  private final InactivityReporter reporter;
  private final Sleep sleeper;
  private final Thread thread;
  private final WaitTime waitTime;

  public ActionExecutionInactivityWatchdog(InactivityMonitor monitor, InactivityReporter reporter,
      int progressIntervalFlagValue) {
    this(monitor, reporter, progressIntervalFlagValue, new Sleep() {
      @Override
      public void sleep(int durationMilliseconds) throws InterruptedException {
        Thread.sleep(durationMilliseconds);
      }
    });
  }

  @VisibleForTesting
  public ActionExecutionInactivityWatchdog(InactivityMonitor monitor, InactivityReporter reporter,
      int progressIntervalFlagValue, Sleep sleeper) {
    this.monitor = Preconditions.checkNotNull(monitor);
    this.reporter = Preconditions.checkNotNull(reporter);
    this.sleeper = Preconditions.checkNotNull(sleeper);
    this.waitTime = new WaitTime(progressIntervalFlagValue);
    this.thread = new Thread(() -> enterWatchdogLoop(), "action-execution-watchdog");
    this.thread.setDaemon(true);
  }

  /** Starts the watchdog thread. This method should only be called once. */
  public void start() {
    Preconditions.checkState(!isRunning.getAndSet(true));
    thread.start();
  }

  /**
   * Stops the watchdog thread. This method should only be called once.
   *
   * <p>The method waits for the thread to terminate. If the caller thread is interrupted
   * in the meantime, the interrupted status will be set.
   */
  public void stop() {
    Preconditions.checkState(isRunning.getAndSet(false));
    thread.interrupt();
    try {
      thread.join();
    } catch (InterruptedException e) {
      // When Thread.join throws, the interrupted status is cleared. We need to set it again.
      Thread.currentThread().interrupt();
    }
  }

  private void enterWatchdogLoop() {
    while (isRunning.get()) {
      try {
        // Wait a while for any SkyFunction to finish. The returned number indicates how many
        // actions completed during the wait. It's possible that this is more than 1, since
        // this thread may not immediately regain control.
        int completedActions = monitor.waitForNextCompletion(waitTime.next());
        if (!isRunning.get()) {
          break;
        }

        int pending = monitor.getPending();
        if (!monitor.hasStarted() || completedActions > 0 || pending == 0) {
          // If no keys have been enqueued yet (execution hasn't started), or some actions
          // were completed since this thread was notified (we are making visible progress),
          // or there are currently no enqueued actions waiting to be processed (perhaps all
          // have completed and we are about to stop monitoring), then there's no need to
          // display any messages.
          waitTime.reset();

          // Sleep a while before checking again. Actions might be executing at a nice rate, no
          // need to worry about inactivity. This extra sleep isn't required but it's nice to
          // have: without it we would, at times of high action completion rate, unnecessarily
          // put the monitor into a fast sleep-wake cycle --- not a big problem but wasteful.
          sleeper.sleep(1000);
        } else {
          // If actions are executing but we haven't made any progress in a while (no new
          // action completion), then reassure the user that we're still running. Next time
          // wait a little longer.
          reporter.maybeReportInactivity();
        }
      } catch (InterruptedException ie) {
        Thread.currentThread().interrupt();
        return;
      }
    }
  }
}
