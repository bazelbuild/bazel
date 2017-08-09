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

package com.google.devtools.build.lib.shell;

import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * {@link KillableObserver} implementation which will kill its observed {@link Killable} if it is
 * still being observed after a given amount of time has elapsed.
 *
 * <p>Note that this class can only observe one {@link Killable} at a time; multiple instances
 * should be used for concurrent calls to {@link Command#execute}.
 */
final class TimeoutKillableObserver implements KillableObserver {

  private static final Logger log =
      Logger.getLogger(TimeoutKillableObserver.class.getCanonicalName());

  private final long timeoutMS;
  private Killable killable;
  private SleeperThread sleeperThread;
  private boolean timedOut;

  // TODO(bazel-team): I'd like to use ThreadPool2, but it doesn't currently
  // provide a way to interrupt a thread

  public TimeoutKillableObserver(final long timeoutMS) {
    this.timeoutMS = timeoutMS;
  }

  /**
   * Starts a new {@link Thread} to wait for the timeout period. This is
   * interrupted by the {@link #stopObserving(Killable)} method.
   *
   * @param killable killable to kill when the timeout period expires
   */
  @Override
  public synchronized void startObserving(final Killable killable) {
    this.timedOut = false;
    this.killable = killable;
    this.sleeperThread = new SleeperThread();
    this.sleeperThread.start();
  }

  @Override
  public synchronized void stopObserving(final Killable killable) {
    if (!this.killable.equals(killable)) {
      throw new IllegalStateException("start/stopObservering called with " +
                                      "different Killables");
    }
    if (sleeperThread.isAlive()) {
      sleeperThread.interrupt();
    }
    this.killable = null;
    sleeperThread = null;
  }

  private final class SleeperThread extends Thread {
    @Override public void run() {
      try {
        if (log.isLoggable(Level.FINE)) {
          log.fine("Waiting for " + timeoutMS + "ms to kill process");
        }
        Thread.sleep(timeoutMS);
        // timeout expired; kill it
        synchronized (TimeoutKillableObserver.this) {
          if (killable != null) {
            log.fine("Killing process");
            killable.kill();
            timedOut = true;
          }
        }
      } catch (InterruptedException ie) {
        // continue -- process finished before timeout
        log.fine("Wait interrupted since process finished; continuing...");
      }
    }
  }

  /**
   * Returns true if the observed process was killed by this observer.
   */
  public synchronized boolean hasTimedOut() {
    // synchronized needed for memory model visibility.
    return timedOut;
  }
}
