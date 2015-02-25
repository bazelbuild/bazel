// Copyright 2014 Google Inc. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.util.Clock;

import java.util.concurrent.TimeUnit;

import javax.annotation.Nullable;

/**
 * This class records the critical path for the graph of actions executed.
 */
@ThreadCompatible
public class AbstractCriticalPathComponent<C extends AbstractCriticalPathComponent<C>> {

  /** Start time in nanoseconds. Only to be used for measuring elapsed time. */
  private final long relativeStartNanos;
  /** Finish time for the action in nanoseconds. Only to be used for measuring elapsed time. */
  private long relativeFinishNanos = 0;
  protected volatile boolean isRunning = true;

  /** We keep here the critical path time for the most expensive child. */
  private long childAggregatedElapsedTime = 0;

  /** The action for which we are storing the stat. */
  private final Action action;

  /**
   * Child with the maximum critical path.
   */
  @Nullable
  private C child;

  public AbstractCriticalPathComponent(Action action, long relativeStartNanos) {
    this.action = action;
    this.relativeStartNanos = relativeStartNanos;
  }

  /**
   * Sets the finish time for the action in nanoseconds for computing the duration of the action.
   */
  public void setRelativeStartNanos(long relativeFinishNanos) {
    Preconditions.checkState(isRunning, "Already stopped! %s.", action);
    this.relativeFinishNanos = relativeFinishNanos;
    isRunning = false;
  }

  /** The action for which we are storing the stat. */
  public Action getAction() {
    return action;
  }

  /**
   * Add statistics for one dependency of this action.
   */
  public void addDepInfo(C dep) {
    Preconditions.checkState(!dep.isRunning,
        "Cannot add critical path stats when the action is not finished. %s. %s", action,
        dep.getAction());
    long childAggregatedWallTime = dep.getAggregatedElapsedTimeNanos();
    // Replace the child if its critical path had the maximum elapsed time.
    if (child == null || childAggregatedWallTime > this.childAggregatedElapsedTime) {
      this.childAggregatedElapsedTime = childAggregatedWallTime;
      child = dep;
    }
  }

  public long getElapsedTimeMillis() {
    return TimeUnit.NANOSECONDS.toMillis(getElapsedTimeNanos());
  }

  long getElapsedTimeNanos() {
    Preconditions.checkState(!isRunning, "Still running %s", action);
    return relativeFinishNanos - relativeStartNanos;
  }

  /**
   * Returns the current critical path for the action in nanoseconds.
   *
   * <p>Critical path is defined as : action_execution_time + max(child_critical_path).
   */
  public long getAggregatedElapsedTimeMillis() {
    return TimeUnit.NANOSECONDS.toMillis(getAggregatedElapsedTimeNanos());
  }

  long getAggregatedElapsedTimeNanos() {
    Preconditions.checkState(!isRunning, "Still running %s", action);
    return getElapsedTimeNanos() + childAggregatedElapsedTime;
  }

  /**
   * Get the child critical path component.
   *
   * <p>The component dependency with the maximum total critical path time.
   */
  @Nullable
  public C getChild() {
    return child;
  }

  /**
   * Returns a human readable representation of the critical path stats with all the details.
   */
  @Override
  public String toString() {
    String currentTime = "still running ";
    if (!isRunning) {
      currentTime = String.format("%.2f", getElapsedTimeMillis() / 1000.0) + "s ";
    }
    return currentTime + action.describe();
  }

  /**
   * When {@code clock} is the same {@link Clock} that was used for computing
   * {@link #relativeStartNanos}, it returns the wall time since epoch representing when
   * the action was started.
   */
  public long getStartWallTimeMillis(Clock clock) {
    long millis = clock.currentTimeMillis();
    long nanoElapsed = clock.nanoTime();
    return millis - TimeUnit.NANOSECONDS.toMillis((nanoElapsed - relativeStartNanos));
  }
}

