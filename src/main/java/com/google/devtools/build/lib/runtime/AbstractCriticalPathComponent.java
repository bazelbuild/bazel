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

import javax.annotation.Nullable;

/**
 * This class records the critical path for the graph of actions executed.
 */
@ThreadCompatible
public class AbstractCriticalPathComponent<C extends AbstractCriticalPathComponent<C>> {

  /** Wall time start time for the action. In milliseconds. */
  private final long startTime;
  /** Wall time finish time for the action. In milliseconds. */
  private long finishTime = 0;
  protected volatile boolean isRunning = true;

  /** We keep here the critical path time for the most expensive child. */
  private long childAggregatedWallTime = 0;

  /** The action for which we are storing the stat. */
  private final Action action;

  /**
   * Child with the maximum critical path.
   */
  @Nullable
  private C child;

  public AbstractCriticalPathComponent(Action action, long startTime) {
    this.action = action;
    this.startTime = startTime;
  }

  /** Sets the finish time for the action in milliseconds. */
  public void setFinishTimeMillis(long finishTime) {
    Preconditions.checkState(isRunning, "Already stopped! %s.", action);
    this.finishTime = finishTime;
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
    long childAggregatedWallTime = dep.getAggregatedWallTime();
    // Replace the child if its critical path had the maximum wall time.
    if (child == null || childAggregatedWallTime > this.childAggregatedWallTime) {
      this.childAggregatedWallTime = childAggregatedWallTime;
      child = dep;
    }
  }

  public long getActionWallTime() {
    Preconditions.checkState(!isRunning, "Still running %s", action);
    return finishTime - startTime;
  }

  /**
   * Returns the current critical path for the action in milliseconds.
   *
   * <p>Critical path is defined as : action_execution_time + max(child_critical_path).
   */
  public long getAggregatedWallTime() {
    Preconditions.checkState(!isRunning, "Still running %s", action);
    return getActionWallTime() + childAggregatedWallTime;
  }

  /** Time when the action started to execute. Milliseconds since epoch time. */
  public long getStartTime() {
    return startTime;
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
      currentTime = String.format("%.2f", getActionWallTime() / 1000.0) + "s ";
    }
    return currentTime + action.describe();
  }
}

