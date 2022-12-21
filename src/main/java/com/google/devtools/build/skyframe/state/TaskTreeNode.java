// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe.state;

import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.events.ExtendedEventHandler;
import java.util.ArrayList;
import javax.annotation.Nullable;

/**
 * Represents the thread that runs a {@link StateMachine}.
 *
 * <p>Concurrency in the {@link StateMachine} is organized as a tree where the root is the only node
 * with a {@code null} parent.
 */
final class TaskTreeNode {
  @Nullable private final TaskTreeNode parent;
  private StateMachine state;
  private int pendingChildCount = 0;

  TaskTreeNode(@Nullable TaskTreeNode parent, StateMachine state) {
    this.parent = parent;
    this.state = state;
  }

  /**
   * Runs the machine.
   *
   * @return true if the machine and all its children are done.
   */
  boolean run(StateMachine.Tasks tasks, ExtendedEventHandler listener) throws InterruptedException {
    checkState(pendingChildCount == 0);
    while (state != null) {
      state = state.step(tasks, listener);
      if (pendingChildCount > 0) {
        return false;
      }
    }
    return true;
  }

  @Nullable // Null if this is the root.
  TaskTreeNode parent() {
    return parent;
  }

  void incrementChildCount() {
    ++pendingChildCount;
  }

  /**
   * Decrements the child count.
   *
   * @return true if the child count becomes 0, meaning this machine is ready for execution.
   */
  boolean decrementChildCount() {
    return --pendingChildCount <= 0;
  }

  @Override
  public String toString() {
    var stack = new ArrayList<TaskTreeNode>();
    for (var next = this; next != null; next = next.parent) {
      stack.add(next);
    }
    StringBuilder buf = new StringBuilder("TaskTreeNode[");
    boolean isFirst = true;
    // Traverses the stack backwards so the output is in root to leaf order.
    for (int i = stack.size() - 1; i >= 0; --i) {
      if (isFirst) {
        isFirst = false;
      } else {
        buf.append(",\n ");
      }
      buf.append(stack.get(i).state);
    }
    buf.append("]\n");
    return buf.toString();
  }
}
