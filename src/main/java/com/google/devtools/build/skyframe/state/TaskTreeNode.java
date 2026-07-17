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

import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.Lookup.ConsumerLookup;
import com.google.devtools.build.skyframe.state.Lookup.ValueOrException2Lookup;
import com.google.devtools.build.skyframe.state.Lookup.ValueOrException3Lookup;
import com.google.devtools.build.skyframe.state.Lookup.ValueOrExceptionLookup;
import java.util.ArrayList;
import java.util.function.Consumer;
import javax.annotation.Nullable;

/**
 * Represents the thread that runs a {@link StateMachine}.
 *
 * <p>Concurrency in the {@link StateMachine} is organized as a tree where the root is the only node
 * with a {@code null} parent.
 */
final class TaskTreeNode implements StateMachine.Tasks {
  private final Driver driver;
  @Nullable // Null for the root state machine.
  private final TaskTreeNode parent;
  private StateMachine state;
  private int pendingChildCount = 0;

  TaskTreeNode(Driver driver, @Nullable TaskTreeNode parent, StateMachine state) {
    this.driver = driver;
    this.parent = parent;
    this.state = state;
  }

  @Override
  public void enqueue(StateMachine subtask) {
    ++pendingChildCount;
    driver.addReady(new TaskTreeNode(driver, this, subtask));
  }

  @Override
  public void lookUp(SkyKey key, Consumer<SkyValue> sink) {
    ++pendingChildCount;
    driver.addLookup(new ConsumerLookup(this, key, sink));
  }

  @Override
  public <E extends Exception> void lookUp(
      SkyKey key, Class<E> exceptionClass, StateMachine.ValueOrExceptionSink<E> sink) {
    ++pendingChildCount;
    driver.addLookup(new ValueOrExceptionLookup<>(this, key, exceptionClass, sink));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception> void lookUp(
      SkyKey key,
      Class<E1> exceptionClass1,
      Class<E2> exceptionClass2,
      StateMachine.ValueOrException2Sink<E1, E2> sink) {
    ++pendingChildCount;
    driver.addLookup(
        new ValueOrException2Lookup<>(this, key, exceptionClass1, exceptionClass2, sink));
  }

  @Override
  public <E1 extends Exception, E2 extends Exception, E3 extends Exception> void lookUp(
      SkyKey key,
      Class<E1> exceptionClass1,
      Class<E2> exceptionClass2,
      Class<E3> exceptionClass3,
      StateMachine.ValueOrException3Sink<E1, E2, E3> sink) {
    ++pendingChildCount;
    driver.addLookup(
        new ValueOrException3Lookup<>(
            this, key, exceptionClass1, exceptionClass2, exceptionClass3, sink));
  }

  /** Runs the state machine bound to this node. */
  void run() throws InterruptedException {
    checkState(pendingChildCount == 0);
    while (state != StateMachine.DONE) {
      state = state.step(this);
      if (pendingChildCount > 0) {
        return;
      }
    }
    if (parent != null) {
      parent.signalChildDoneAndEnqueueIfReady();
    }
  }

  /**
   * Signals that a previously requested child is done.
   *
   * <p>Enqueues this node if all children are done.
   */
  void signalChildDoneAndEnqueueIfReady() {
    if (--pendingChildCount == 0) {
      driver.addReady(this);
    }
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
