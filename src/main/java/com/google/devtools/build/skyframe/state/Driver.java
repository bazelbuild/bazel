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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.collect.Lists;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.SkyframeLookupResult;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.function.Consumer;

/**
 * This class drives a {@link StateMachine} instance.
 *
 * <p>One recommended usage pattern for this class is to embed an instance within a top level {@link
 * StateMachine} implementation and from there, re-export the {@link #drive} method. Then the
 * results from the {@link StateMachine} will be readily retrievable from the {@link SkyFunction}
 * state.
 */
// TODO(shahan); this is incompatible with partial re-evaluation, which causes the assumption that
// an unavailable previously requested dependency implies an error to no longer be true. This can be
// fixed by integrating with the partial re-evaluation mailbox.
public final class Driver {
  // TODO(shahan): tune this parameter.
  private static final int DEFAULT_LIST_CAPACITY = 128;

  private final ArrayDeque<TaskTreeNode> ready = new ArrayDeque<>();

  /** A Skyframe lookup has not yet been made for the key. */
  private final ArrayList<Lookup> newlyAdded = new ArrayList<>(DEFAULT_LIST_CAPACITY);

  /** A Skyframe lookup has already been made for the key, but it was not available. */
  private final ArrayList<Lookup> pending = new ArrayList<>(DEFAULT_LIST_CAPACITY);

  public Driver(StateMachine root) {
    ready.addLast(new TaskTreeNode(/* parent= */ null, root));
  }

  /**
   * Drives the machine as far as it can go without a Skyframe restart.
   *
   * @return true if execution is complete, false if a restart is needed.
   */
  public boolean drive(Environment env, ExtendedEventHandler listener) throws InterruptedException {
    if (!pending.isEmpty()) {
      // If pending is non-empty, it means there was a Skyframe restart. Either everything that was
      // pending is available now or we are in error bubbling. In the latter case, an error will be
      // present and this code will fail fast.
      //
      // NB: this assumption does not hold under partial re-evaluation and likewise the inference
      // below about unavailable values being errors.
      SkyframeLookupResult result = env.getLookupHandleForPreviouslyRequestedDeps();
      boolean hasException = false;
      for (var lookup : pending) {
        if (!result.queryDep(lookup.key(), lookup)) {
          // Since the key was previously requested, unavailability here implies an unhandled or
          // unavailable exception. Requests the error to ensure that this environment instance
          // knows the failure is due to a child error.
          var unusedNull = env.getValue(lookup.key());
          // Failing fast here would make behavior dependent on element ordering, so instead, flags
          // the exception and fails after all elements have been processed.
          hasException = true;
        }
      }
      if (hasException) {
        return false;
      }
      pending.clear();
    }

    while (true) {
      // Executes as many ready tasks as possible.
      while (!ready.isEmpty()) {
        TaskTreeNode next = ready.removeFirst();
        if (next.run(new TasksImpl(next), listener)) {
          TaskTreeNode parent = next.parent();
          if (parent == null) {
            return true; // Root is complete.
          }
          signalParentAndEnqueueIfReady(parent);
        }
      }

      if (newlyAdded.isEmpty()) {
        return false; // No tasks are ready and all remaining lookups are pending.
      }

      // Performs lookups for any newly added keys.
      SkyframeLookupResult result =
          env.getValuesAndExceptions(Lists.transform(newlyAdded, Lookup::key));
      for (var lookup : newlyAdded) {
        if (!result.queryDep(lookup.key(), lookup)) {
          pending.add(lookup); // Unhandled exceptions also end up here.
        }
      }
      newlyAdded.clear(); // Every entry is either done or has moved to pending.
    }
  }

  private void signalParentAndEnqueueIfReady(TaskTreeNode parent) {
    if (parent.decrementChildCount()) {
      ready.addLast(parent);
    }
  }

  private final class TasksImpl implements StateMachine.Tasks {
    private final TaskTreeNode parent;

    private TasksImpl(TaskTreeNode parent) {
      this.parent = parent;
    }

    @Override
    public void enqueue(StateMachine subtask) {
      parent.incrementChildCount();
      ready.addLast(new TaskTreeNode(parent, subtask));
    }

    @Override
    public void lookUp(SkyKey key, Consumer<SkyValue> sink) {
      lookUp(key, sink, exception -> false);
    }

    @Override
    public <E extends Exception> void lookUp(
        SkyKey key, Class<E> exceptionClass, StateMachine.ValueOrExceptionSink<E> sink) {
      lookUp(
          key,
          value -> sink.accept(value, /* exception= */ null),
          e -> {
            if (exceptionClass.isInstance(e)) {
              sink.accept(/* value= */ null, exceptionClass.cast(e));
              return true;
            }
            return false;
          });
    }

    @Override
    public <E1 extends Exception, E2 extends Exception> void lookUp(
        SkyKey key,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        StateMachine.ValueOrException2Sink<E1, E2> sink) {
      lookUp(
          key,
          value -> sink.accept(value, /* e1= */ null, /* e2= */ null),
          e -> {
            if (exceptionClass1.isInstance(e)) {
              sink.accept(/* value= */ null, exceptionClass1.cast(e), /* e2= */ null);
              return true;
            }
            if (exceptionClass2.isInstance(e)) {
              sink.accept(/* value= */ null, /* e1= */ null, exceptionClass2.cast(e));
              return true;
            }
            return false;
          });
    }

    @Override
    public <E1 extends Exception, E2 extends Exception, E3 extends Exception> void lookUp(
        SkyKey key,
        Class<E1> exceptionClass1,
        Class<E2> exceptionClass2,
        Class<E3> exceptionClass3,
        StateMachine.ValueOrException3Sink<E1, E2, E3> sink) {
      lookUp(
          key,
          value -> sink.accept(value, /* e1= */ null, /* e2= */ null, /* e3= */ null),
          e -> {
            if (exceptionClass1.isInstance(e)) {
              sink.accept(
                  /* value= */ null, exceptionClass1.cast(e), /* e2= */ null, /* e3= */ null);
              return true;
            }
            if (exceptionClass2.isInstance(e)) {
              sink.accept(
                  /* value= */ null, /* e1= */ null, exceptionClass2.cast(e), /* e3= */ null);
              return true;
            }
            if (exceptionClass3.isInstance(e)) {
              sink.accept(
                  /* value= */ null, /* e1= */ null, /* e2= */ null, exceptionClass3.cast(e));
              return true;
            }
            return false;
          });
    }

    /**
     * Adds a dependency to look up.
     *
     * <p>Exactly one of {@code sink} or {@code exceptionHandler} will be called. The callback could
     * be deferred until the next Skyframe restart if the queried key is not immediately available.
     */
    private void lookUp(SkyKey key, Consumer<SkyValue> sink, ExceptionHandler exceptionHandler) {
      parent.incrementChildCount();
      newlyAdded.add(new Lookup(parent, checkNotNull(key), sink, exceptionHandler));
    }
  }

  private final class Lookup implements SkyframeLookupResult.QueryDepCallback {
    private final TaskTreeNode parent;
    private final SkyKey key;
    private final Consumer<SkyValue> sink;
    private final ExceptionHandler exceptionHandler;

    private Lookup(
        TaskTreeNode parent,
        SkyKey key,
        Consumer<SkyValue> sink,
        ExceptionHandler exceptionHandler) {
      this.parent = parent;
      this.key = key;
      this.sink = sink;
      this.exceptionHandler = exceptionHandler;
    }

    private SkyKey key() {
      return key;
    }

    @Override
    public void acceptValue(SkyKey unusedKey, SkyValue value) {
      sink.accept(value);
      signalParentAndEnqueueIfReady(parent);
    }

    @Override
    public boolean tryHandleException(SkyKey unusedKey, Exception exception) {
      if (exceptionHandler.tryHandleException(exception)) {
        signalParentAndEnqueueIfReady(parent);
        return true;
      }
      return false;
    }
  }

  /**
   * Handles exceptions.
   *
   * <p>Receiving an exception and not handling it results in it bubbling up at the Skyframe level.
   *
   * <p>An exception may be offered to this method multiple times. Changing the return value results
   * in undefined behavior.
   */
  @FunctionalInterface
  interface ExceptionHandler {
    /** Returns true if an exception was handled and false otherwise. */
    boolean tryHandleException(Exception e);
  }
}
