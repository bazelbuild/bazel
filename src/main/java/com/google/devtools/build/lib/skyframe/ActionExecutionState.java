// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.skyframe.SkyFunction;
import javax.annotation.Nullable;
import javax.annotation.concurrent.GuardedBy;

/**
 * A state machine representing the synchronous or asynchronous execution of an action. This is
 * shared between all instances of the same shared action and must therefore be thread-safe. Note
 * that only one caller will receive events and output for this action.
 */
final class ActionExecutionState {
  /** The owner of this object. Only the owner is allowed to continue work on the state machine. */
  private final ActionLookupData actionLookupData;

  // Both state and completionFuture may only be read or set when holding the lock for this. The
  // state machine for these looks like this:
  //
  // !state.isDone,completionFuture=null -----> !state.isDone,completionFuture=<value>
  //                           |                  |
  //                           |                  | completionFuture.set()
  //                           v                  v
  //                    state.isDone,completionFuture=null
  //
  // No other states are legal. In particular, state.isDone,completionFuture=<value> is not a legal
  // state.

  @GuardedBy("this")
  private ActionStepOrResult state;

  /**
   * A future to represent action completion of the primary action (randomly picked from the set of
   * shared actions). This is initially {@code null}, and is only set to a non-null value if this
   * turns out to be a shared action, and the primary action is not finished yet (i.e., {@code
   * !state.isDone}. It is non-null while the primary action is being executed, at which point the
   * thread completing the primary action completes the future, and also sets this field to null.
   *
   * <p>The reason for this roundabout approach is to avoid memory allocation if this is not a
   * shared action, and to release the memory once the action is done.
   *
   * <p>Skyframe will attempt to cancel this future if the evaluation is interrupted, which violates
   * the concurrency assumptions this class makes. Beware!
   */
  @GuardedBy("this")
  @Nullable
  private SettableFuture<Void> completionFuture;

  ActionExecutionState(ActionLookupData actionLookupData, ActionStepOrResult state) {
    this.actionLookupData = Preconditions.checkNotNull(actionLookupData);
    this.state = Preconditions.checkNotNull(state);
  }

  ActionExecutionValue getResultOrDependOnFuture(
      SkyFunction.Environment env,
      ActionLookupData actionLookupData,
      Action action,
      SharedActionCallback sharedActionCallback)
      throws ActionExecutionException, InterruptedException {
    if (this.actionLookupData.equals(actionLookupData)) {
      // This continuation is owned by the Skyframe node executed by the current thread, so we use
      // it to run the state machine.
      return runStateMachine(env);
    }

    // This is a shared action, and the executed action is owned by another Skyframe node. If the
    // other node is done, we simply return the done value. Otherwise we register a dependency on
    // the completionFuture and return null.
    ActionExecutionValue result;
    synchronized (this) {
      if (!state.isDone()) {
        if (completionFuture == null) {
          completionFuture = SettableFuture.create();
        }
        // We expect to only call this once per shared action; this method should only be called
        // again after the future is completed.
        sharedActionCallback.actionStarted();
        env.dependOnFuture(completionFuture);
        if (!env.valuesMissing()) {
          Preconditions.checkState(
              completionFuture.isCancelled(), "%s %s", this.actionLookupData, actionLookupData);
          // The future is unexpectedly done. This must be because it was registered by another
          // thread earlier and was canceled by Skyframe. We are about to be interrupted ourselves,
          // but have to do something in the meantime. We can just register a dep with a new future,
          // then complete it and return. If for some reason this argument is incorrect, we will be
          // restarted immediately and hopefully have a more consistent result.
          SettableFuture<Void> dummyFuture = SettableFuture.create();
          env.dependOnFuture(dummyFuture);
          dummyFuture.set(null);
          return null;
        }
        // No other thread can modify completionFuture until we exit the synchronized block.
        Preconditions.checkState(
            !completionFuture.isDone(),
            "Completion future modified? %s %s %s %s",
            this.actionLookupData,
            actionLookupData,
            action,
            completionFuture);
        return null;
      }
      result = state.get();
    }
    sharedActionCallback.actionCompleted();
    return result.transformForSharedAction(action.getOutputs());
  }

  private ActionExecutionValue runStateMachine(SkyFunction.Environment env)
      throws ActionExecutionException, InterruptedException {
    ActionStepOrResult original;
    synchronized (this) {
      original = state;
    }
    ActionStepOrResult current = original;
    // We do the work _outside_ a synchronized block to avoid blocking threads working on shared
    // actions that only want to register with the completionFuture.
    try {
      while (!current.isDone()) {
        // Run the state machine for one step; isDone returned false, so this is safe.
        current = current.run(env);

        // This method guarantees that it either blocks until the action is completed and returns
        // a non-null value, or it registers a dependency with Skyframe and returns null; it must
        // not return null without registering a dependency, i.e., if {@code !env.valuesMissing()}.
        if (env.valuesMissing()) {
          return null;
        }
      }
    } finally {
      synchronized (this) {
        Preconditions.checkState(state == original, "Another thread modified state");
        state = current;
        if (current.isDone() && completionFuture != null) {
          completionFuture.set(null);
          completionFuture = null;
        }
      }
    }
    // We're done, return the value to the caller (or throw an exception).
    return current.get();
  }

  /** A callback to receive events for shared actions that are not executed. */
  public interface SharedActionCallback {
    /** Called if the action is shared and the primary action is already executing. */
    void actionStarted();

    /**
     * Called when the primary action is done (on the next call to {@link
     * #getResultOrDependOnFuture}.
     */
    void actionCompleted();
  }

  /**
   * A state machine where instances of this interface either represent an intermediate state that
   * requires more work to be done (possibly waiting for a ListenableFuture to complete) or the
   * final result of the executed action (either an ActionExecutionValue or an Exception).
   *
   * <p>This design allows us to store the current state of the in-progress action execution using a
   * single object reference.
   *
   * <p>Do not implement this interface directly! In order to implement an action step, subclass
   * {@link ActionStep}, and implement {@link #run}. In order to represent a result, use {@link
   * #of}.
   */
  interface ActionStepOrResult {
    static ActionStepOrResult of(ActionExecutionValue value) {
      return new Finished(value);
    }

    static ActionStepOrResult of(ActionExecutionException exception) {
      return new Exceptional(exception);
    }

    static ActionStepOrResult of(InterruptedException exception) {
      return new Exceptional(exception);
    }

    /**
     * Returns true if and only if the underlying action is complete, i.e., it is legal to call
     * {@link #get}. The return value of a single object must not change over time. Instead, call
     * {@link ActionStepOrResult#of} to return a final result (or exception).
     */
    boolean isDone();

    /**
     * Returns the next state of the state machine after performing some work towards the end goal
     * of executing the action. This must only be called if {@link #isDone} returns false, and must
     * only be called by one thread at a time for the same instance.
     */
    ActionStepOrResult run(SkyFunction.Environment env) throws InterruptedException;

    /**
     * Returns the final value of the action or an exception to indicate that the action failed (or
     * the process was interrupted). This must only be called if {@link #isDone} returns true.
     */
    ActionExecutionValue get() throws ActionExecutionException, InterruptedException;
  }

  /**
   * Abstract implementation of {@link ActionStepOrResult} that declares final implementations for
   * {@link #isDone} (to return false) and {@link #get} (tho throw {@link IllegalStateException}).
   *
   * <p>The framework prevents concurrent calls to {@link #run}, so implementations can keep state
   * without having to lock. Note that there may be multiple calls to {@link #run} from different
   * threads, as long as they do not overlap in time.
   */
  abstract static class ActionStep implements ActionStepOrResult {
    @Override
    public final boolean isDone() {
      return false;
    }

    @Override
    public final ActionExecutionValue get() {
      throw new IllegalStateException();
    }
  }

  /**
   * Represents a finished action with a specific value. We specifically avoid anonymous inner
   * classes to not accidentally retain a reference to the ActionRunner.
   */
  private static final class Finished implements ActionStepOrResult {
    private final ActionExecutionValue value;

    Finished(ActionExecutionValue value) {
      this.value = value;
    }

    @Override
    public boolean isDone() {
      return true;
    }

    @Override
    public ActionStepOrResult run(SkyFunction.Environment env) {
      throw new IllegalStateException();
    }

    @Override
    public ActionExecutionValue get() {
      return value;
    }
  }

  /**
   * Represents a finished action with an exception. We specifically avoid anonymous inner classes
   * to not accidentally retain a reference to the ActionRunner.
   */
  private static final class Exceptional implements ActionStepOrResult {
    private final Exception e;

    Exceptional(ActionExecutionException e) {
      this.e = e;
    }

    Exceptional(InterruptedException e) {
      this.e = e;
    }

    @Override
    public boolean isDone() {
      return true;
    }

    @Override
    public ActionStepOrResult run(SkyFunction.Environment env) {
      throw new IllegalStateException();
    }

    @Override
    public ActionExecutionValue get() throws ActionExecutionException, InterruptedException {
      if (e instanceof InterruptedException) {
        throw (InterruptedException) e;
      }
      throw (ActionExecutionException) e;
    }
  }
}
