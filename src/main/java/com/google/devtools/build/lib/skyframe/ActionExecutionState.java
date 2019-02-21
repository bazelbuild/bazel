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

import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.skyframe.SkyframeActionExecutor.ActionCompletedReceiver;
import com.google.devtools.build.skyframe.SkyFunction;
import javax.annotation.concurrent.GuardedBy;

/**
 * A state machine representing the synchronous or asynchronous execution of an action. This is
 * shared between all instances of the same shared action and must therefore be thread-safe. Note
 * that only one caller will receive events and output for this action.
 */
final class ActionExecutionState {
  private final ActionLookupData actionLookupData;

  @GuardedBy("this")
  private ActionStepOrResult state;

  ActionExecutionState(ActionLookupData actionLookupData, ActionStepOrResult state) {
    this.actionLookupData = actionLookupData;
    this.state = state;
  }

  public ActionLookupData getActionLookupData() {
    return actionLookupData;
  }

  public ActionExecutionValue getResultOrDependOnFuture(
      SkyFunction.Environment env,
      ActionLookupData actionLookupData,
      Action action,
      ActionCompletedReceiver actionCompletedReceiver)
      throws ActionExecutionException, InterruptedException {
    if (actionLookupData.equals(this.actionLookupData)) {
      // This continuation is owned by the Skyframe node executed by the current thread, so we use
      // it to run the state machine.
      return runStateMachine(env);
    }
    // This is a shared action, and the executed action is owned by another Skyframe node. We do
    // not attempt to make progress, but instead block waiting for the owner to complete the
    // action. This is the same behavior as before this comment was added.
    //
    // When we async action execution we MUST also change this to async execution. Otherwise we
    // can end up with a deadlock where all Skyframe threads are blocked here, and no thread is
    // available to make progress on the original action.
    synchronized (this) {
      while (!state.isDone()) {
        this.wait();
      }
      try {
        return state.get().transformForSharedAction(action.getOutputs());
      } finally {
        if (action.getProgressMessage() != null) {
          actionCompletedReceiver.actionCompleted(actionLookupData);
        }
      }
    }
  }

  private synchronized ActionExecutionValue runStateMachine(SkyFunction.Environment env)
      throws ActionExecutionException, InterruptedException {
    while (!state.isDone()) {
      // Run the state machine for one step; isDone returned false, so this is safe.
      state = state.run(env);

      // This method guarantees that it either blocks until the action is completed, or it
      // registers a dependency on a ListenableFuture and returns null (it may only return null if
      // valuesMissing returns true).
      if (env.valuesMissing()) {
        return null;
      }
    }
    this.notifyAll();
    // We're done, return the value to the caller (or throw an exception).
    return state.get();
  }

  /**
   * A state machine where instances of this interface either represent an intermediate state that
   * requires more work to be done (possibly waiting for a ListenableFuture to complete) or the
   * final result of the executed action (either an ActionExecutionValue or an Exception).
   *
   * <p>This design allows us to store the current state of the in-progress action execution using a
   * single object reference.
   */
  interface ActionStepOrResult {
    static ActionStepOrResult of(ActionExecutionValue value) {
      return new FinishedActionStepOrResult(value);
    }

    static ActionStepOrResult of(ActionExecutionException exception) {
      return new ExceptionalActionStepOrResult(exception);
    }

    static ActionStepOrResult of(InterruptedException exception) {
      return new ExceptionalActionStepOrResult(exception);
    }

    /**
     * Returns true if and only if the underlying action is complete, i.e., it is legal to call
     * {@link #get}.
     */
    default boolean isDone() {
      return true;
    }

    /**
     * Returns the next state of the state machine after performing some work towards the end goal
     * of executing the action. This must only be called if {@link #isDone} returns false, and must
     * only be called by one thread at a time for the same instance.
     */
    default ActionStepOrResult run(SkyFunction.Environment env) {
      throw new IllegalStateException();
    }

    /**
     * Returns the final value of the action or an exception to indicate that the action failed (or
     * the process was interrupted). This must only be called if {@link #isDone} returns true.
     */
    default ActionExecutionValue get() throws ActionExecutionException, InterruptedException {
      throw new IllegalStateException();
    }
  }

  /**
   * Represents a finished action with a specific value. We specifically avoid anonymous inner
   * classes to not accidentally retain a reference to the ActionRunner.
   */
  private static final class FinishedActionStepOrResult implements ActionStepOrResult {
    private final ActionExecutionValue value;

    FinishedActionStepOrResult(ActionExecutionValue value) {
      this.value = value;
    }

    public ActionExecutionValue get() {
      return value;
    }
  }

  /**
   * Represents a finished action with an exception. We specifically avoid anonymous inner classes
   * to not accidentally retain a reference to the ActionRunner.
   */
  private static final class ExceptionalActionStepOrResult implements ActionStepOrResult {
    private final Exception e;

    ExceptionalActionStepOrResult(ActionExecutionException e) {
      this.e = e;
    }

    ExceptionalActionStepOrResult(InterruptedException e) {
      this.e = e;
    }

    public ActionExecutionValue get() throws ActionExecutionException, InterruptedException {
      if (e instanceof InterruptedException) {
        throw (InterruptedException) e;
      }
      throw (ActionExecutionException) e;
    }
  }
}
