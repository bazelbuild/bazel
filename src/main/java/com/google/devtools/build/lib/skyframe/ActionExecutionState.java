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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.base.Preconditions;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.Artifact.OwnerlessArtifactWrapper;
import com.google.devtools.build.lib.actions.LostInputsActionExecutionException;
import com.google.devtools.build.lib.actions.SharedActionEvent;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.skyframe.ActionExecutionValue.ActionTransformException;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunction.Environment;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.errorprone.annotations.DoNotCall;
import java.util.concurrent.ConcurrentMap;
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
  // (Also, via obsolete(), all states can transition to state==Obsolete.INSTANCE with a null
  // completionFuture, which is terminal.)
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

  @Nullable
  ActionExecutionValue getResultOrDependOnFuture(
      SkyFunction.Environment env,
      ActionLookupData actionLookupData,
      Action action,
      SharedActionCallback sharedActionCallback)
      throws ActionExecutionException, InterruptedException {
    if (this.actionLookupData.equals(actionLookupData)) {
      // This object is owned by the Skyframe node executed by the current thread, so we use it to
      // run the state machine.
      return runStateMachine(env);
    }

    // This is a shared action, and the primary action is owned by another Skyframe node. If the
    // primary action is done, we simply return the done value. If this state is obsolete (e.g.
    // because the other node is rewinding), we restart. Otherwise we register a dependency on the
    // completionFuture and return null.
    ActionExecutionValue result;
    synchronized (this) {
      if (state == Obsolete.INSTANCE) {
        scheduleRestart(env);
        return null;
      }
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
          scheduleRestart(env);
        }
        return null;
      }
      result = state.get();
    }
    sharedActionCallback.actionCompleted();

    ActionExecutionValue transformed;
    try {
      transformed = result.transformForSharedAction(action);
    } catch (ActionTransformException e) {
      throw new IllegalStateException(
          String.format("Cannot share %s and %s", this.actionLookupData, actionLookupData), e);
    }
    env.getListener().post(new SharedActionEvent(result, transformed));
    return transformed;
  }

  private static void scheduleRestart(Environment env) {
    SettableFuture<Void> dummyFuture = SettableFuture.create();
    env.dependOnFuture(dummyFuture);
    dummyFuture.set(null);
  }

  @Nullable
  private ActionExecutionValue runStateMachine(SkyFunction.Environment env)
      throws ActionExecutionException, InterruptedException {
    ActionStepOrResult original;
    synchronized (this) {
      if (state == Obsolete.INSTANCE) {
        scheduleRestart(env);
        return null;
      }
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
          if (current.isDone()) {
            // This can happen if there was an error in a dep, but another dep was missing. The
            // Skyframe contract is that this SkyFunction should eagerly process that exception, so
            // that errors can be transformed in --nokeep_going mode.
            ActionExecutionValue value = current.get();
            BugReport.sendBugReport(
                new IllegalStateException(
                    actionLookupData + " returned " + value + " with values missing"));
          }
          return null;
        }
      }
    } finally {
      synchronized (this) {
        if (state != Obsolete.INSTANCE) {
          Preconditions.checkState(state == original, "Another thread illegally modified state");
          state = current;
          if (current.isDone() && completionFuture != null) {
            completionFuture.set(null);
            completionFuture = null;
          }
        }
      }
    }
    // We're done, return the value to the caller (or throw an exception).
    return current.get();
  }

  /**
   * Removes this state from {@code buildActionMap}, marks it obsolete so that racing shared actions
   * with a reference to this state will restart, and signals to coalesced shared actions that they
   * should re-evaluate.
   */
  synchronized void obsolete(
      SkyKey requester,
      ConcurrentMap<OwnerlessArtifactWrapper, ActionExecutionState> buildActionMap,
      OwnerlessArtifactWrapper ownerlessArtifactWrapper) {
    if (actionLookupData.equals(requester)) {
      // An action state's owner only obsoletes it when rewinding. The lost inputs exception thrown
      // from ActionStepOrResult#run left its state undone.
      Preconditions.checkState(
          !state.isDone(), "owner unexpectedly obsoleted done state: %s", actionLookupData);
      ActionExecutionState removedState = buildActionMap.remove(ownerlessArtifactWrapper);
      Preconditions.checkState(
          removedState == this,
          "owner removed unexpected state from buildActionMap; owner: %s, removed: %s",
          actionLookupData,
          removedState.actionLookupData);
      state = Obsolete.INSTANCE;
      if (completionFuture != null) {
        completionFuture.set(null);
        completionFuture = null;
      }
      return;
    }
    if (!state.isDone()) {
      // An action obsoletes other actions' states when rewinding its dependencies. It may race with
      // other actions to do so. Removing the buildActionMap entry must only be done by the race's
      // winner, to ensure the removal only happens once and removes this state.
      //
      // An action may also attempt to obsolete a dependency's not-done state, if it lost the race
      // with another rewinding action, and the dep started evaluating. If so, then do nothing,
      // because that dep is already doing what it needs to.
      return;
    }
    ActionExecutionState removedState = buildActionMap.remove(ownerlessArtifactWrapper);
    Preconditions.checkState(
        removedState == this,
        "removed unexpected state from buildActionMap; requester: %s, this: %s, removed: %s",
        requester,
        actionLookupData,
        removedState.actionLookupData);
    state = Obsolete.INSTANCE;
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
  sealed interface ActionStepOrResult permits ActionStep, Finished, Exceptional, Obsolete {
    static ActionStepOrResult of(ActionExecutionValue value) {
      return new Finished(value);
    }

    /**
     * Must not be called with a {@link LostInputsActionExecutionException}. Throw it from {@link
     * #run} instead.
     */
    static ActionStepOrResult of(ActionExecutionException exception) {
      checkArgument(
          !(exception instanceof LostInputsActionExecutionException),
          "unexpected LostInputs exception: %s",
          exception);
      return new Exceptional(exception);
    }

    @DoNotCall("Throw from #run instead.")
    static ActionStepOrResult of(LostInputsActionExecutionException ignored) {
      throw new IllegalArgumentException();
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
    ActionStepOrResult run(SkyFunction.Environment env)
        throws LostInputsActionExecutionException, InterruptedException;

    /**
     * Returns the final value of the action or an exception to indicate that the action failed (or
     * the process was interrupted). This must only be called if {@link #isDone} returns true.
     */
    ActionExecutionValue get() throws ActionExecutionException, InterruptedException;
  }

  /**
   * Abstract implementation of {@link ActionStepOrResult} that declares final implementations for
   * {@link #isDone} (to return false) and {@link #get} (to throw {@link IllegalStateException}).
   *
   * <p>The framework prevents concurrent calls to {@link #run}, so implementations can keep state
   * without having to lock. Note that there may be multiple calls to {@link #run} from different
   * threads, as long as they do not overlap in time.
   */
  abstract static non-sealed class ActionStep implements ActionStepOrResult {
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

  /**
   * Represents an action state that is obsolete. Any non-primary shared actions observing this
   * state must restart (see {@link #scheduleRestart}.
   */
  private static final class Obsolete implements ActionStepOrResult {
    private static final Obsolete INSTANCE = new Obsolete();

    @Override
    public boolean isDone() {
      return false;
    }

    @Override
    public ActionStepOrResult run(SkyFunction.Environment env) {
      throw new IllegalStateException();
    }

    @Override
    public ActionExecutionValue get() {
      throw new IllegalStateException();
    }
  }
}
