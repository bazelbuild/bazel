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

package com.google.devtools.build.lib.actions;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;

import java.util.Set;

import javax.annotation.Nullable;

/**
 * The DependentAction class is used to record which Actions depend on which
 * input Artifacts.
 *
 * <p>Each instance records the dependencies of a single Action.  We also have a
 * special DependentAction instance which records the dependencies of the
 * top-level build request.
 *
 * <p>The Action Graph (represented by the Action and Artifact classes) keeps
 * pointers from outputs to actions and actions to inputs, which are
 * respectively accessible via {@link ActionGraph#getGeneratingAction(Artifact)} and
 * {@link Action#getInputs()}. But to do parallel builds, we also use the
 * inverse mapping from Artifacts to the Actions that depend on them.
 * This is done so that when we notice that an Artifact has been built, we can
 * add all the Actions that depend on that Artifact to the work queue.
 *
 * <p>Please note that this class and all related data structures assume that
 * updates will be done only by the main builder's thread
 * and not by individual worker threads.
 */
@ThreadCompatible
public final class DependentAction {
  /**
   * Number of unbuilt input files that we're waiting for.
   */
  private int unbuiltInputs;

  /**
   * Original size of unbuilt inputs.
   */
  private final int originalSize;

  /**
   * An input artifact that we couldn't build.
   * If non-null, this means that an error was encountered
   * when building one or more of the inputs,
   * and so we can't attempt to build this action.
   */
  private Artifact cannotBuildInput = null;

  /**
   * The action that will be executed when all of the
   * input files are available. Null for the top-level build request.
   */
  @Nullable private final Action action;

  /**
   * Whether this was a root (top-level) action requested to be executed.
   */
  private final boolean rootAction;

  /**
   * Construct a new DependentAction, with all inputs initially marked as not
   * yet built. If the action has no inputs, add it to the work queue.
   *
   * <p>To create an action, use the factory methods
   * {@link #createDependency(Action, boolean)} and
   * {@link #createTopLevelDependency(Set)}.
   */
  private DependentAction(@Nullable Action action, boolean rootAction, int unbuilt) {
    this.action = action;
    this.rootAction = rootAction;
    this.unbuiltInputs = unbuilt;
    this.originalSize = unbuiltInputs;
  }

  private DependentAction(Action action, boolean toplevelAction) {
    this(action, toplevelAction, action.getInputCount());
  }

  /**
   * Mark all inputs unbuilt and reset the {@link #getLastProblemInput() root cause artifact}.
   */
  void reset() {
    this.unbuiltInputs = isVirtualCompletionAction() ? originalSize : action.getInputCount();
    cannotBuildInput = null;
  }

  /**
   * Construct a new non-virtual DependentAction.
   */
  static DependentAction createDependency(Action action, boolean rootAction) {
    Preconditions.checkNotNull(action);
    return new DependentAction(action, rootAction);
  }

  /**
   * Construct a new DependentAction for the virtual top-level build request.
   */
  static DependentAction createTopLevelDependency(Set<Artifact> topLevelTargets) {
    Preconditions.checkNotNull(topLevelTargets);
    return new DependentAction(null, false, topLevelTargets.size());
  }

  /**
   * Returns the number of input artifacts we have yet to build.
   */
  public int getUnbuiltInputs() {
    return unbuiltInputs;
  }

  /**
   * Returns the last artifact which failed to build. Often checked for null.
   */
  public Artifact getLastProblemInput() {
    return cannotBuildInput;
  }

  /**
   * Returns whether or not this was a root action.
   */
  public boolean isRootAction() {
    return rootAction;
  }

  /**
   * Decrease the number of unbuilt inputs.
   *
   * <p>Call this method when an input artifact has been built. When this counter reaches zero, this
   * {@link DependentAction} can be scheduled for execution. See {@link DependentAction#getAction}.
   *
   * <p>Note: {@link DependentAction} does not have references to the input artifacts, it only knows
   * the amount of them. The caller should make sure this method is called only once per input.
   */
  public void builtOneInput() {
    Preconditions.checkState(unbuiltInputs > 0);
    unbuiltInputs--;
  }

  /**
   * Record the root cause of a failed build.
   *
   * <p>Attempting to set null is forbidden; however {@link #reset()} can reset the problem.
   *
   * @param problem the Artifact which prevented the build from succeeding.
   */
  public void recordProblem(Artifact problem) {
    cannotBuildInput = Preconditions.checkNotNull(problem);
  }

  public String prettyPrint() {
    return isVirtualCompletionAction() ? "top-level virtual build request" : action.prettyPrint();
  }

  /**
   * Retrieves the {@link Action} associated with this dependency node.
   */
  public Action getAction() {
    return action;
  }

  /**
   * Returns true if and only if this DependentAction is the virtual top-level node.
   */
  public boolean isVirtualCompletionAction() {
    return action == null;
  }

  @Override
  public String toString() {
    return "<" + prettyPrint() + " waiting on " + unbuiltInputs + " inputs>";
  }
}
