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
package com.google.devtools.build.lib.actions;

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.devtools.build.lib.causes.ActionFailed;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.skyframe.DetailedException;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/**
 * This exception gets thrown if {@link Action#execute(ActionExecutionContext)} is unsuccessful.
 * Typically these are re-raised ExecException throwables.
 */
@ThreadSafe
public class ActionExecutionException extends Exception implements DetailedException {

  private final ActionAnalysisMetadata action;
  private final NestedSet<Cause> rootCauses;
  private final boolean catastrophe;
  private final DetailedExitCode detailedExitCode;

  public ActionExecutionException(
      Throwable cause,
      ActionAnalysisMetadata action,
      boolean catastrophe,
      DetailedExitCode detailedExitCode) {
    super(cause.getMessage(), cause);
    this.action = action;
    this.detailedExitCode = detailedExitCode;
    this.rootCauses = rootCausesFromAction(action, detailedExitCode);
    this.catastrophe = catastrophe;
  }

  public ActionExecutionException(
      String message,
      Throwable cause,
      ActionAnalysisMetadata action,
      boolean catastrophe,
      DetailedExitCode detailedExitCode) {
    super(message, cause);
    this.action = action;
    this.catastrophe = catastrophe;
    this.detailedExitCode = checkNotNull(detailedExitCode);
    this.rootCauses = rootCausesFromAction(action, detailedExitCode);
  }

  public ActionExecutionException(
      String message,
      ActionAnalysisMetadata action,
      boolean catastrophe,
      DetailedExitCode detailedExitCode) {
    super(message);
    this.action = action;
    this.catastrophe = catastrophe;
    this.detailedExitCode = detailedExitCode;
    this.rootCauses = rootCausesFromAction(action, this.detailedExitCode);
  }

  public ActionExecutionException(
      String message,
      ActionAnalysisMetadata action,
      NestedSet<Cause> rootCauses,
      boolean catastrophe,
      DetailedExitCode detailedExitCode) {
    super(message);
    this.action = action;
    this.rootCauses = rootCauses;
    this.catastrophe = catastrophe;
    this.detailedExitCode = detailedExitCode;
  }

  public ActionExecutionException(
      String message,
      Throwable cause,
      ActionAnalysisMetadata action,
      NestedSet<Cause> rootCauses,
      boolean catastrophe,
      DetailedExitCode detailedExitCode) {
    super(message, cause);
    this.action = action;
    this.rootCauses = rootCauses;
    this.catastrophe = catastrophe;
    this.detailedExitCode = checkNotNull(detailedExitCode);
  }

  private static NestedSet<Cause> rootCausesFromAction(
      ActionAnalysisMetadata action, DetailedExitCode detailedExitCode) {
    return action == null || action.getOwner() == null || action.getOwner().getLabel() == null
        ? NestedSetBuilder.emptySet(Order.STABLE_ORDER)
        : NestedSetBuilder.create(
            Order.STABLE_ORDER,
            new ActionFailed(
                action.getPrimaryOutput().getExecPath(),
                action.getOwner().getLabel(),
                action.getOwner().getConfigurationChecksum(),
                detailedExitCode));
  }

  public static ActionExecutionException fromExecException(ExecException exception, Action action) {
    return fromExecException(exception, null, action);
  }

  /**
   * Returns a new ActionExecutionException given an optional action subtask describing which part
   * of the action failed (should be null for standard action failures). When appropriate (we use
   * some heuristics to decide), produces an abbreviated message incorporating just the termination
   * status if available.
   *
   * @param exception initial ExecException
   * @param actionSubtask additional information about the action
   * @param action failed action
   * @return ActionExecutionException object describing the action failure
   */
  public static ActionExecutionException fromExecException(
      ExecException exception, @Nullable String actionSubtask, Action action) {
    // Message from ActionExecutionException will be prepended with action.describe() where
    // necessary: because not all ActionExecutionExceptions come from this codepath, it is safer
    // for consumers to manually prepend. We still put action.describe() in the failure detail
    // message argument.
    String message =
        (actionSubtask == null ? "" : actionSubtask + ": ")
            + exception.getMessageForActionExecutionException();

    DetailedExitCode code =
        DetailedExitCode.of(exception.getFailureDetail(action.describe() + " failed: " + message));
    if (exception instanceof LostInputsExecException lostInputsExecException) {
      return lostInputsExecException.fromExecException(message, action, code);
    }

    return fromExecException(exception, message, action, code);
  }

  public static ActionExecutionException fromExecException(
      ExecException exception, String message, Action action, DetailedExitCode code) {
    return new ActionExecutionException(
        message, exception, action, exception.isCatastrophic(), code);
  }

  /** Returns the action that failed. */
  public ActionAnalysisMetadata getAction() {
    return action;
  }

  /**
   * Return the root causes that should be reported. Usually the owner of the action, but it can be
   * the label of a missing artifact.
   */
  public NestedSet<Cause> getRootCauses() {
    return rootCauses;
  }

  /**
   * Returns the location of the owner of this action.  May be null.
   */
  public Location getLocation() {
    return action.getOwner().getLocation();
  }

  /**
   * Catastrophic exceptions should stop builds, even if --keep_going.
   */
  public boolean isCatastrophe() {
    return catastrophe;
  }

  /**
   * Returns the exit code to return from this Bazel invocation because of this action execution
   * failure.
   */
  public ExitCode getExitCode() {
    return detailedExitCode.getExitCode();
  }

  @Override
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }

  /**
   * Returns true if the error should be shown.
   */
  public boolean showError() {
    return getMessage() != null;
  }
}
