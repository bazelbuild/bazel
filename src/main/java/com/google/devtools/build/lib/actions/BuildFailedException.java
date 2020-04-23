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

import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import javax.annotation.Nullable;

/**
 * This exception gets thrown if there were errors during the execution phase of the build.
 *
 * <p>The argument to the constructor may be null if the thrower has already printed an error
 * message; in this case, no error message should be printed by the catcher. (Typically, this
 * happens when the builder is unsuccessful and {@code --keep_going} was specified. This error
 * corresponds to one or more actions failing, but since those actions' failures will be reported
 * separately, the exception carries no message and is just used for control flow.)
 *
 * <p>This exception typically leads to Bazel termination with exit code {@link
 * ExitCode#BUILD_FAILURE}. However, if a more specific exit code is appropriate, it can be
 * propagated by specifying the exit code to the constructor using a {@link DetailedExitCode}.
 */
@ThreadSafe
public class BuildFailedException extends Exception {
  private final boolean catastrophic;
  @Nullable private final Action action;
  private final NestedSet<Cause> rootCauses;
  private final boolean errorAlreadyShown;
  private final DetailedExitCode detailedExitCode;

  public BuildFailedException() {
    this(null);
  }

  public BuildFailedException(String message) {
    this(
        message,
        /*catastrophic=*/ false,
        /*action=*/ null,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /*errorAlreadyShown=*/ false,
        DetailedExitCode.justExitCode(ExitCode.BUILD_FAILURE));
  }

  public BuildFailedException(String message, DetailedExitCode detailedExitCode) {
    this(
        message,
        /*catastrophic=*/ false,
        /*action=*/ null,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /*errorAlreadyShown=*/ false,
        detailedExitCode);
  }

  public BuildFailedException(String message, boolean catastrophic) {
    this(
        message,
        catastrophic,
        /*action=*/ null,
        NestedSetBuilder.emptySet(Order.STABLE_ORDER),
        /*errorAlreadyShown=*/ false,
        DetailedExitCode.justExitCode(ExitCode.BUILD_FAILURE));
  }

  public BuildFailedException(
      String message,
      boolean catastrophic,
      Action action,
      NestedSet<Cause> rootCauses,
      boolean errorAlreadyShown,
      DetailedExitCode detailedExitCode) {
    super(message);
    this.catastrophic = catastrophic;
    this.rootCauses = rootCauses;
    this.action = action;
    this.errorAlreadyShown = errorAlreadyShown;
    this.detailedExitCode = checkNotNull(detailedExitCode);
  }

  public boolean isCatastrophic() {
    return catastrophic;
  }

  @Nullable
  public Action getAction() {
    return action;
  }

  public NestedSet<Cause> getRootCauses() {
    return rootCauses;
  }

  public boolean isErrorAlreadyShown() {
    return errorAlreadyShown || getMessage() == null;
  }

  /**
   * Returns the pair of {@link ExitCode} and optional {@link FailureDetail} to return from this
   * Bazel invocation.
   */
  public DetailedExitCode getDetailedExitCode() {
    return detailedExitCode;
  }
}
