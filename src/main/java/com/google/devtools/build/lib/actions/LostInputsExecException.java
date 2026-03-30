// Copyright 2018 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import javax.annotation.Nullable;

/**
 * An {@link ExecException} thrown when an action fails to execute because one or more of its inputs
 * was lost. In some cases, Bazel may know how to fix this on its own.
 */
public final class LostInputsExecException extends ExecException {

  /** Maps lost input digests to their {@link ActionInput}s. */
  private final ImmutableSetMultimap<String, ActionInput> lostInputs;

  public LostInputsExecException(ImmutableSetMultimap<String, ActionInput> lostInputs) {
    this(lostInputs, /* cause= */ null);
  }

  public LostInputsExecException(
      ImmutableSetMultimap<String, ActionInput> lostInputs, @Nullable Throwable cause) {
    super("lost inputs with digests: " + String.join(",", lostInputs.keySet()), cause);
    checkArgument(!lostInputs.isEmpty(), "No inputs were lost");
    this.lostInputs = lostInputs;
  }

  @VisibleForTesting
  public ImmutableSetMultimap<String, ActionInput> getLostInputs() {
    return lostInputs;
  }

  ActionExecutionException fromExecException(String message, Action action, DetailedExitCode code) {
    return new LostInputsActionExecutionException(
        message, lostInputs, action, /* cause= */ this, code);
  }

  @Override
  protected FailureDetail getFailureDetail(String message) {
    return FailureDetail.newBuilder()
        .setExecution(Execution.newBuilder().setCode(Code.ACTION_INPUT_LOST))
        .setMessage(message)
        .build();
  }

  public LostInputsExecException combine(LostInputsExecException other) {
    ImmutableSetMultimap<String, ActionInput> combinedLostInputs =
        ImmutableSetMultimap.<String, ActionInput>builder()
            .putAll(lostInputs)
            .putAll(other.lostInputs)
            .build();
    LostInputsExecException combined =
        new LostInputsExecException(combinedLostInputs, /* cause= */ this);
    combined.addSuppressed(other);
    return combined;
  }

  public void combineAndThrow(LostInputsExecException other) throws LostInputsExecException {
    throw combine(other);
  }
}
