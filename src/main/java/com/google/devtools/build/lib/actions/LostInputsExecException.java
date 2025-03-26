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
import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.rewinding.LostInputOwners;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.Optional;
import javax.annotation.Nullable;

/**
 * An {@link ExecException} thrown when an action fails to execute because one or more of its inputs
 * was lost. In some cases, Bazel may know how to fix this on its own.
 */
public final class LostInputsExecException extends ExecException {

  /** Maps lost input digests to their {@link ActionInput}. */
  private final ImmutableMap<String, ActionInput> lostInputs;

  /**
   * Optional mapping of lost inputs to their owning expansion artifacts (tree artifacts, filesets,
   * runfiles).
   *
   * <p>If present, the mapping must be complete. Spawn runners should only provide this if they can
   * do so correctly and efficiently. If not provided, {@link
   * com.google.devtools.build.lib.skyframe.rewinding.ActionRewindStrategy} will calculate the
   * ownership mappings - the only benefit of providing them here is the performance benefit of
   * skipping that step.
   */
  private final Optional<LostInputOwners> owners;

  public LostInputsExecException(ImmutableMap<String, ActionInput> lostInputs) {
    this(lostInputs, /* owners= */ Optional.empty(), /* cause= */ null);
  }

  public LostInputsExecException(
      ImmutableMap<String, ActionInput> lostInputs, LostInputOwners owners) {
    this(lostInputs, Optional.of(owners), /* cause= */ null);
  }

  public LostInputsExecException(
      ImmutableMap<String, ActionInput> lostInputs,
      Optional<LostInputOwners> owners,
      @Nullable Throwable cause) {
    super("lost inputs with digests: " + String.join(",", lostInputs.keySet()), cause);
    checkArgument(!lostInputs.isEmpty(), "No inputs were lost");
    this.lostInputs = lostInputs;
    this.owners = checkNotNull(owners);
  }

  @VisibleForTesting
  public ImmutableMap<String, ActionInput> getLostInputs() {
    return lostInputs;
  }

  @VisibleForTesting
  public Optional<LostInputOwners> getOwners() {
    return owners;
  }

  ActionExecutionException fromExecException(String message, Action action, DetailedExitCode code) {
    return new LostInputsActionExecutionException(
        message, lostInputs, owners, action, /* cause= */ this, code);
  }

  @Override
  protected FailureDetail getFailureDetail(String message) {
    return FailureDetail.newBuilder()
        .setExecution(Execution.newBuilder().setCode(Code.ACTION_INPUT_LOST))
        .setMessage(message)
        .build();
  }

  public LostInputsExecException combine(LostInputsExecException other) {
    // Key collisions are expected when the two sources of the original exceptions shared knowledge
    // of what was lost. For example, a SpawnRunner may discover a lost input and look it up in an
    // action filesystem in which it's also lost. The SpawnRunner and the filesystem may then each
    // throw a LostInputsExecException with the same information.
    //
    // In the case of shared artifacts, it is currently important that other's lost inputs take
    // precedence over this exception's lost inputs.
    // TODO: b/321128298 - This is untested and way too delicate. Improve it.
    ImmutableMap<String, ActionInput> combinedLostInputs =
        ImmutableMap.<String, ActionInput>builder()
            .putAll(lostInputs)
            .putAll(other.lostInputs)
            .buildKeepingLast();
    LostInputsExecException combined =
        new LostInputsExecException(
            combinedLostInputs, /* owners= */ Optional.empty(), /* cause= */ this);
    combined.addSuppressed(other);
    return combined;
  }

  public void combineAndThrow(LostInputsExecException other) throws LostInputsExecException {
    throw combine(other);
  }
}
