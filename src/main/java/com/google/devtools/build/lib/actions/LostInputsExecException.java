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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.Execution.Code;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.util.DetailedExitCode;
import java.util.HashMap;
import java.util.Map;

/**
 * An {@link ExecException} thrown when an action fails to execute because one or more of its inputs
 * was lost. In some cases, Bazel may know how to fix this on its own.
 */
public class LostInputsExecException extends ExecException {

  /** Maps lost input digests to their ActionInputs. */
  private final ImmutableMap<String, ActionInput> lostInputs;

  private final ActionInputDepOwners owners;

  public LostInputsExecException(
      ImmutableMap<String, ActionInput> lostInputs, ActionInputDepOwners owners) {
    super(getMessage(lostInputs));
    this.lostInputs = lostInputs;
    this.owners = owners;
  }

  public LostInputsExecException(
      ImmutableMap<String, ActionInput> lostInputs, ActionInputDepOwners owners, Throwable cause) {
    super(getMessage(lostInputs), cause);
    this.lostInputs = lostInputs;
    this.owners = owners;
  }

  private static String getMessage(ImmutableMap<String, ActionInput> lostInputs) {
    return "lost inputs with digests: " + Joiner.on(",").join(lostInputs.keySet());
  }

  @VisibleForTesting
  public ImmutableMap<String, ActionInput> getLostInputs() {
    return lostInputs;
  }

  @VisibleForTesting
  public ActionInputDepOwners getOwners() {
    return owners;
  }

  @Override
  protected ActionExecutionException toActionExecutionException(
      String message, Action action, DetailedExitCode code) {
    return new LostInputsActionExecutionException(
        message, lostInputs, owners, action, /*cause=*/ this, code);
  }

  @Override
  protected FailureDetail getFailureDetail(String message) {
    return FailureDetail.newBuilder()
        .setExecution(Execution.newBuilder().setCode(Code.ACTION_INPUT_LOST))
        .setMessage(message)
        .build();
  }

  public void combineAndThrow(LostInputsExecException other) throws LostInputsExecException {
    // This uses a HashMap when merging the two lostInputs maps because key collisions are expected.
    // In contrast, ImmutableMap.Builder treats collisions as errors. Collisions will happen when
    // the two sources of the original exceptions shared knowledge of what was lost. For example,
    // a SpawnRunner may discover a lost input and look it up in an action filesystem in which it's
    // also lost. The SpawnRunner and the filesystem may then each throw a LostInputsExecException
    // with the same information.
    Map<String, ActionInput> map = new HashMap<>();
    map.putAll(lostInputs);
    map.putAll(other.lostInputs);
    LostInputsExecException combined =
        new LostInputsExecException(
            ImmutableMap.copyOf(map), new MergedActionInputDepOwners(owners, other.owners), this);
    combined.addSuppressed(other);
    throw combined;
  }

  private static class MergedActionInputDepOwners implements ActionInputDepOwners {

    private final ActionInputDepOwners left;
    private final ActionInputDepOwners right;

    MergedActionInputDepOwners(ActionInputDepOwners left, ActionInputDepOwners right) {
      this.left = left;
      this.right = right;
    }

    @Override
    public ImmutableSet<Artifact> getDepOwners(ActionInput input) {
      return ImmutableSet.<Artifact>builder()
          .addAll(left.getDepOwners(input))
          .addAll(right.getDepOwners(input))
          .build();
    }
  }
}
