// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.extra;

import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.analysis.actions.ProtoDeterministicWriter;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Spawn;
import com.google.devtools.build.lib.server.FailureDetails.Spawn.Code;
import com.google.devtools.build.lib.util.Fingerprint;
import javax.annotation.Nullable;

/**
 * Requests extra action info from shadowed action and writes it, in protocol buffer format, to an
 * .xa file for use by an extra action. This can only be done at execution time because actions may
 * store information only known at execution time into the protocol buffer.
 */
@Immutable // if shadowedAction is immutable
public final class ExtraActionInfoFileWriteAction extends AbstractFileWriteAction {
  private static final String UUID = "1759f81d-e72e-477d-b182-c4532bdbaeeb";

  private final Action shadowedAction;

  ExtraActionInfoFileWriteAction(ActionOwner owner, Artifact primaryOutput, Action shadowedAction) {
    super(
        owner,
        shadowedAction.discoversInputs()
            ? NestedSetBuilder.<Artifact>stableOrder().addAll(shadowedAction.getOutputs()).build()
            : NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        primaryOutput);

    this.shadowedAction = Preconditions.checkNotNull(shadowedAction, primaryOutput);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws ExecException, InterruptedException {
    try {
      return new ProtoDeterministicWriter(
          shadowedAction.getExtraActionInfo(ctx.getActionKeyContext()).build());
    } catch (CommandLineExpansionException e) {
      throw new UserExecException(
          e,
          FailureDetail.newBuilder()
              .setMessage(Strings.nullToEmpty(e.getMessage()))
              .setSpawn(Spawn.newBuilder().setCode(Code.COMMAND_LINE_EXPANSION_FAILURE))
              .build());
    }
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, InterruptedException {
    fp.addString(UUID);
    fp.addString(shadowedAction.getKey(actionKeyContext, artifactExpander));
    fp.addBytes(shadowedAction.getExtraActionInfo(actionKeyContext).build().toByteArray());
  }
}
